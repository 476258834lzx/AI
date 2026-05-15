"""狼人杀游戏异常压测脚本"""
import sys
sys.path.insert(0, '/data/Workspace/airelearn')

from Day012.python.lyingman.main import WerewolfGame
from Day012.python.lyingman.config import LLMProvider, set_provider
from Day012.python.lyingman.game_state import PlayerStatus, NightPhase
from Day012.python.lyingman.referee import ROLE_CONFIGS
import traceback
import random

# 设置为vllm
set_provider(LLMProvider.VLLM)

def test_case(name, func):
    """执行测试用例"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print('='*60)
    try:
        func()
        print(f"✅ 通过: {name}")
        return True
    except Exception as e:
        print(f"❌ 异常: {name}")
        print(f"   错误: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False

# ===== 测试计数器 =====
results = {"passed": 0, "failed": 0}

def run_test(name, func):
    if test_case(name, func):
        results["passed"] += 1
    else:
        results["failed"] += 1

# ========== 1. 白天并行投票异常测试 ==========

def test_vote_empty_candidates():
    """测试空候选人列表投票"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()
    player = game.players[0]
    # 传入空列表
    target = player.vote([])
    assert target == -1 or target is None, f"空列表应返回默认值，实际: {target}"
    print(f"   空候选人返回: {target}")

def test_vote_all_dead():
    """测试所有玩家死亡后的投票"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()
    # 杀死所有玩家
    for p in game.game_state.players.values():
        p.status = PlayerStatus.DEAD_NIGHT
    player = game.players[0]
    alive = game.game_state.get_alive_players()
    assert len(alive) == 0, "应该没有存活玩家"
    # 应该返回-1或第一个候选（但候选列表为空）
    target = player.vote([])

def test_parallel_vote_all_dead():
    """测试所有玩家死亡后并行投票"""
    import concurrent.futures
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    # 杀死所有玩家
    for p in game.game_state.players.values():
        p.status = PlayerStatus.DEAD_NIGHT
    alive_ids = []
    # 并行投票
    def vote_player(pid):
        return game.players[pid].vote(alive_ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(vote_player, pid) for pid in range(8)]
        results = [f.result() for f in futures]
    print(f"   并行投票结果: {results}")

def test_vote_self_only():
    """测试只有自己一个候选人"""
    game = WerewolfGame(player_count=2, mode='watch')
    game.setup_game()
    # 杀死其他玩家，只留自己
    for pid, p in game.game_state.players.items():
        if pid != 0:
            p.status = PlayerStatus.DEAD_NIGHT
    player = game.players[0]
    candidates = [0]  # 只有自己
    target = player.vote(candidates)
    assert target == 0, f"只有自己时应投给自己，实际: {target}"

def test_parallel_vote_same_target():
    """测试并行投票都投给同一目标"""
    import concurrent.futures
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    alive_ids = [p.id for p in game.game_state.get_alive_players()]
    
    def vote_player(pid):
        return (pid, game.players[pid].vote(alive_ids))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(vote_player, pid) for pid in alive_ids]
        results = [f.result() for f in futures]
    
    # 统计票数
    vote_counts = {}
    for pid, target in results:
        vote_counts[target] = vote_counts.get(target, 0) + 1
    print(f"   票数统计: {vote_counts}")
    assert game.game_state.players[max(vote_counts, key=vote_counts.get)].vote_count == max(vote_counts.values())

def test_vote_tie():
    """测试平票场景"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()
    # 手动设置票数使其平票
    for pid in range(4):
        game.game_state.players[pid].vote_count = 1
    
    # 处理投票
    game.handle_day_vote()
    print("   平票处理完成")

# ========== 2. 狼人夜间商讨异常测试 ==========

def test_wolf_single():
    """测试只有1个狼人的场景"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    wolves = game._get_wolf_players()
    print(f"   狼人数量: {len(wolves)}")
    if len(wolves) == 1:
        game._handle_wolf_discuss(wolves)
        print(f"   单狼商讨完成，目标: {game.game_state.wolf_kill_target}")

def test_wolf_no_wolves():
    """测试没有狼人的场景"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()
    # 手动移除所有狼人（极端情况）
    wolves = []
    game._handle_wolf_discuss(wolves)
    print("   无狼人场景处理完成")

def test_wolf_all_dead():
    """测试狼人全部死亡后的商讨"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    # 杀死所有狼人
    wolves = game._get_wolf_players()
    for wid in wolves:
        game.game_state.players[wid].status = PlayerStatus.DEAD_NIGHT
    print(f"   狼人死亡后数量: {len(game._get_wolf_players())}")
    game._handle_wolf_discuss([])

def test_wolf_propose_dead_target():
    """测试狼人提议刀死亡玩家"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    wolves = game._get_wolf_players()
    # 标记一个玩家死亡
    game.game_state.players[1].status = PlayerStatus.DEAD_NIGHT
    alive = game.game_state.get_alive_players()
    non_wolf = [p for p in alive if p.id not in wolves]
    print(f"   存活非狼人: {[p.id for p in non_wolf]}")

def test_wolf_consensus_empty():
    """测试狼人无法达成共识"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    wolves = game._get_wolf_players()
    # 狼人商讨完成
    game._handle_wolf_discuss(wolves)
    print(f"   最终刀人目标: {game.game_state.wolf_kill_target}")

# ========== 3. 猎人技能异常测试 ==========

def test_hunter_no_candidates():
    """测试猎人没有可开枪目标"""
    game = WerewolfGame(player_count=2, mode='watch')
    game.setup_game()
    hunter_id = [pid for pid, p in game.game_state.players.items() 
                 if p.role and p.role.name == '猎人']
    if hunter_id:
        # 杀死猎人外的所有玩家
        for pid in game.game_state.players:
            if pid != hunter_id[0]:
                game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT
        # 回调应该返回None
        target = game._hunter_shoot_callback(hunter_id[0], [])
        print(f"   无目标时返回: {target}")

def test_hunter_poisoned():
    """测试猎人被女巫毒死不能开枪"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    hunter_id = [pid for pid, p in game.game_state.players.items() 
                 if p.role and p.role.name == '猎人']
    if hunter_id:
        hid = hunter_id[0]
        # 设置猎人被毒死
        game.game_state.players[hid].is_poisoned = True
        game.game_state.wolf_kill_target = hid
        game.god.start_night()
        deaths = game.god.night_settle(hunter_callback=game._hunter_shoot_callback)
        # 猎人应该死亡但不开枪
        assert hid in deaths, "猎人应该死亡"
        print(f"   被毒死猎人: {deaths}")

def test_hunter_solo():
    """测试场上只有猎人一人"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    hunter_id = [pid for pid, p in game.game_state.players.items() 
                 if p.role and p.role.name == '猎人']
    if hunter_id:
        hid = hunter_id[0]
        # 杀死除猎人外的所有人
        for pid in game.game_state.players:
            if pid != hid:
                game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT
        # 狼人刀猎人
        game.game_state.wolf_kill_target = hid
        game.god.start_night()
        deaths = game.god.night_settle(hunter_callback=game._hunter_shoot_callback)
        print(f"   猎人是最后一人，死亡: {deaths}")

def test_hunter_callback_with_invalid_candidates():
    """测试猎人回调收到无效候选"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    hunter_id = [pid for pid, p in game.game_state.players.items() 
                 if p.role and p.role.name == '猎人']
    if hunter_id:
        hid = hunter_id[0]
        # 传入包含不存在ID的候选
        invalid_candidates = [99, 100, 999]
        target = game._hunter_shoot_callback(hid, invalid_candidates)
        print(f"   无效候选返回: {target}")

# ========== 4. 女巫技能异常测试 ==========

def test_witch_no_potions():
    """测试女巫没有药水"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    witch_id = [pid for pid, p in game.game_state.players.items() 
                if p.role and p.role.name == '女巫']
    if witch_id:
        wid = witch_id[0]
        # 使用解药和毒药
        game.game_state.witch_heal_used = True
        game.game_state.witch_poison_used = True
        print(f"   女巫药水已用完")

def test_witch_heal_dead():
    """测试女巫救死亡玩家"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    # 设置一个玩家死亡
    game.game_state.players[0].status = PlayerStatus.DEAD_NIGHT
    # 女巫尝试救活
    game.game_state.wolf_kill_target = 0
    game.game_state.witch_heal_used = True
    game.game_state.revives.append(0)
    print(f"   女巫尝试救死亡玩家")

def test_witch_poison_self():
    """测试女巫毒自己"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    witch_id = [pid for pid, p in game.game_state.players.items() 
                if p.role and p.role.name == '女巫']
    if witch_id:
        wid = witch_id[0]
        game.game_state.witch_poison_used = False
        # 测试tool逻辑
        from Day012.python.lyingman.tools import witch_poison
        result = witch_poison(game.game_state.__dict__, wid, wid)
        print(f"   女巫毒自己结果: {result}")

def test_witch_double_heal():
    """测试女巫重复救人"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    game.game_state.witch_heal_used = True
    game.game_state.revives.append(0)
    # 再次尝试救人
    game.game_state.revives.append(0)
    print(f"   重复救人，revives: {game.game_state.revives}")

def test_witch_no_wolf_kill():
    """测试女巫阶段狼人没刀人"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    game.game_state.wolf_kill_target = None
    print(f"   狼人没刀人，女巫不救人")

# ========== 5. 游戏配置异常测试 ==========

def test_min_players():
    """测试最小玩家数量"""
    for count in [5, 4, 3, 2, 1]:
        try:
            game = WerewolfGame(player_count=count, mode='watch')
            game.setup_game()
            print(f"   {count}人: 初始化{'成功' if game.game_state else '失败'}")
        except Exception as e:
            print(f"   {count}人: {e}")

def test_invalid_player_config():
    """测试无效玩家配置"""
    for count in [0, -1, 100]:
        try:
            game = WerewolfGame(player_count=count, mode='watch')
            game.setup_game()
        except Exception as e:
            print(f"   {count}人: {type(e).__name__}")

def test_all_roles_dead():
    """测试所有角色死亡"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    # 杀死所有玩家
    for p in game.game_state.players.values():
        p.status = PlayerStatus.DEAD_NIGHT
    # 检查胜负
    winner = game.referee.check_win_condition()
    print(f"   全死亡胜负: {winner}")

# ========== 6. 并发异常测试 ==========

def test_concurrent_night_actions():
    """测试并发夜间动作"""
    import concurrent.futures
    import time
    
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    
    def wolf_discuss(pid):
        wolves = game._get_wolf_players()
        if pid in wolves:
            alive = game.game_state.get_alive_players()
            non_wolf = [p.id for p in alive if p.id not in wolves]
            return game.players[pid].wolf_discuss(wolves, non_wolf, {})
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(wolf_discuss, pid) for pid in range(8)]
        results = [f.result() for f in futures]
    
    print(f"   并发狼人商讨完成")

def test_concurrent_day_speech():
    """测试并发白天发言"""
    import concurrent.futures
    
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    
    def player_speak(pid):
        if game.game_state.players[pid].is_alive():
            return game.players[pid].speak(f"第{pid}号发言")
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(player_speak, pid) for pid in range(8)]
        results = [f.result() for f in futures if f.result()]
    
    print(f"   并发发言完成: {len(results)}条")

# ========== 7. 状态异常测试 ==========

def test_game_state_corruption():
    """测试游戏状态损坏"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    
    # 损坏状态
    game.game_state.wolf_kill_target = 999
    game.game_state.sheriff_id = 999
    game.game_state.current_day = -1
    
    # 继续游戏
    try:
        game.handle_night()
        print(f"   损坏状态处理完成")
    except:
        print(f"   损坏状态导致异常")

def test_role_none():
    """测试角色为None的场景"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    # 手动设置某个玩家角色为None
    game.game_state.players[0].role = None
    role = game.players[0].get_role()
    print(f"   角色为None时get_role: {role}")

# ========== 执行所有测试 ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("狼人杀游戏异常压测")
    print("="*60)
    
    # 1. 白天并行投票异常
    run_test("空候选人列表投票", test_vote_empty_candidates)
    run_test("所有玩家死亡后投票", test_vote_all_dead)
    run_test("所有玩家死亡后并行投票", test_parallel_vote_all_dead)
    run_test("只有自己一个候选人", test_vote_self_only)
    run_test("并行投票都投给同一目标", test_parallel_vote_same_target)
    run_test("平票场景", test_vote_tie)
    
    # 2. 狼人夜间商讨异常
    run_test("单狼场景", test_wolf_single)
    run_test("无狼人场景", test_wolf_no_wolves)
    run_test("狼人全部死亡", test_wolf_all_dead)
    run_test("狼人提议刀死亡目标", test_wolf_propose_dead_target)
    run_test("狼人无法达成共识", test_wolf_consensus_empty)
    
    # 3. 猎人技能异常
    run_test("猎人无开枪目标", test_hunter_no_candidates)
    run_test("猎人被毒死", test_hunter_poisoned)
    run_test("猎人是最后一人", test_hunter_solo)
    run_test("猎人无效候选", test_hunter_callback_with_invalid_candidates)
    
    # 4. 女巫技能异常
    run_test("女巫无药水", test_witch_no_potions)
    run_test("女巫救死亡玩家", test_witch_heal_dead)
    run_test("女巫毒自己", test_witch_poison_self)
    run_test("女巫重复救人", test_witch_double_heal)
    run_test("女巫阶段狼人没刀人", test_witch_no_wolf_kill)
    
    # 5. 游戏配置异常
    run_test("最小玩家数量", test_min_players)
    run_test("无效玩家配置", test_invalid_player_config)
    run_test("所有角色死亡", test_all_roles_dead)
    
    # 6. 并发异常
    run_test("并发夜间动作", test_concurrent_night_actions)
    run_test("并发白天发言", test_concurrent_day_speech)
    
    # 7. 状态异常
    run_test("游戏状态损坏", test_game_state_corruption)
    run_test("角色为None", test_role_none)
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"✅ 通过: {results['passed']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"总计: {results['passed'] + results['failed']}")
    
    if results['failed'] > 0:
        print("\n⚠️  有测试失败，需要检查异常处理机制")
        sys.exit(1)
    else:
        print("\n🎉 所有测试通过!")
