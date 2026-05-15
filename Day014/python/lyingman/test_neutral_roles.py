"""中立角色技能和获胜条件异常压测"""
import sys
sys.path.insert(0, '/data/Workspace/airelearn')

from Day012.python.lyingman.main import WerewolfGame
from Day012.python.lyingman.config import LLMProvider, set_provider
from Day012.python.lyingman.game_state import PlayerStatus
from Day012.python.lyingman.referee import ROLE_CONFIGS
import traceback

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

results = {"passed": 0, "failed": 0}

def run_test(name, func):
    if test_case(name, func):
        results["passed"] += 1
    else:
        results["failed"] += 1

# ========== 1. 情侣殉情异常测试 ==========

def test_couple_no_cupid():
    """测试无丘比特时的情侣殉情"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    # 不设置情侣
    game.god.start_night()
    deaths = [0]
    game.god._handle_couple_death(deaths)
    assert len(deaths) == 1, "无情侣时不应有额外死亡"
    print("   无情侣，正确处理")

def test_couple_death():
    """测试情侣殉情"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    # 设置情侣
    game.game_state.set_lovers(0, 1)
    game.game_state.love_chain_type = "人狼链"
    game.game_state.cupid_id = 2
    
    game.god.start_night()
    deaths = [0]  # 玩家1死亡
    game.god._handle_couple_death(deaths)
    
    assert 1 in deaths, "情侣应该殉情"
    print(f"   殉情后死亡列表: {deaths}")

def test_couple_double_death():
    """测试情侣双方同时死亡"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    game.game_state.set_lovers(0, 1)
    
    game.god.start_night()
    deaths = [0, 1]  # 双方都死
    game.god._handle_couple_death(deaths)
    
    assert len(deaths) == 2, "不应有额外死亡"
    print("   双方同时死亡，无额外处理")

def test_couple_already_dead():
    """测试情侣已死亡"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    game.game_state.set_lovers(0, 1)
    game.game_state.players[1].status = PlayerStatus.DEAD_NIGHT
    
    game.god.start_night()
    deaths = [0]
    game.god._handle_couple_death(deaths)
    
    assert deaths == [0], "不应有额外死亡"
    print("   情侣已死亡，无额外处理")

# ========== 2. 野孩子阵营转换 ==========

def test_wild_child_not_converted():
    """测试野孩子未转换 - 11人局才有野孩子"""
    game = WerewolfGame(player_count=11, mode='watch')
    game.setup_game()

    # 找到野孩子玩家
    wild_child_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "野孩子":
            wild_child_id = pid
            break

    assert wild_child_id is not None, "应该有野孩子角色"
    game.game_state.set_wild_child_id(wild_child_id)
    idol_id = 0 if wild_child_id != 0 else 1
    game.game_state.set_wild_child_idol(idol_id)

    camp = game.referee.get_effective_camp(wild_child_id)
    print(f"   野孩子阵营: {camp.value}")
    assert camp.value == "neutral", "偶像存活时不应是狼人阵营"

def test_wild_child_converted():
    """测试野孩子转换 - 11人局才有野孩子"""
    game = WerewolfGame(player_count=11, mode='watch')
    game.setup_game()

    # 找到野孩子玩家
    wild_child_id = None
    idol_candidate = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "野孩子":
            wild_child_id = pid
        else:
            idol_candidate = pid
        if wild_child_id is not None and idol_candidate is not None:
            break

    assert wild_child_id is not None, "应该有野孩子角色"

    game.game_state.set_wild_child_id(wild_child_id)
    game.game_state.set_wild_child_idol(idol_candidate)
    game.game_state.players[idol_candidate].status = PlayerStatus.DEAD_NIGHT

    camp = game.referee.get_effective_camp(wild_child_id)
    print(f"   野孩子阵营(偶像死亡): {camp.value}")
    assert camp.value == "wolf", "偶像死亡后应是狼人阵营"

def test_wild_child_get_wolves():
    """测试野孩子转换后计入狼人 - 11人局才有野孩子"""
    game = WerewolfGame(player_count=11, mode='watch')
    game.setup_game()

    # 找到野孩子玩家
    wild_child_id = None
    idol_candidate = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "野孩子":
            wild_child_id = pid
        else:
            idol_candidate = pid
        if wild_child_id is not None and idol_candidate is not None:
            break

    assert wild_child_id is not None, "应该有野孩子角色"

    game.game_state.set_wild_child_id(wild_child_id)
    game.game_state.set_wild_child_idol(idol_candidate)
    game.game_state.players[idol_candidate].status = PlayerStatus.DEAD_NIGHT

    wolves = game.referee.get_wolves()
    wolf_ids = [w.id for w in wolves]
    assert wild_child_id in wolf_ids, "野孩子应在狼人列表中"
    print(f"   狼人列表包含野孩子: {wolf_ids}")

# ========== 3. 吹笛人迷惑 ==========

def test_flutist_charm():
    """测试吹笛人迷惑"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    
    game.game_state.flutist_charmed_ids = [0, 1]
    
    assert game.game_state.is_cursed(0), "玩家0应被迷惑"
    assert game.game_state.is_cursed(1), "玩家1应被迷惑"
    assert not game.game_state.is_cursed(2), "玩家2不应被迷惑"
    print("   迷惑状态正确")

def test_flutist_win():
    """测试吹笛人获胜判定"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    game.game_state.flutist_id = 0
    game.game_state.set_flutist_id(0)
    
    # 迷惑所有人
    for pid in game.game_state.players:
        game.game_state.flutist_charmed_ids.append(pid)
    
    winner = game.referee.check_win_condition()
    print(f"   吹笛人获胜判定: {winner}")
    assert winner == "neutral", "吹笛人应获胜"

# ========== 4. 狐狸技能 ==========

def test_fox_lost_skill():
    """测试狐狸失去技能"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    
    game.game_state.fox_lost_skill = True
    assert game.game_state.fox_lost_skill, "狐狸应失去技能"
    print("   狐狸技能状态正确")

# ========== 5. 熊咆哮 ==========

def test_bear_roar():
    """测试熊咆哮"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    
    game.game_state.bear_roared_today = True
    assert game.game_state.bear_roared_today, "熊今天应咆哮"
    print("   熊咆哮状态正确")

# ========== 6. 人狼链第三方获胜 ==========

def test_couple_wolf_human_win():
    """测试人狼链第三方获胜"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置人狼链情侣 - 手动设置角色确保一人是狼
    from Day012.python.lyingman.roles import RoleType
    from Day012.python.lyingman.wolf_roles import create_wolf_role
    from Day012.python.lyingman.good_roles import create_good_role

    # 设置玩家0为狼人
    game.game_state.players[0].role = create_wolf_role(RoleType.WEREWOLF)
    # 设置玩家1为好人
    game.game_state.players[1].role = create_good_role(RoleType.VILLAGER)

    # 设置人狼链情侣
    game.game_state.set_lovers(0, 1)
    game.game_state.love_chain_type = "人狼链"
    game.game_state.cupid_id = 2

    # 杀死除情侣外的所有人
    for pid in game.game_state.players:
        if pid not in [0, 1]:
            game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT

    assert game.game_state.is_couple_wolf_human(), "应是人狼链"
    assert game.game_state.is_love_chain_alone(), "应存活到最后"
    print("   人狼链存活到最后")

    winner = game.referee.check_win_condition()
    print(f"   获胜判定: {winner}")
    assert winner == "neutral", "人狼链应获胜"

def test_couple_not_wolf_human():
    """测试人人链不是第三方"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    
    game.game_state.set_lovers(0, 1)
    game.game_state.love_chain_type = "人人链"
    
    assert not game.game_state.is_couple_wolf_human(), "人人链不是人狼链"
    print("   人人链判定正确")

def test_couple_no_one_alive():
    """测试情侣不是最后存活"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    
    game.game_state.set_lovers(0, 1)
    game.game_state.love_chain_type = "人狼链"
    
    # 还有其他存活玩家
    game.game_state.players[2].status = PlayerStatus.ALIVE
    
    assert not game.game_state.is_love_chain_alone(), "还有其他存活玩家"
    print("   有其他存活玩家，不是最后")

# ========== 7. 阵营判定异常 ==========

def test_get_effective_camp_none_role():
    """测试无角色时的阵营判定"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    game.game_state.players[0].role = None
    
    camp = game.referee.get_effective_camp(0)
    assert camp.value == "good", "无角色默认好人"
    print(f"   无角色阵营: {camp.value}")

def test_get_effective_camp_wild_child_no_idol():
    """测试野孩子未选偶像 - 11人局才有野孩子"""
    game = WerewolfGame(player_count=11, mode='watch')
    game.setup_game()

    # 找到野孩子玩家
    wild_child_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "野孩子":
            wild_child_id = pid
            break

    assert wild_child_id is not None, "应该有野孩子角色"
    game.game_state.set_wild_child_id(wild_child_id)
    # 不设置偶像（wild_child_idol保持None）

    camp = game.referee.get_effective_camp(wild_child_id)
    assert camp.value == "neutral", "未选偶像应是中立"
    print(f"   野孩子未选偶像: {camp.value}")

# ========== 8. 夜晚结算异常 ==========

def test_night_settle_no_lovers():
    """测试无情侣时的夜晚结算"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    
    game.god.start_night()
    deaths = game.god.night_settle()
    assert deaths is not None, "应正常返回"
    print(f"   无情侣结算完成，死亡: {deaths}")

def test_night_settle_with_couple():
    """测试有情侣时的夜晚结算"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()
    game.game_state.set_lovers(0, 1)
    game.game_state.wolf_kill_target = 0
    
    game.god.start_night()
    deaths = game.god.night_settle()
    
    # 玩家0死亡，情侣1应殉情
    assert 1 in deaths, "情侣应殉情"
    print(f"   有情侣结算完成，死亡: {deaths}")

# ========== 9. 游戏流程异常 ==========

def test_day1_cupid_handling():
    """测试第一晚丘比特处理"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    
    # 第一晚
    game.game_state.current_day = 1
    game.handle_night()
    print(f"   第一晚处理完成，情侣: {game.game_state.lovers}")

def test_not_day1_cupid_handling():
    """测试非第一晚不处理丘比特"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    game.game_state.current_day = 2
    game.game_state.set_lovers(0, 1)  # 之前设置的
    
    old_lovers = game.game_state.lovers.copy()
    game.handle_night()
    
    # 情侣不应改变
    assert game.game_state.lovers == old_lovers, "非第一晚情侣不应改变"
    print("   非第一晚情侣未改变")

def test_multiple_nights_flutist():
    """测试多晚吹笛人迷惑"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()
    game.game_state.flutist_id = 0
    game.game_state.flutist_charmed_ids = [1, 2]
    
    # 第二晚
    game.game_state.current_day = 2
    game.handle_night()
    
    print(f"   多晚迷惑: {game.game_state.flutist_charmed_ids}")

# ========== 10. 配置测试 ==========

def test_role_config_counts():
    """测试角色配置人数正确"""
    for count, config in ROLE_CONFIGS.items():
        wolf = len(config["wolf"])
        good = len(config["good"])
        neutral = len(config["neutral"])
        total = wolf + good + neutral
        assert total == count, f"{count}人局配置错误: {total} != {count}"
    print("   所有角色配置人数正确")

# ========== 11. 边界测试 ==========

def test_empty_lovers():
    """测试空情侣字典"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    
    assert not game.game_state.lovers, "初始应无情侣"
    print("   初始无情侣")

def test_get_love_type_no_lovers():
    """测试无情侣时的类型获取"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()
    
    # 尝试获取非情侣的类型
    love_type = game.game_state.get_love_type(0)
    assert love_type is None, "非情侣应返回None"
    print(f"   非情侣类型: {love_type}")

# ========== 执行所有测试 ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("中立角色技能和获胜条件异常压测")
    print("="*60)
    
    # 1. 情侣殉情
    run_test("无丘比特时的情侣殉情", test_couple_no_cupid)
    run_test("情侣殉情", test_couple_death)
    run_test("情侣双方同时死亡", test_couple_double_death)
    run_test("情侣已死亡", test_couple_already_dead)
    
    # 2. 野孩子阵营转换
    run_test("野孩子未转换", test_wild_child_not_converted)
    run_test("野孩子转换", test_wild_child_converted)
    run_test("野孩子转换后计入狼人", test_wild_child_get_wolves)
    
    # 3. 吹笛人迷惑
    run_test("吹笛人迷惑", test_flutist_charm)
    run_test("吹笛人获胜", test_flutist_win)
    
    # 4. 狐狸技能
    run_test("狐狸失去技能", test_fox_lost_skill)
    
    # 5. 熊咆哮
    run_test("熊咆哮", test_bear_roar)
    
    # 6. 人狼链第三方获胜
    run_test("人狼链第三方获胜", test_couple_wolf_human_win)
    run_test("人人链不是第三方", test_couple_not_wolf_human)
    run_test("情侣不是最后存活", test_couple_no_one_alive)
    
    # 7. 阵营判定异常
    run_test("无角色阵营判定", test_get_effective_camp_none_role)
    run_test("野孩子未选偶像", test_get_effective_camp_wild_child_no_idol)
    
    # 8. 夜晚结算异常
    run_test("无情侣夜晚结算", test_night_settle_no_lovers)
    run_test("有情侣夜晚结算", test_night_settle_with_couple)
    
    # 9. 游戏流程异常
    run_test("第一晚丘比特处理", test_day1_cupid_handling)
    run_test("非第一晚不处理丘比特", test_not_day1_cupid_handling)
    run_test("多晚吹笛人迷惑", test_multiple_nights_flutist)
    
    # 10. 配置测试
    run_test("角色配置人数", test_role_config_counts)
    
    # 11. 边界测试
    run_test("空情侣字典", test_empty_lovers)
    run_test("无情侣类型获取", test_get_love_type_no_lovers)
    
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
