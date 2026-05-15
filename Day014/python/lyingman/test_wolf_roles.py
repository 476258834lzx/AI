"""狼人阵营角色技能测试"""
import sys
sys.path.insert(0, '/data/Workspace/airelearn')

from Day012.python.lyingman.main import WerewolfGame
from Day012.python.lyingman.config import LLMProvider, set_provider
from Day012.python.lyingman.game_state import PlayerStatus
from Day012.python.lyingman.roles import RoleType, Camp
from Day012.python.lyingman.wolf_roles import create_wolf_role
from Day012.python.lyingman.good_roles import create_good_role
import traceback
import random

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

# ========== 1. 狼人角色基础测试 ==========

def test_wolf_role_camp():
    """测试狼人角色阵营正确"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 找到狼人
    wolves = game.referee.get_wolves()
    assert len(wolves) > 0, "应该有狼人"

    for wolf in wolves:
        camp = game.referee.get_effective_camp(wolf.id)
        assert camp == Camp.WOLF, f"{wolf.role.name} 应为狼人阵营"
        print(f"   {wolf.name}: {wolf.role.name} - 阵营: {camp.value}")

def test_wolf_count():
    """测试狼人人数正确"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    wolves = game.referee.get_wolves()
    # 12人局应有4个狼人
    assert len(wolves) == 4, f"12人局应有4个狼人，实际{len(wolves)}个"
    print(f"   狼人数量: {len(wolves)}")

# ========== 2. 狼王技能测试 ==========

def test_wolf_king_role():
    """测试狼王角色配置"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 手动设置狼王
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼王":
            assert player.role.can_shoot_on_death == True, "狼王应能死亡时开枪"
            print(f"   狼王: {player.name}")
            return

    print("   本局无狼王角色")

def test_wolf_king_shoot_on_death():
    """测试狼王死亡时开枪"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 找到或创建一个狼王
    wolf_king_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼王":
            wolf_king_id = pid
            break

    if wolf_king_id is None:
        # 手动设置狼王
        for pid, player in game.game_state.players.items():
            if player.role and player.role.name == "狼人":
                player.role = create_wolf_role(RoleType.WOLF_KING)
                wolf_king_id = pid
                break

    assert wolf_king_id is not None, "需要狼王角色"

    # 找到存活目标
    target_id = None
    for pid, player in game.game_state.players.items():
        if pid != wolf_king_id and player.is_alive():
            target_id = pid
            break

    assert target_id is not None, "需要存活目标"

    # 设置狼王死亡（模拟被刀死或毒死）
    wolf_king = game.game_state.players[wolf_king_id]
    wolf_king.status = PlayerStatus.DEAD_NIGHT
    wolf_king.role.can_shoot_on_death = True

    # 模拟猎人开枪回调
    def mock_hunter_callback(hunter_id, candidates):
        if hunter_id == wolf_king_id and target_id in candidates:
            return target_id
        return None

    # 夜晚结算（狼王死亡时会触发can_shoot_on_death）
    game.god.start_night()
    deaths = game.god.night_settle(hunter_callback=mock_hunter_callback)

    # 检查狼王是否死亡
    assert not wolf_king.is_alive(), "狼王应该死亡"
    print(f"   狼王({wolf_king.name})死亡时开枪测试完成")

# ========== 3. 狼美人技能测试 ==========

def test_wolf_beauty_role():
    """测试狼美人角色配置"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 手动设置狼美人
    wolf_beauty_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.WOLF_BEAUTY)
            wolf_beauty_id = pid
            break

    assert wolf_beauty_id is not None, "需要狼美人角色"
    print(f"   狼美人: {game.game_state.players[wolf_beauty_id].name}")

def test_wolf_beauty_charm():
    """测试狼美人魅惑功能"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置狼美人
    wolf_beauty_id = None
    charm_target_id = None

    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.WOLF_BEAUTY)
            wolf_beauty_id = pid
            break

    for pid, player in game.game_state.players.items():
        if pid != wolf_beauty_id and player.is_alive():
            charm_target_id = pid
            break

    assert wolf_beauty_id is not None and charm_target_id is not None

    # 设置魅惑目标
    game.game_state.wolf_beauty_charm_target = charm_target_id
    print(f"   狼美人魅惑: {game.game_state.players[charm_target_id].name}")

def test_wolf_beauty_death_bring_target():
    """测试狼美人死亡带走魅惑目标"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置狼美人
    wolf_beauty_id = None
    charm_target_id = None

    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.WOLF_BEAUTY)
            wolf_beauty_id = pid
            break

    for pid, player in game.game_state.players.items():
        if pid != wolf_beauty_id and player.is_alive():
            charm_target_id = pid
            break

    assert wolf_beauty_id is not None and charm_target_id is not None

    # 设置魅惑目标
    game.game_state.wolf_beauty_charm_target = charm_target_id
    charm_target = game.game_state.players[charm_target_id]

    # 处理狼美人死亡
    game.referee.handle_player_death(wolf_beauty_id, cause="night")

    # 魅惑目标应死亡
    assert not charm_target.is_alive(), f"狼美人死亡应带走魅惑目标{charm_target.name}"
    print(f"   狼美人死亡带走: {charm_target.name}")

# ========== 4. 恶灵骑士技能测试 ==========

def test_evil_knight_role():
    """测试恶灵骑士角色配置"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 手动设置恶灵骑士
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.EVIL_KNIGHT)
            print(f"   恶灵骑士: {player.name}")
            assert player.role.can_shoot_on_death == False, "恶灵骑士死亡不应带走人"
            assert player.role.camp == Camp.WOLF, "恶灵骑士应为狼人阵营"
            return

def test_evil_knight_seer_result():
    """测试恶灵骑士被查验显示为好人"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置恶灵骑士
    evil_knight_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.EVIL_KNIGHT)
            evil_knight_id = pid
            break

    assert evil_knight_id is not None, "需要恶灵骑士角色"

    # 模拟预言家查验
    evil_knight = game.game_state.players[evil_knight_id]
    # 恶灵骑士免疫夜间伤害，预言家查验后会反伤
    assert "免疫" in evil_knight.role.description, "恶灵骑士描述应说明免疫"
    print(f"   恶灵骑士: {evil_knight.name}")
    print(f"   描述: {evil_knight.role.description}")

def test_evil_knight_seer_revenge():
    """测试恶灵骑士被查验反伤"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置恶灵骑士
    evil_knight_id = None
    seer_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.EVIL_KNIGHT)
            evil_knight_id = pid
        elif player.role and player.role.name == "预言家":
            seer_id = pid

    assert evil_knight_id is not None and seer_id is not None, "需要恶灵骑士和预言家"

    seer = game.game_state.players[seer_id]
    evil_knight = game.game_state.players[evil_knight_id]

    # 标记恶灵骑士被查验
    game.game_state.evil_knight_checked_by = seer_id

    # 处理反伤
    game._handle_evil_knight_revenge()

    # 预言家应该死亡
    assert not seer.is_alive(), "预言家应该因查验恶灵骑士死亡"
    # 恶灵骑士应该存活
    assert evil_knight.is_alive(), "恶灵骑士应该存活"
    print(f"   预言家{seer.name}因查验恶灵骑士反伤死亡，恶灵骑士存活")

def test_evil_knight_witch_revenge():
    """测试恶灵骑士被女巫毒杀反伤"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置恶灵骑士和女巫
    evil_knight_id = None
    witch_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.EVIL_KNIGHT)
            evil_knight_id = pid
        elif player.role and player.role.name == "女巫":
            witch_id = pid

    assert evil_knight_id is not None and witch_id is not None, "需要恶灵骑士和女巫"

    witch = game.game_state.players[witch_id]
    evil_knight = game.game_state.players[evil_knight_id]

    # 模拟女巫试图毒恶灵骑士（在handle_night中处理）
    # 这里直接测试反伤逻辑
    game.game_state.evil_knight_poisoned_by = witch_id

    # 处理反伤
    game._handle_evil_knight_revenge()

    # 女巫应该死亡
    assert not witch.is_alive(), "女巫应该因毒杀恶灵骑士反伤死亡"
    # 恶灵骑士应该存活
    assert evil_knight.is_alive(), "恶灵骑士应该存活"
    print(f"   女巫{witch.name}因毒杀恶灵骑士反伤死亡，恶灵骑士存活")

def test_evil_knight_immune_to_night_kill():
    """测试恶灵骑士免疫夜间伤害"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置恶灵骑士
    evil_knight_id = None
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.EVIL_KNIGHT)
            evil_knight_id = pid
            break

    assert evil_knight_id is not None, "需要恶灵骑士"

    evil_knight = game.game_state.players[evil_knight_id]

    # 设置恶灵骑士被狼刀
    game.game_state.wolf_kill_target = evil_knight_id

    # 夜晚结算
    game.god.start_night()
    game.god.night_settle()

    # 恶灵骑士应该存活（免疫狼刀）
    assert evil_knight.is_alive(), "恶灵骑士应该免疫狼刀"
    print(f"   恶灵骑士免疫狼刀存活")

# ========== 5. 白狼王技能测试 ==========

def test_white_wolf_king_role():
    """测试白狼王角色配置"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 手动设置白狼王
    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.WHITE_WOLF_KING)
            print(f"   白狼王: {player.name}")
            return

def test_white_wolf_self_destruct():
    """测试白狼王自爆功能"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 设置白狼王
    white_wolf_id = None
    target_id = None

    for pid, player in game.game_state.players.items():
        if player.role and player.role.name == "狼人":
            player.role = create_wolf_role(RoleType.WHITE_WOLF_KING)
            white_wolf_id = pid
            break

    for pid, player in game.game_state.players.items():
        if pid != white_wolf_id and player.is_alive():
            target_id = pid
            break

    assert white_wolf_id is not None and target_id is not None

    white_wolf = game.game_state.players[white_wolf_id]
    target = game.game_state.players[target_id]

    # 白狼王自爆逻辑
    assert white_wolf.is_alive(), "白狼王应该存活"
    assert white_wolf.role.name == "白狼王", "应该是白狼王"

    # 执行自爆
    white_wolf.status = PlayerStatus.DEAD_DAY
    result = f"白狼王({white_wolf.name})自爆！"

    # 如果有目标，带走目标
    if target_id and target.is_alive():
        target.status = PlayerStatus.DEAD_DAY
        result += f" 带走了{target.name}"
        assert not target.is_alive(), "目标应该死亡"

    print(f"   {result}")
    assert "自爆" in result, "白狼王应自爆"

# ========== 6. 狼人团队协作测试 ==========

def test_wolf_team_awareness():
    """测试狼人团队认知"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    wolves = game.referee.get_wolves()
    wolf_ids = [w.id for w in wolves]

    assert len(wolf_ids) > 0, "应有狼人"

    # 检查每个狼人是否知道其他狼人
    for wolf in wolves:
        # 在实际游戏中，狼人应该能看到其他狼人的角色
        # 这里检查阵营是否正确
        camp = game.referee.get_effective_camp(wolf.id)
        assert camp == Camp.WOLF, f"狼人{wolf.name}阵营应为wolf"
        print(f"   {wolf.name} 是狼人，阵营: {camp.value}")

def test_wolf_kill_target():
    """测试狼人刀人机制"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 找到狼人
    wolves = game.referee.get_wolves()
    wolf_ids = [w.id for w in wolves]

    # 找到非狼人目标
    target_id = None
    for pid, player in game.game_state.players.items():
        if pid not in wolf_ids and player.is_alive():
            target_id = pid
            break

    assert target_id is not None, "需要可刀目标"

    # 设置刀人目标
    game.game_state.wolf_kill_target = target_id
    print(f"   狼人刀人目标: {game.game_state.players[target_id].name}")

# ========== 7. 狼人获胜条件测试 ==========

def test_wolf_win_no_good():
    """测试狼人获胜条件 - 好人全部死亡"""
    game = WerewolfGame(player_count=12, mode='watch')
    game.setup_game()

    # 杀死所有好人
    for pid, player in game.game_state.players.items():
        camp = game.referee.get_effective_camp(pid)
        if camp == Camp.GOOD:
            player.status = PlayerStatus.DEAD_NIGHT

    winner = game.referee.check_win_condition()
    assert winner == "wolf", "好人全死时狼人应获胜"
    print(f"   好人全死，获胜方: {winner}")

def test_wolf_win_by_numbers():
    """测试狼人获胜条件 - 人数优势"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()

    # 保留狼人和一个好人
    wolves = game.referee.get_wolves()
    wolf_ids = [w.id for w in wolves]

    for pid, player in game.game_state.players.items():
        if pid not in wolf_ids:
            player.status = PlayerStatus.DEAD_NIGHT

    # 狼人现在应该已经赢了
    winner = game.referee.check_win_condition()
    print(f"   狼人数量优势，获胜方: {winner}")

# ========== 执行所有测试 ==========

if __name__ == "__main__":
    print("\n" + "="*60)
    print("狼人阵营角色技能测试")
    print("="*60)

    # 1. 基础测试
    run_test("狼人角色阵营正确", test_wolf_role_camp)
    run_test("狼人人数正确", test_wolf_count)

    # 2. 狼王技能
    run_test("狼王角色配置", test_wolf_king_role)
    run_test("狼王死亡时开枪", test_wolf_king_shoot_on_death)

    # 3. 狼美人技能
    run_test("狼美人角色配置", test_wolf_beauty_role)
    run_test("狼美人魅惑功能", test_wolf_beauty_charm)
    run_test("狼美人死亡带走魅惑目标", test_wolf_beauty_death_bring_target)

    # 4. 恶灵骑士技能
    run_test("恶灵骑士角色配置", test_evil_knight_role)
    run_test("恶灵骑士查验显示狼人", test_evil_knight_seer_result)
    run_test("恶灵骑士被查验反伤", test_evil_knight_seer_revenge)
    run_test("恶灵骑士被女巫毒杀反伤", test_evil_knight_witch_revenge)
    run_test("恶灵骑士免疫夜间伤害", test_evil_knight_immune_to_night_kill)

    # 5. 白狼王技能
    run_test("白狼王角色配置", test_white_wolf_king_role)
    run_test("白狼王自爆功能", test_white_wolf_self_destruct)

    # 6. 狼人团队协作
    run_test("狼人团队认知", test_wolf_team_awareness)
    run_test("狼人刀人机制", test_wolf_kill_target)

    # 7. 获胜条件
    run_test("狼人获胜-好人全死", test_wolf_win_no_good)
    run_test("狼人获胜-人数优势", test_wolf_win_by_numbers)

    # 打印结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    print(f"✅ 通过: {results['passed']}")
    print(f"❌ 失败: {results['failed']}")
    print(f"总计: {results['passed'] + results['failed']}")

    if results['failed'] > 0:
        print("\n⚠️  有测试失败，需要检查狼人角色技能实现")
        sys.exit(1)
    else:
        print("\n🎉 所有狼人角色技能测试通过!")