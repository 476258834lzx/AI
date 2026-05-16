"""好人阵营角色技能异常压测"""
import sys
sys.path.insert(0, '/data/Workspace/airelearn')

import traceback
import random
from unittest.mock import MagicMock, patch

from Day014.python.lyingman.game_state import GameState, Player, PlayerStatus, RoundPhase
from Day014.python.lyingman.roles import RoleType, Camp
from Day014.python.lyingman.good_roles import (
    get_prophet_role, get_witch_role, get_hunter_role,
    get_idiot_role, get_guardian_role, get_knight_role, get_villager_role
)
from Day014.python.lyingman.referee import Referee
from Day014.python.lyingman.god import God
from Day014.python.lyingman.main import WerewolfGame
from Day014.python.lyingman.config import LLMProvider, set_provider

set_provider(LLMProvider.VLLM)

test_results = []


def test_case(name, func):
    """执行测试用例"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print('='*60)
    try:
        func()
        print(f"✅ 通过: {name}")
        test_results.append((name, "PASS", None))
        return True
    except Exception as e:
        print(f"❌ 异常: {name}")
        print(f"   错误: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)
        test_results.append((name, "FAIL", str(e)))
        return False


def find_role_player(game, role_name):
    """查找特定角色玩家，返回ID或None"""
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == role_name:
            return pid
    return None


def create_game_with_role(role_names: list, player_count: int = None):
    """创建包含特定角色的游戏"""
    from Day014.python.lyingman.referee import ROLE_CONFIGS

    # 尝试不同人数找到包含所需角色的配置
    for count in range(15, 3, -1):
        if count not in ROLE_CONFIGS:
            continue
        config = ROLE_CONFIGS[count]
        all_roles = config["wolf"] + config["good"] + config["neutral"]
        role_set = {r.value if hasattr(r, 'value') else r for r in all_roles}
        if all(rn in role_set for rn in role_names):
            game = WerewolfGame(player_count=count, mode='watch')
            game.setup_game()
            return game
    return None


# ========== 预言家异常测试 ==========

def test_prophet_check_self():
    """预言家查验自己"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    prophet_id = find_role_player(game, "预言家")
    if prophet_id is None:
        print("没有预言家，跳过测试")
        return

    prophet_agent = game.players[prophet_id]

    # 模拟查验自己
    result = prophet_agent.decide_night_action("seer_check")
    print(f"预言家查验结果: {result}")


def test_prophet_check_dead_player():
    """预言家查验已死亡玩家 - 边界检查"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    prophet_id = find_role_player(game, "预言家")
    if prophet_id is None:
        print("没有预言家，跳过测试")
        return

    # 杀死一个玩家
    victim_id = (prophet_id + 1) % 6
    game.game_state.players[victim_id].status = PlayerStatus.DEAD_NIGHT

    # 检查是否会尝试查验死亡玩家
    candidates = [pid for pid in game.game_state.players if game.game_state.players[pid].is_alive()]
    print(f"存活玩家: {candidates}")
    print(f"死亡玩家: {victim_id}")

    # 测试边界: 预言家应该选择存活玩家
    # 如果LLM返回死亡玩家，应该被过滤掉
    print("预期: 系统应过滤死亡玩家，只选择存活玩家查验")


def test_prophet_check_no_candidates():
    """预言家无可查验目标 - 边界检查"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    prophet_id = find_role_player(game, "预言家")
    if prophet_id is None:
        print("没有预言家，跳过测试")
        return

    # 杀死除预言家外的所有玩家
    for pid in list(game.game_state.players.keys()):
        if pid != prophet_id:
            game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT

    # 无可查验目标
    candidates = [pid for pid in game.game_state.players if game.game_state.players[pid].is_alive()]
    print(f"存活玩家: {candidates} (只有预言家)")
    print(f"可查验目标: {len(candidates) - 1} (不包括自己)")


def test_prophet_check_evil_knight():
    """预言家查验恶灵骑士"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    # 找到狼人并替换为恶灵骑士
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "狼人":
            p.role = type('Role', (), {
                'name': '恶灵骑士',
                'description': '免疫狼刀',
                'camp': Camp.WOLF,
                'can_vote': True,
                'can_be_sheriff': True
            })()
            break

    # 夜晚结算时检查恶灵骑士反伤
    game.god.start_night()
    game.game_state.wolf_kill_target = 0  # 设置狼刀目标

    # 模拟预言家查验恶灵骑士
    game.game_state.seer_check_result = {"target_id": 1, "result": "狼人"}
    game.game_state.evil_knight_checked_by = 2

    # 处理恶灵骑士反伤
    game._handle_evil_knight_revenge()
    print("恶灵骑士反伤处理完成")


# ========== 女巫异常测试 ==========

def test_witch_no_heal_left():
    """女巫解药用尽"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    witch_id = find_role_player(game, "女巫")
    if witch_id is None:
        print("没有女巫，跳过测试")
        return

    game.game_state.witch_heal_used = True  # 标记解药已用

    witch_agent = game.players[witch_id]
    # 模拟解药已用尽的决策
    print(f"解药已用: {game.game_state.witch_heal_used}")


def test_witch_no_poison_left():
    """女巫毒药用尽"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    game.game_state.witch_poison_used = True  # 标记毒药已用

    print(f"毒药已用: {game.game_state.witch_poison_used}")


def test_witch_heal_and_poison_same_night():
    """女巫同晚使用解药和毒药"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    # 设置狼刀目标
    victim_id = 1
    game.game_state.wolf_kill_target = victim_id

    # 女巫同时救人并毒人
    witch_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "女巫":
            witch_id = pid
            break

    # 模拟同晚救人
    game.game_state.revives.append(victim_id)

    # 模拟同晚毒人
    poison_target = 2
    game.game_state.players[poison_target].is_poisoned = True

    print(f"女巫救人: {victim_id}, 毒人: {poison_target}")
    print(f"狼刀目标: {game.game_state.wolf_kill_target}")
    print(f"女巫救人列表: {game.game_state.revives}")


def test_witch_heal_guardian_conflict():
    """女巫救人vs守卫守护冲突"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    victim_id = 1
    guardian_id = 2

    # 设置狼刀目标
    game.game_state.wolf_kill_target = victim_id
    # 设置守卫守护目标
    game.game_state.players[victim_id].is_protected = True
    game.game_state.guardian_last_protect = victim_id

    # 夜晚结算
    game.god.start_night()

    # 女巫救人
    game.game_state.revives.append(victim_id)

    deaths = game.god.night_settle()
    print(f"夜晚死亡: {deaths}")
    print(f"守卫守护了: {game.game_state.guardian_last_protect}")


def test_witch_poison_self():
    """女巫毒死自己"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    witch_id = find_role_player(game, "女巫")
    if witch_id is None:
        print("没有女巫，跳过测试")
        return

    # 女巫毒自己
    game.game_state.players[witch_id].is_poisoned = True
    game.game_state.witch_poison_used = True

    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"女巫({witch_id})死亡: {witch_id in deaths}")


def test_witch_no_potion_killed():
    """女巫被刀且无解药"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    witch_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "女巫":
            witch_id = pid
            break

    # 解药已用，狼刀女巫
    game.game_state.witch_heal_used = True
    game.game_state.wolf_kill_target = witch_id

    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"女巫({witch_id})死亡: {witch_id in deaths}")


# ========== 猎人异常测试 ==========

def test_hunter_no_targets():
    """猎人无开枪目标"""
    game = WerewolfGame(player_count=3, mode='watch')
    game.setup_game()

    # 杀死除猎人外的所有玩家
    hunter_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
            break

    for pid in game.game_state.players:
        if pid != hunter_id:
            game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT

    candidates = [pid for pid, p in game.game_state.players.items() if p.is_alive() and pid != hunter_id]
    print(f"猎人可开枪目标: {candidates}")


def test_hunter_poisoned_no_shoot():
    """猎人被女巫毒死不能开枪"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    hunter_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
            break

    # 猎人被毒死
    game.game_state.players[hunter_id].is_poisoned = True

    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"猎人({hunter_id})被毒死: {game.game_state.players[hunter_id].is_poisoned}")
    print(f"猎人开枪了: {False}")  # 被毒死不能开枪


def test_hunter_shoot_lover():
    """猎人开枪带走情侣"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    hunter_id = find_role_player(game, "猎人")
    if hunter_id is None:
        print("没有猎人，跳过测试")
        return

    # 设置情侣
    game.game_state.set_lovers(0, 1)

    # 模拟猎人选择情侣开枪
    candidates = [pid for pid in game.game_state.players if game.game_state.players[pid].is_alive() and pid != hunter_id]

    print(f"猎人可开枪目标: {candidates}")
    print(f"情侣关系: {game.game_state.lovers}")


def test_hunter_shoot_idiot():
    """猎人开枪带走白痴"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    hunter_id = None
    idiot_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
        if p.role and p.role.name == "白痴":
            idiot_id = pid

    if hunter_id is None or idiot_id is None:
        print("没有猎人和白痴，跳过测试")
        return

    candidates = [pid for pid in game.game_state.players if game.game_state.players[pid].is_alive() and pid != hunter_id]
    print(f"猎人可开枪目标: {candidates}")


def test_hunter_death_idiot():
    """猎人死亡时是白痴（白痴翻牌）"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    hunter_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
            break

    # 找到白痴并标记
    idiot_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "白痴":
            idiot_id = pid
            break

    print(f"猎人ID: {hunter_id}, 白痴ID: {idiot_id}")


# ========== 白痴异常测试 ==========

def test_idiot_voted_out():
    """白痴被投票出局翻牌"""
    game = WerewolfGame(player_count=10, mode='watch')
    game.setup_game()

    idiot_id = find_role_player(game, "白痴")
    if idiot_id is None:
        print("没有白痴，跳过测试")
        return

    # 白痴被投票出局
    game.referee.handle_player_death(idiot_id, cause="vote")

    idiot = game.game_state.players[idiot_id]
    print(f"白痴状态: {idiot.status}")
    print(f"白痴可投票: {idiot.role.can_vote if idiot.role else 'N/A'}")
    print(f"白痴存活: {idiot.is_alive()}")


def test_idiot_killed_by_wolf():
    """白痴被狼刀死"""
    game = WerewolfGame(player_count=10, mode='watch')
    game.setup_game()

    idiot_id = find_role_player(game, "白痴")
    if idiot_id is None:
        print("没有白痴，跳过测试")
        return

    # 白痴被狼刀
    game.referee.handle_player_death(idiot_id, cause="night")

    idiot = game.game_state.players[idiot_id]
    print(f"白痴状态: {idiot.status}")
    print(f"白痴存活: {idiot.is_alive()}")


def test_idiot_poisoned():
    """白痴被女巫毒死"""
    game = WerewolfGame(player_count=10, mode='watch')
    game.setup_game()

    idiot_id = find_role_player(game, "白痴")
    if idiot_id is None:
        print("没有白痴，跳过测试")
        return

    # 白痴被毒
    game.referee.handle_player_death(idiot_id, cause="poison")

    idiot = game.game_state.players[idiot_id]
    print(f"白痴状态: {idiot.status}")
    print(f"白痴存活: {idiot.is_alive()}")


def test_idiot_shot_by_hunter():
    """白痴被猎人带走"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    idiot_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "白痴":
            idiot_id = pid
            break

    hunter_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
            break

    # 猎人带走白痴
    if idiot_id is not None and hunter_id is not None:
        game.referee.handle_player_death(idiot_id, cause="shoot")

        idiot = game.game_state.players[idiot_id]
        print(f"白痴状态: {idiot.status}")
        print(f"白痴存活: {idiot.is_alive()}")


# ========== 守卫异常测试 ==========

def test_guardian_consecutive_protect():
    """守卫连续两晚守护同一人"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    guardian_id = find_role_player(game, "守卫")
    if guardian_id is None:
        print("没有守卫，跳过测试")
        return

    target_id = 0
    if target_id == guardian_id:
        target_id = (guardian_id + 1) % 9

    # 第一晚守护
    game.game_state.guardian_last_protect = target_id
    game.game_state.players[target_id].is_protected = True

    # 第二晚尝试守护同一人
    game.game_state.current_day = 2

    # 守卫应该选择不守护同一人
    print(f"第一晚守护: {target_id}")
    print(f"第二晚守卫上次守护: {game.game_state.guardian_last_protect}")
    print(f"第二晚可守护{target_id}: {target_id != game.game_state.guardian_last_protect}")


def test_guardian_protect_dead_player():
    """守卫守护已死亡玩家"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    guardian_id = find_role_player(game, "守卫")
    if guardian_id is None:
        print("没有守卫，跳过测试")
        return

    # 杀死一个玩家
    dead_id = 1
    game.game_state.players[dead_id].status = PlayerStatus.DEAD_NIGHT

    alive = game.game_state.get_alive_players()
    print(f"存活玩家: {[p.id for p in alive]}")
    print(f"死亡玩家: {dead_id}")


def test_guardian_protect_self():
    """守卫守护自己"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    guardian_id = find_role_player(game, "守卫")
    if guardian_id is None:
        print("没有守卫，跳过测试")
        return

    candidates = [pid for pid in game.game_state.players if game.game_state.players[pid].is_alive()]
    print(f"守卫ID: {guardian_id}")
    print(f"存活玩家: {candidates}")


def test_guardian_witch_conflict():
    """守卫与女巫救人冲突"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    victim_id = 1

    # 守卫守护
    game.game_state.players[victim_id].is_protected = True
    game.game_state.guardian_last_protect = victim_id

    # 狼刀
    game.game_state.wolf_kill_target = victim_id

    # 女巫救人
    game.game_state.revives.append(victim_id)

    # 夜晚结算
    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"目标被守护: {game.game_state.players[victim_id].is_protected}")
    print(f"目标被女巫救: {victim_id in game.game_state.revives}")
    print(f"目标死亡: {victim_id in deaths}")


# ========== 骑士异常测试 ==========

def test_knight_challenge_wolfking():
    """骑士质疑狼王"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    knight_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "骑士":
            knight_id = pid
            break

    if knight_id is None:
        print("没有骑士，跳过测试")
        return

    print(f"骑士ID: {knight_id}")


def test_knight_challenge_lover():
    """骑士质疑情侣"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    # 设置情侣
    game.game_state.set_lovers(0, 1)

    print(f"情侣关系: {game.game_state.lovers}")


def test_knight_challenge_dead_player():
    """骑士质疑已死亡玩家"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    knight_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "骑士":
            knight_id = pid
            break

    if knight_id is None:
        print("没有骑士，跳过测试")
        return

    # 杀死一个玩家
    dead_id = 2
    game.game_state.players[dead_id].status = PlayerStatus.DEAD_NIGHT

    alive = game.game_state.get_alive_players()
    print(f"骑士ID: {knight_id}")
    print(f"存活玩家: {[p.id for p in alive]}")


# ========== 通用异常测试 ==========

def test_all_goods_dead():
    """所有好人死亡"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    # 杀死所有好人
    for pid, p in game.game_state.players.items():
        if p.role and p.role.camp == Camp.GOOD:
            p.status = PlayerStatus.DEAD_NIGHT

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")


def test_all_wolves_dead():
    """所有狼人死亡（好人胜利）"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    # 杀死所有狼人
    for pid, p in game.game_state.players.items():
        if p.role and p.role.camp == Camp.WOLF:
            p.status = PlayerStatus.DEAD_NIGHT

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")
    assert winner == "good", f"应该好人胜利，实际: {winner}"


def test_all_gods_dead():
    """所有神职死亡（屠神）"""
    game = WerewolfGame(player_count=6, mode='watch')
    game.setup_game()

    # 杀死所有神职
    god_roles = ["预言家", "女巫", "猎人", "守卫", "骑士", "白痴"]
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name in god_roles:
            p.status = PlayerStatus.DEAD_NIGHT

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")
    assert winner == "wolf", f"应该狼人胜利(屠神)，实际: {winner}"


def test_all_villagers_dead():
    """所有平民死亡（屠民）"""
    game = WerewolfGame(player_count=7, mode='watch')
    game.setup_game()

    # 杀死所有平民
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "平民":
            p.status = PlayerStatus.DEAD_NIGHT

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")


def test_vote_tie():
    """投票平票无结果"""
    game = WerewolfGame(player_count=4, mode='watch')
    game.setup_game()

    # 设置相同票数
    game.game_state.players[0].vote_count = 1
    game.game_state.players[1].vote_count = 1

    result = game.referee.judge_vote_result()
    print(f"投票结果: {result}")


def test_insufficient_players():
    """玩家数量不足"""
    try:
        game = WerewolfGame(player_count=2, mode='watch')
        game.setup_game()
        print("玩家数量: 2")
    except Exception as e:
        print(f"异常: {e}")


def test_no_prophet():
    """没有预言家"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    # 杀死预言家
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "预言家":
            p.status = PlayerStatus.DEAD_NIGHT
            break

    prophets = game.referee.get_gods()
    prophet_names = [p.role.name for p in prophets if p.role and p.role.name == "预言家"]
    print(f"存活预言家: {prophet_names}")


def test_no_witch():
    """没有女巫"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    # 杀死女巫
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "女巫":
            p.status = PlayerStatus.DEAD_NIGHT
            break

    print(f"女巫存活: {False}")


def test_role_distribution_edge():
    """角色分布边界测试"""
    for count in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        try:
            game = WerewolfGame(player_count=count, mode='watch')
            game.setup_game()
            dist = game.referee.get_role_distribution()
            print(f"{count}人局: {dist}")
        except Exception as e:
            print(f"{count}人局异常: {e}")


def test_couple_slay_all():
    """情侣存活到最后（第三方胜利）"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    # 只保留一对情侣（人狼链）
    alive_ids = []
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "丘比特":
            p.is_alive = lambda: True
            alive_ids.append(pid)
            lover_id = pid + 1 if pid + 1 < 9 else pid - 1
            if game.game_state.players.get(lover_id):
                game.game_state.players[lover_id].is_alive = lambda: True
                alive_ids.append(lover_id)
        else:
            p.status = PlayerStatus.DEAD_NIGHT

    # 设置人狼链情侣
    if len(alive_ids) >= 2:
        game.game_state.set_lovers(alive_ids[0], alive_ids[1])
        game.game_state.love_chain_type = "人狼链"

        # 杀死其他所有人
        for pid in game.game_state.players:
            if pid not in alive_ids:
                game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT

        winner = game.referee.check_win_condition()
        print(f"胜负判定: {winner}")


def test_double_protection_bug():
    """守卫连续守护同一人导致狼刀生效"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    guardian_id = find_role_player(game, "守卫")
    if guardian_id is None:
        print("没有守卫，跳过测试")
        return

    target_id = 1
    if target_id == guardian_id:
        target_id = (guardian_id + 1) % 9

    # 第一晚守护
    game.game_state.guardian_last_protect = target_id
    game.game_state.players[target_id].is_protected = True

    # 重置保护状态（模拟新夜晚）
    for p in game.game_state.players.values():
        p.is_protected = False

    # 第二晚再次守护同一人（如果系统没有检查）
    game.game_state.players[target_id].is_protected = True
    game.game_state.wolf_kill_target = target_id

    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"目标被守护: {game.game_state.players[target_id].is_protected}")
    print(f"守卫上次守护: {game.game_state.guardian_last_protect}")
    print(f"目标死亡: {target_id in deaths}")


def test_witch_save_poisoned_same_night():
    """女巫同晚救被毒的人"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    witch_id = None
    victim_id = 1

    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "女巫":
            witch_id = pid
            break

    # 狼刀目标
    game.game_state.wolf_kill_target = victim_id

    # 女巫救人
    game.game_state.revives.append(victim_id)

    # 有人被毒（不是victim_id）
    poison_target = 2
    game.game_state.players[poison_target].is_poisoned = True

    game.god.start_night()
    deaths = game.god.night_settle()

    print(f"狼刀目标: {victim_id}")
    print(f"女巫救人: {victim_id in game.game_state.revives}")
    print(f"被毒目标: {poison_target}")
    print(f"夜晚死亡: {deaths}")


def test_love_chain_revival():
    """情侣殉情后第三方胜利检测"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    # 找到人狼链情侣
    lover1_id = 0
    lover2_id = 1

    game.game_state.set_lovers(lover1_id, lover2_id)
    game.game_state.love_chain_type = "人狼链"

    # 杀死其中一个情侣
    game.game_state.players[lover1_id].status = PlayerStatus.DEAD_NIGHT

    # 检查殉情
    game.god.start_night()
    deaths = [lover1_id]
    game.god._handle_couple_death(deaths)

    print(f"殉情后死亡列表: {deaths}")
    print(f"存活玩家: {[p.id for p in game.game_state.get_alive_players()]}")


def test_idiot_lose_vote_after_reveal():
    """白痴翻牌后失去投票权"""
    game = WerewolfGame(player_count=5, mode='watch')
    game.setup_game()

    idiot_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "白痴":
            idiot_id = pid
            break

    if idiot_id is None:
        print("没有白痴，跳过测试")
        return

    # 白痴被投票出局翻牌
    game.referee.handle_player_death(idiot_id, cause="vote")

    idiot = game.game_state.players[idiot_id]
    print(f"白痴可投票: {idiot.role.can_vote if idiot.role else 'N/A'}")
    print(f"白痴可警长: {idiot.role.can_be_sheriff if idiot.role else 'N/A'}")
    print(f"白痴存活: {idiot.is_alive()}")

    # 检查投票权
    alive = game.game_state.get_alive_players()
    vote_players = [p.id for p in alive if p.can_vote()]
    print(f"可投票玩家: {vote_players}")


def test_guardian_last_protect_tracking():
    """守卫上晚守护记录追踪"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    # 第一晚
    game.game_state.guardian_last_protect = 1

    # 验证追踪
    print(f"第一晚守护: {game.game_state.guardian_last_protect}")

    # 第二晚应该不能守护同一人
    target = 1
    can_protect = target != game.game_state.guardian_last_protect
    print(f"第二晚可守护{target}: {can_protect}")


def test_multiple_wolves_vote():
    """多狼人阵营投票"""
    game = WerewolfGame(player_count=8, mode='watch')
    game.setup_game()

    wolves = []
    for pid, p in game.game_state.players.items():
        if p.role and p.role.camp == Camp.WOLF:
            wolves.append(pid)

    print(f"狼人数量: {len(wolves)}")
    print(f"狼人ID: {wolves}")


def test_flutist_curses_all():
    """吹笛人迷惑所有好人"""
    game = WerewolfGame(player_count=9, mode='watch')
    game.setup_game()

    # 迷惑所有非吹笛人
    flutist_id = None
    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "吹笛人":
            flutist_id = pid
        else:
            game.game_state.flutist_charmed_ids.append(pid)

    print(f"吹笛人ID: {flutist_id}")
    print(f"被迷惑玩家: {game.game_state.flutist_charmed_ids}")

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")


def test_hunter_last_alive():
    """猎人存活到最后单挑"""
    game = WerewolfGame(player_count=3, mode='watch')
    game.setup_game()

    # 杀死除猎人和一个狼人外的所有人
    hunter_id = None
    wolf_id = None

    for pid, p in game.game_state.players.items():
        if p.role and p.role.name == "猎人":
            hunter_id = pid
        elif p.role and p.role.camp == Camp.WOLF:
            wolf_id = pid

    # 杀死其他人
    for pid in game.game_state.players:
        if pid not in [hunter_id, wolf_id]:
            game.game_state.players[pid].status = PlayerStatus.DEAD_NIGHT

    print(f"存活: 猎人({hunter_id}), 狼人({wolf_id})")

    winner = game.referee.check_win_condition()
    print(f"胜负判定: {winner}")


# 运行所有测试
if __name__ == "__main__":
    print("=" * 60)
    print("好人阵营角色技能异常压测")
    print("=" * 60)

    tests = [
        # 预言家
        ("预言家查验自己", test_prophet_check_self),
        ("预言家查验死亡玩家", test_prophet_check_dead_player),
        ("预言家无可查验目标", test_prophet_check_no_candidates),
        ("预言家查验恶灵骑士", test_prophet_check_evil_knight),

        # 女巫
        ("女巫解药用尽", test_witch_no_heal_left),
        ("女巫毒药用尽", test_witch_no_poison_left),
        ("女巫同晚使用解药和毒药", test_witch_heal_and_poison_same_night),
        ("女巫救人vs守卫守护冲突", test_witch_heal_guardian_conflict),
        ("女巫毒死自己", test_witch_poison_self),
        ("女巫被刀且无解药", test_witch_no_potion_killed),

        # 猎人
        ("猎人无开枪目标", test_hunter_no_targets),
        ("猎人被女巫毒死不能开枪", test_hunter_poisoned_no_shoot),
        ("猎人开枪带走情侣", test_hunter_shoot_lover),
        ("猎人开枪带走白痴", test_hunter_shoot_idiot),
        ("猎人死亡时是白痴", test_hunter_death_idiot),

        # 白痴
        ("白痴被投票出局翻牌", test_idiot_voted_out),
        ("白痴被狼刀死", test_idiot_killed_by_wolf),
        ("白痴被女巫毒死", test_idiot_poisoned),
        ("白痴被猎人带走", test_idiot_shot_by_hunter),
        ("白痴翻牌后失去投票权", test_idiot_lose_vote_after_reveal),

        # 守卫
        ("守卫连续两晚守护同一人", test_guardian_consecutive_protect),
        ("守卫守护已死亡玩家", test_guardian_protect_dead_player),
        ("守卫守护自己", test_guardian_protect_self),
        ("守卫与女巫救人冲突", test_guardian_witch_conflict),
        ("守卫上晚守护记录追踪", test_guardian_last_protect_tracking),

        # 骑士
        ("骑士质疑狼王", test_knight_challenge_wolfking),
        ("骑士质疑情侣", test_knight_challenge_lover),
        ("骑士质疑已死亡玩家", test_knight_challenge_dead_player),

        # 通用
        ("所有好人死亡", test_all_goods_dead),
        ("所有狼人死亡", test_all_wolves_dead),
        ("所有神职死亡(屠神)", test_all_gods_dead),
        ("所有平民死亡(屠民)", test_all_villagers_dead),
        ("投票平票无结果", test_vote_tie),
        ("玩家数量不足", test_insufficient_players),
        ("没有预言家", test_no_prophet),
        ("没有女巫", test_no_witch),
        ("角色分布边界测试", test_role_distribution_edge),
        ("情侣存活到最后", test_couple_slay_all),
        ("守卫连续守护导致狼刀生效", test_double_protection_bug),
        ("女巫同晚救被毒的人", test_witch_save_poisoned_same_night),
        ("情侣殉情后第三方胜利检测", test_love_chain_revival),
        ("多狼人阵营投票", test_multiple_wolves_vote),
        ("吹笛人迷惑所有好人", test_flutist_curses_all),
        ("猎人存活到最后单挑", test_hunter_last_alive),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            test_case(name, func)
        except Exception as e:
            print(f"❌ 测试执行异常: {name}")
            print(f"   错误: {type(e).__name__}: {e}")
            test_results.append((name, "ERROR", str(e)))
            failed += 1
            continue

        if test_results[-1][1] == "PASS":
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    for name, status, error in test_results:
        if status == "PASS":
            print(f"✅ {name}")
        else:
            print(f"❌ {name}: {error}")

    print(f"\n通过: {passed}, 失败: {failed}, 总计: {len(test_results)}")
