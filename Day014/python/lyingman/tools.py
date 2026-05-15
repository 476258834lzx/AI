"""狼人杀技能tools - 所有角色技能"""
from typing import Optional, List
from langchain_core.tools import tool
from .game_state import GameState, NightPhase, PlayerStatus
from .roles import RoleType


# ===== 狼人阵营技能 =====

@tool
def wolf_discuss_propose(game_state: dict, player_id: int, target_id: int) -> str:
    """
    狼人商讨-提议刀人目标。

    狼人在夜间可以互相确认身份并商讨战术。每个狼人提出自己想要刀的目标，
    只有所有狼人都认可同一个目标时，才能成功刀人。

    Args:
        game_state: 游戏状态
        player_id: 狼人ID
        target_id: 提议刀的目标玩家ID

    Returns:
        提议结果
    """
    state = GameState(**game_state)
    player = state.players.get(player_id)
    target = state.players.get(target_id)

    if not player or not target:
        return "玩家不存在"

    if not player.is_alive():
        return "你已死亡"

    wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
    if not player.role or player.role.name not in wolf_roles:
        return "你不是狼人，无法参与商讨"

    if not target.is_alive():
        return "目标已死亡"

    if player_id == target_id:
        return "不能提议刀自己"

    state.wolf_discuss_proposals[player_id] = target_id
    return f"你提议刀杀玩家 {target.name}（ID: {target_id}）"


@tool
def wolf_discuss_agree(game_state: dict, player_id: int, target_id: int) -> str:
    """
    狼人商讨-同意某提议。

    狼人同意某个提议的目标。所有人同意后刀人目标确定。

    Args:
        game_state: 游戏状态
        player_id: 狼人ID
        target_id: 同意的目标玩家ID

    Returns:
        同意结果
    """
    state = GameState(**game_state)
    player = state.players.get(player_id)

    if not player:
        return "玩家不存在"

    if not player.is_alive():
        return "你已死亡"

    wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
    if not player.role or player.role.name not in wolf_roles:
        return "你不是狼人"

    # 检查目标是否存在
    if target_id not in state.wolf_discuss_proposals.values():
        return "该目标没有被提议，请先提议"

    # 检查是否所有人都同意了
    wolves = [pid for pid, p in state.players.items()
              if p.is_alive() and p.role and p.role.name in wolf_roles]

    # 计算已同意该目标的人数
    agree_count = sum(1 for pid, t in state.wolf_discuss_proposals.items()
                      if pid in wolves and t == target_id)

    # 检查当前提议
    current_proposal = state.wolf_discuss_proposals.get(player_id)
    if current_proposal == target_id:
        return f"你已经提议并同意刀杀该目标了"

    # 更新为同意该目标
    state.wolf_discuss_proposals[player_id] = target_id

    # 重新计算同意人数
    agree_count = sum(1 for pid, t in state.wolf_discuss_proposals.items()
                      if pid in wolves and t == target_id)

    if agree_count == len(wolves):
        state.wolf_consensus_target = target_id
        target = state.players.get(target_id)
        return f"所有狼人达成共识！今晚刀杀 {target.name}（ID: {target_id}）"
    else:
        remaining = len(wolves) - agree_count
        target = state.players.get(target_id)
        return f"你同意刀杀 {target.name}，还差{remaining}人同意"


@tool
def wolf_confirm_kill(game_state: dict, player_id: int) -> str:
    """
    狼人确认刀人（条件边触发）。

    当所有狼人都认可同一个目标后，每个狼人需要调用此函数确认。
    只有所有狼人都确认后，刀人才会执行。

    Args:
        game_state: 游戏状态
        player_id: 狼人ID

    Returns:
        确认结果
    """
    state = GameState(**game_state)
    player = state.players.get(player_id)

    if not player:
        return "玩家不存在"

    if not player.is_alive():
        return "你已死亡"

    wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
    if not player.role or player.role.name not in wolf_roles:
        return "你不是狼人"

    # 检查是否有共识目标
    if state.wolf_consensus_target is None:
        # 检查是否所有人都提议了同一目标
        wolves = [pid for pid, p in state.players.items()
                  if p.is_alive() and p.role and p.role.name in wolf_roles]

        if len(state.wolf_discuss_proposals) < len(wolves):
            return "还有狼人没有提议或同意，等待所有人参与商讨"

        proposals = list(state.wolf_discuss_proposals.values())
        if len(set(proposals)) == 1:
            state.wolf_consensus_target = proposals[0]
        else:
            return "狼人们尚未达成共识，请继续商讨"

    # 添加到确认列表
    if player_id not in state.wolf_awaiting_confirm:
        state.wolf_awaiting_confirm.append(player_id)

    wolves = [pid for pid, p in state.players.items()
              if p.is_alive() and p.role and p.role.name in wolf_roles]

    if len(state.wolf_awaiting_confirm) == len(wolves):
        # 所有狼人确认，执行刀人
        state.wolf_kill_target = state.wolf_consensus_target
        state.wolf_discuss_proposals.clear()
        state.wolf_awaiting_confirm.clear()
        target = state.players.get(state.wolf_consensus_target)
        return f"所有狼人确认完毕！今晚刀杀 {target.name}（ID: {state.wolf_consensus_target}）"
    else:
        remaining = len(wolves) - len(state.wolf_awaiting_confirm)
        return f"你已确认，还差{remaining}个狼人确认"


@tool
def wolf_kill(game_state: dict, killer_id: int, target_id: int) -> str:
    """
    狼人刀人技能（直接刀人，无需商讨）。

    Args:
        game_state: 游戏状态
        killer_id: 杀手ID
        target_id: 目标玩家ID

    Returns:
        执行结果
    """
    state = GameState(**game_state)
    killer = state.players.get(killer_id)
    target = state.players.get(target_id)

    if not killer or not target:
        return "玩家不存在"

    if not killer.is_alive():
        return "杀手已死亡，无法行动"

    wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
    if not killer.role or killer.role.name not in wolf_roles:
        return "你不能使用刀人技能"

    if not target.is_alive():
        return "目标已死亡"

    state.wolf_kill_target = target_id
    return f"你选择刀杀玩家 {target.name}（ID: {target_id}）"


@tool
def wolf_self_kill(game_state: dict, killer_id: int) -> str:
    """狼人自刀（用于自刀战术）"""
    state = GameState(**game_state)
    killer = state.players.get(killer_id)

    if not killer:
        return "玩家不存在"

    if not killer.is_alive():
        return "你已死亡"

    if not killer.role or not killer.role.name.endswith("狼"):
        return "只有狼人可以自刀"

    state.wolf_kill_target = killer_id
    return "你选择自刀"


# ===== 预言家技能 =====

@tool
def seer_check(game_state: dict, seer_id: int, target_id: int) -> str:
    """
    预言家查验技能。

    Args:
        game_state: 游戏状态
        seer_id: 预言家ID
        target_id: 查验目标ID

    Returns:
        查验结果（好人/狼人）
    """
    state = GameState(**game_state)
    seer = state.players.get(seer_id)
    target = state.players.get(target_id)

    if not seer or not target:
        return "玩家不存在"

    if not seer.is_alive():
        return "预言家已死亡"

    if seer.role and seer.role.name != "预言家":
        return "你不是预言家"

    if not target.is_alive():
        return "不能查验已死亡玩家"

    if seer_id == target_id:
        return "不能查验自己"

    # 恶灵骑士被查验显示为好人
    if target.role and target.role.name == "恶灵骑士":
        result = "好人"
    elif target.role and target.role.name.endswith("狼"):
        result = "狼人"
    else:
        result = "好人"

    state.seer_check_result = {
        "target_id": target_id,
        "target_name": target.name,
        "result": result,
        "seer_id": seer_id,
    }

    return f"查验结果：玩家 {target.name}（ID: {target_id}）是【{result}】"


# ===== 女巫技能 =====

@tool
def witch_heal(game_state: dict, witch_id: int, target_id: int) -> str:
    """女巫救人技能"""
    state = GameState(**game_state)
    witch = state.players.get(witch_id)
    target = state.players.get(target_id)

    if not witch or not target:
        return "玩家不存在"

    if not witch.is_alive():
        return "女巫已死亡"

    if witch.role and witch.role.name != "女巫":
        return "你不是女巫"

    if state.witch_heal_used:
        return "解药已使用"

    state.witch_heal_used = True
    state.revives.append(target_id)

    return f"你使用解药救活了玩家 {target.name}（ID: {target_id}）"


@tool
def witch_poison(game_state: dict, witch_id: int, target_id: int) -> str:
    """女巫毒人技能"""
    state = GameState(**game_state)
    witch = state.players.get(witch_id)
    target = state.players.get(target_id)

    if not witch or not target:
        return "玩家不存在"

    if not witch.is_alive():
        return "女巫已死亡"

    if witch.role and witch.role.name != "女巫":
        return "你不是女巫"

    if not target.is_alive():
        return "不能毒杀已死亡玩家"

    if state.witch_poison_used:
        return "毒药已使用"

    if witch_id == target_id:
        return "不能毒杀自己"

    state.witch_poison_used = True
    target.is_poisoned = True
    state.deaths.append(target_id)

    return f"你使用毒药毒杀了玩家 {target.name}（ID: {target_id}）"


@tool
def witch_skip_heal(game_state: dict, witch_id: int) -> str:
    """女巫不使用解药"""
    state = GameState(**game_state)
    witch = state.players.get(witch_id)

    if not witch:
        return "玩家不存在"

    if not witch.is_alive():
        return "女巫已死亡"

    if witch.role and witch.role.name != "女巫":
        return "你不是女巫"

    if state.witch_heal_used:
        return "解药已使用"

    return "你选择不使用解药"


@tool
def witch_skip_poison(game_state: dict, witch_id: int) -> str:
    """女巫不使用毒药"""
    state = GameState(**game_state)
    witch = state.players.get(witch_id)

    if not witch:
        return "玩家不存在"

    if not witch.is_alive():
        return "女巫已死亡"

    if witch.role and witch.role.name != "女巫":
        return "你不是女巫"

    if state.witch_poison_used:
        return "毒药已使用"

    return "你选择不使用毒药"


# ===== 守卫技能 =====

@tool
def guardian_protect(game_state: dict, guardian_id: int, target_id: int) -> str:
    """守卫守护技能"""
    state = GameState(**game_state)
    guardian = state.players.get(guardian_id)
    target = state.players.get(target_id)

    if not guardian or not target:
        return "玩家不存在"

    if not guardian.is_alive():
        return "守卫已死亡"

    if guardian.role and guardian.role.name != "守卫":
        return "你不是守卫"

    if not target.is_alive():
        return "不能守护已死亡玩家"

    if state.guardian_last_protect == target_id:
        return f"不能连续两晚守护同一人，上晚守护的是玩家{state.guardian_last_protect}"

    target.is_protected = True
    state.guardian_last_protect = target_id

    return f"你守护了玩家 {target.name}（ID: {target_id}）"


@tool
def guardian_skip_protect(game_state: dict, guardian_id: int) -> str:
    """守卫不守护任何人"""
    state = GameState(**game_state)
    guardian = state.players.get(guardian_id)

    if not guardian:
        return "玩家不存在"

    if not guardian.is_alive():
        return "守卫已死亡"

    if guardian.role and guardian.role.name != "守卫":
        return "你不是守卫"

    return "你选择不守护任何人"


# ===== 猎人技能 =====

@tool
def hunter_shoot(game_state: dict, hunter_id: int, target_id: int) -> str:
    """猎人开枪带走技能"""
    state = GameState(**game_state)
    hunter = state.players.get(hunter_id)
    target = state.players.get(target_id)

    if not hunter or not target:
        return "玩家不存在"

    if not hunter.is_alive():
        return "猎人已死亡"

    if hunter.role and hunter.role.name != "猎人":
        return "你不是猎人"

    if not target.is_alive():
        return "不能带走已死亡玩家"

    if hunter.is_poisoned:
        return "你被毒死，无法开枪"

    state.deaths.append(target_id)
    return f"你开枪带走了玩家 {target.name}（ID: {target_id}）"


@tool
def hunter_skip_shoot(game_state: dict, hunter_id: int) -> str:
    """猎人不开枪"""
    state = GameState(**game_state)
    hunter = state.players.get(hunter_id)

    if not hunter:
        return "玩家不存在"

    if not hunter.is_alive():
        return "猎人已死亡"

    if hunter.role and hunter.role.name != "猎人":
        return "你不是猎人"

    return "你选择不开枪"


# ===== 白痴技能 =====

@tool
def idiot_reveal(game_state: dict, idiot_id: int) -> str:
    """白痴翻牌技能"""
    state = GameState(**game_state)
    idiot = state.players.get(idiot_id)

    if not idiot:
        return "玩家不存在"

    if not idiot.is_alive():
        return "白痴已死亡"

    if idiot.role and idiot.role.name != "白痴":
        return "你不是白痴"

    idiot.role.can_vote = False
    return "你翻牌展示身份，存活但失去投票权"


# ===== 骑士技能 =====

@tool
def knight_challenge(game_state: dict, knight_id: int, target_id: int) -> str:
    """骑士质疑技能"""
    state = GameState(**game_state)
    knight = state.players.get(knight_id)
    target = state.players.get(target_id)

    if not knight or not target:
        return "玩家不存在"

    if not knight.is_alive():
        return "骑士已死亡"

    if knight.role and knight.role.name != "骑士":
        return "你不是骑士"

    if not target.is_alive():
        return "不能质疑已死亡玩家"

    if target.role and target.role.name.endswith("狼"):
        state.deaths.append(target_id)
        return f"质疑成功！玩家 {target.name}（ID: {target_id}）是狼人，已被处决"
    else:
        state.deaths.append(knight_id)
        return f"质疑失败！玩家 {target.name}（ID: {target_id}）不是狼人，骑士死亡"


@tool
def knight_skip_challenge(game_state: dict, knight_id: int) -> str:
    """骑士不质疑"""
    state = GameState(**game_state)
    knight = state.players.get(knight_id)

    if not knight:
        return "玩家不存在"

    if not knight.is_alive():
        return "骑士已死亡"

    if knight.role and knight.role.name != "骑士":
        return "你不是骑士"

    return "你选择不质疑任何人"


# ===== 投票技能 =====

@tool
def vote_player(game_state: dict, voter_id: int, target_id: int) -> str:
    """投票技能（所有存活玩家可用）"""
    state = GameState(**game_state)
    voter = state.players.get(voter_id)
    target = state.players.get(target_id)

    if not voter or not target:
        return "玩家不存在"

    if not voter.can_vote():
        return "你不能投票（已死亡或被迷惑）"

    if not target.is_alive():
        return "不能投票给已死亡玩家"

    vote_count = 1.5 if voter.is_sheriff else 1
    voter.vote_count += vote_count
    target.vote_count += vote_count

    return f"你投票给了玩家 {target.name}（ID: {target_id}）"


@tool
def vote_skip(game_state: dict, voter_id: int) -> str:
    """跳过投票（投自己）"""
    state = GameState(**game_state)
    voter = state.players.get(voter_id)

    if not voter:
        return "玩家不存在"

    if not voter.can_vote():
        return "你不能投票（已死亡或被迷惑）"

    voter.vote_count += 0
    return "你选择弃票"


# ===== 警长竞选技能 =====

@tool
def run_for_sheriff(game_state: dict, player_id: int) -> str:
    """参加警长竞选"""
    state = GameState(**game_state)
    player = state.players.get(player_id)

    if not player:
        return "玩家不存在"

    if not player.is_alive():
        return "已死亡玩家不能竞选警长"

    if player.role and not player.role.can_be_sheriff:
        return f"{player.role.name}不能竞选警长"

    if player_id in state.sheriff_candidate_ids:
        return "你已经在竞选中了"

    state.sheriff_candidate_ids.append(player_id)
    return "你参加警长竞选"


@tool
def withdraw_from_sheriff(game_state: dict, player_id: int) -> str:
    """退出警长竞选"""
    state = GameState(**game_state)

    if player_id in state.sheriff_candidate_ids:
        state.sheriff_candidate_ids.remove(player_id)
        return "你退出警长竞选"
    else:
        return "你不在竞选中"


# ===== 白狼王自爆 =====

@tool
def white_wolf_self_destruct(game_state: dict, white_wolf_id: int, target_id: Optional[int] = None) -> str:
    """白狼王自爆带走技能"""
    state = GameState(**game_state)
    white_wolf = state.players.get(white_wolf_id)

    if not white_wolf:
        return "玩家不存在"

    if not white_wolf.is_alive():
        return "白狼王已死亡"

    if white_wolf.role and white_wolf.role.name != "白狼王":
        return "你不是白狼王"

    state.deaths.append(white_wolf_id)

    result = "白狼王自爆！"
    if target_id:
        target = state.players.get(target_id)
        if target and target.is_alive():
            state.deaths.append(target_id)
            result += f"同时带走了玩家 {target.name}（ID: {target_id}）"

    return result


@tool
def wolf_self_destruct(game_state: dict, wolf_id: int) -> str:
    """普通狼人/狼王/狼美人自爆（不能带人，直接进入黑夜）"""
    state = GameState(**game_state)
    wolf = state.players.get(wolf_id)

    if not wolf:
        return "玩家不存在"

    if not wolf.is_alive():
        return "狼人已死亡"

    wolf_role_names = ["狼人", "狼王", "狼美人"]
    if not wolf.role or wolf.role.name not in wolf_role_names:
        return "你不是狼人阵营"

    wolf.status = PlayerStatus.DEAD_DAY
    return f"【{wolf.name}】自爆出局！游戏直接进入黑夜。"


# ===== 中立角色技能 =====

@tool
def cupid_link(game_state: dict, cupid_id: int, lover1_id: int, lover2_id: int) -> str:
    """丘比特指定情侣"""
    state = GameState(**game_state)
    cupid = state.players.get(cupid_id)
    lover1 = state.players.get(lover1_id)
    lover2 = state.players.get(lover2_id)

    if not all([cupid, lover1, lover2]):
        return "玩家不存在"

    if not cupid.is_alive():
        return "丘比特已死亡"

    if cupid.role and cupid.role.name != "丘比特":
        return "你不是丘比特"

    if lover1_id == lover2_id:
        return "情侣不能是同一人"

    state.day_actions.append({
        "phase": "cupid_link",
        "cupid_id": cupid_id,
        "lovers": [lover1_id, lover2_id],
    })

    return f"你指定 {lover1.name} 和 {lover2.name} 为情侣"


@tool
def wild_child_choose_idol(game_state: dict, wild_child_id: int, idol_id: int) -> str:
    """野孩子选择偶像"""
    state = GameState(**game_state)
    wild_child = state.players.get(wild_child_id)
    idol = state.players.get(idol_id)

    if not wild_child or not idol:
        return "玩家不存在"

    if not wild_child.is_alive():
        return "野孩子已死亡"

    if wild_child.role and wild_child.role.name != "野孩子":
        return "你不是野孩子"

    if idol_id == wild_child_id:
        return "不能选择自己为偶像"

    state.day_actions.append({
        "phase": "wild_child_choose_idol",
        "wild_child_id": wild_child_id,
        "idol_id": idol_id,
    })

    return f"你选择 {idol.name} 为你的偶像"


@tool
def flutist_charm(game_state: dict, flutist_id: int, target1_id: int, target2_id: int) -> str:
    """吹笛人迷惑技能"""
    state = GameState(**game_state)
    flutist = state.players.get(flutist_id)
    target1 = state.players.get(target1_id)
    target2 = state.players.get(target2_id)

    if not all([flutist, target1, target2]):
        return "玩家不存在"

    if not flutist.is_alive():
        return "吹笛人已死亡"

    if flutist.role and flutist.role.name != "吹笛人":
        return "你不是吹笛人"

    target1.is_cursed = True
    target2.is_cursed = True

    return f"你迷惑了 {target1.name} 和 {target2.name}，他们失去投票权"


@tool
def fox_check(game_state: dict, fox_id: int, target1_id: int, target2_id: int, target3_id: int) -> str:
    """狐狸查验技能（查验3个相邻位置）"""
    state = GameState(**game_state)
    fox = state.players.get(fox_id)
    targets = [
        state.players.get(target1_id),
        state.players.get(target2_id),
        state.players.get(target3_id),
    ]

    if not fox:
        return "玩家不存在"

    if not fox.is_alive():
        return "狐狸已死亡"

    if fox.role and fox.role.name != "狐狸":
        return "你不是狐狸"

    # 检查是否有狼人
    has_wolf = False
    for target in targets:
        if target and target.role and target.role.name.endswith("狼"):
            has_wolf = True
            break

    if has_wolf:
        # 有狼人，狐狸失去技能
        return "查验范围内有狼人，你失去技能"
    else:
        # 无狼人，狐狸免疫狼刀
        return "查验范围内无狼人，你免疫狼刀"


@tool
def bear_check(game_state: dict, bear_id: int) -> dict:
    """熊咆哮检测（由系统调用）"""
    state = GameState(**game_state)
    bear = state.players.get(bear_id)

    if not bear or not bear.is_alive():
        return {"has_bear": False, "roar": False}

    if bear.role and bear.role.name != "熊":
        return {"has_bear": False, "roar": False}

    # 检查相邻位置是否有狼人
    bear_pos = bear.id
    positions = [(bear_pos - 1) % len(state.players), (bear_pos + 1) % len(state.players)]

    has_wolf = False
    for pos in positions:
        player = state.players.get(pos)
        if player and player.is_alive() and player.role and player.role.name.endswith("狼"):
            has_wolf = True
            break

    return {"has_bear": True, "roar": has_wolf}


# ===== 角色技能映射 =====

def get_available_tools(role_name: str) -> list:
    """获取角色可用的技能"""
    wolf_night_tools = [wolf_discuss_propose, wolf_discuss_agree, wolf_confirm_kill, wolf_kill, wolf_self_kill]
    tools_map = {
        "狼人": wolf_night_tools + [wolf_self_destruct, vote_player],
        "狼王": wolf_night_tools + [wolf_self_destruct, hunter_shoot, vote_player],
        "白狼王": wolf_night_tools + [white_wolf_self_destruct, vote_player],
        "狼美人": wolf_night_tools + [wolf_self_destruct, vote_player],
        "恶灵骑士": wolf_night_tools + [hunter_shoot, vote_player],
        "预言家": [seer_check, vote_player],
        "女巫": [witch_heal, witch_poison, witch_skip_heal, witch_skip_poison, vote_player],
        "猎人": [hunter_shoot, hunter_skip_shoot, vote_player],
        "白痴": [idiot_reveal, vote_player],
        "守卫": [guardian_protect, guardian_skip_protect, vote_player],
        "骑士": [knight_challenge, knight_skip_challenge, vote_player],
        "平民": [vote_player, vote_skip],
        "丘比特": [cupid_link, vote_player],
        "野孩子": [wild_child_choose_idol, vote_player],
        "吹笛人": [flutist_charm, vote_player],
        "狐狸": [fox_check, vote_player],
        "熊": [vote_player],
    }
    return tools_map.get(role_name, [vote_player])


def get_night_action_tools(role_name: str) -> dict:
    """获取夜间动作技能"""
    night_tools = {
        "狼人": {"action": wolf_kill, "skip": None, "target_count": 1},
        "预言家": {"action": seer_check, "skip": None, "target_count": 1},
        "女巫": {
            "heal": witch_heal,
            "poison": witch_poison,
            "skip_heal": witch_skip_heal,
            "skip_poison": witch_skip_poison,
        },
        "守卫": {"action": guardian_protect, "skip": guardian_skip_protect, "target_count": 1},
        "猎人": {"action": hunter_shoot, "skip": hunter_skip_shoot, "target_count": 1},
        "骑士": {"action": knight_challenge, "skip": knight_skip_challenge, "target_count": 1},
        "丘比特": {"action": cupid_link, "skip": None, "target_count": 2},
        "野孩子": {"action": wild_child_choose_idol, "skip": None, "target_count": 1},
        "吹笛人": {"action": flutist_charm, "skip": None, "target_count": 2},
        "狐狸": {"action": fox_check, "skip": None, "target_count": 3},
    }
    return night_tools.get(role_name, {})