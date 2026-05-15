"""游戏状态定义"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from .roles import Camp


class PlayerStatus(Enum):
    """玩家状态"""
    ALIVE = "alive"           # 存活
    DEAD_NIGHT = "dead_night" # 夜间死亡
    DEAD_DAY = "dead_day"     # 白天死亡
    DEAD_REVIVE = "dead_revote" # 被女巫救活


class GamePhase(Enum):
    """游戏阶段"""
    NOT_START = "not_start"
    DAY = "day"
    NIGHT = "night"
    END = "end"


class RoundPhase(Enum):
    """回合阶段"""
    # 白天阶段
    DAY_START = "day_start"
    SHERIFF_ELECTION = "sheriff_election"  # 警长竞选（仅第一天）
    DAY_SPEECH = "day_speech"              # 日常发言
    DAY_VOTE = "day_vote"                   # 放逐投票
    LAST_WORDS = "last_words"               # 遗言

    # 夜晚阶段
    NIGHT_START = "night_start"
    WOLF_KILL = "wolf_kill"                # 狼人刀人
    SEER_CHECK = "seer_check"              # 预言家查验
    WITCH_HEAL = "witch_heal"              # 女巫救人
    WITCH_POISON = "witch_poison"           # 女巫毒人
    GUARDIAN_PROTECT = "guardian_protect"  # 守卫守护
    HUNTER_CHOICE = "hunter_choice"        # 猎人选择
    NIGHT_END = "night_end"                # 夜晚结算


class NightPhase(Enum):
    """夜间动作阶段（固定顺序）"""
    WOLF_DISCUSS = "wolf_discuss"      # 狼人商讨（先讨论再刀人）
    WOLF_KILL = "wolf_kill"           # 狼人刀人
    SEER_CHECK = "seer_check"
    WITCH_HEAL = "witch_heal"
    WITCH_POISON = "witch_poison"
    GUARDIAN_PROTECT = "guardian_protect"
    HUNTER_CHOICE = "hunter_choice"


@dataclass
class Role:
    """角色信息"""
    name: str
    description: str
    camp: Camp = Camp.GOOD  # 默认好人阵营
    can_vote: bool = True
    can_be_sheriff: bool = True
    can_shoot_on_death: bool = False  # 死亡时能否开枪（猎人）
    shoot_target_required: bool = False  # 是否需要选择开枪目标

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "can_vote": self.can_vote,
            "can_be_sheriff": self.can_be_sheriff,
        }


@dataclass
class Player:
    """玩家信息"""
    id: int
    name: str
    role: Optional[Role] = None
    status: PlayerStatus = PlayerStatus.ALIVE
    is_sheriff: bool = False
    is_cursed: bool = False        # 被吹笛人迷惑
    is_protected: bool = False    # 被守卫守护
    is_poisoned: bool = False     # 被女巫毒
    vote_count: int = 0           # 得票数
    last_words: str = ""           # 遗言

    def is_alive(self) -> bool:
        return self.status == PlayerStatus.ALIVE

    def can_vote(self) -> bool:
        return self.is_alive() and self.role and self.role.can_vote and not self.is_cursed

    def to_dict(self, hide_role: bool = False) -> dict:
        """转换为字典，hide_role=True时隐藏角色信息"""
        data = {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "is_sheriff": self.is_sheriff,
            "is_cursed": self.is_cursed,
            "vote_count": self.vote_count,
        }
        if not hide_role and self.role:
            data["role"] = self.role.name
        return data


@dataclass
class NightAction:
    """夜间动作记录"""
    phase: NightPhase
    actor_id: int  # 动作执行者ID
    target_id: Optional[int] = None  # 动作目标ID
    result: Optional[str] = None  # 动作结果
    success: bool = False


@dataclass
class DayAction:
    """白天动作记录"""
    phase: RoundPhase
    speaker_id: int
    content: str
    votes: dict = field(default_factory=dict)  # {player_id: target_id}


@dataclass
class Round:
    """回合信息"""
    day: int
    phase: RoundPhase
    actions: list = field(default_factory=list)  # NightAction or DayAction
    deaths: list = field(default_factory=list)  # [player_id, ...]
    revives: list = field(default_factory=list)  # [player_id, ...]


@dataclass
class GameState:
    """游戏状态"""
    players: dict[int, Player] = field(default_factory=dict)
    current_day: int = 1
    current_phase: RoundPhase = RoundPhase.DAY_START
    sheriff_id: Optional[int] = None
    sheriff_candidate_ids: list[int] = field(default_factory=list)
    winner: Optional[str] = None
    round_history: list[Round] = field(default_factory=list)
    night_actions: list[NightAction] = field(default_factory=list)
    day_actions: list[DayAction] = field(default_factory=list)

    # 夜间信息（临时）
    wolf_kill_target: Optional[int] = None
    seer_check_result: Optional[dict] = None
    witch_heal_used: bool = False
    witch_poison_used: bool = False
    guardian_last_protect: Optional[int] = None

    # 中立角色信息
    lovers: dict[int, int] = field(default_factory=dict)  # player_id -> lover_id
    cupid_id: Optional[int] = None
    flutist_id: Optional[int] = None
    blood_moon_id: Optional[int] = None
    wild_child_id: Optional[int] = None
    wild_child_idol: Optional[int] = None
    love_chain_type: Optional[str] = None  # 人狼链/人人链/狼狼链

    # 狼美人魅惑
    wolf_beauty_charm_target: Optional[int] = None

    # 女巫救人
    revives: list[int] = field(default_factory=list)  # 被女巫救活的玩家ID

    # 狼人夜间商讨（共识机制）
    wolf_discuss_proposals: dict[int, int] = field(default_factory=dict)  # player_id -> target_id
    wolf_consensus_target: Optional[int] = None  # 共识目标
    wolf_awaiting_confirm: list[int] = field(default_factory=list)  # 待确认的狼人ID

    hunter_shoot_target: Optional[int] = None  # 猎人选择开枪带走的目标

    # 恶灵骑士反伤记录
    evil_knight_checked_by: Optional[int] = None  # 被哪个预言家查验过
    evil_knight_poisoned_by: Optional[int] = None  # 被哪个女巫毒过

    # 中立角色技能状态
    fox_lost_skill: bool = False  # 狐狸是否失去技能
    fox_last_check_has_wolf: bool = False  # 狐狸上次查验是否有狼人
    bear_roared_today: bool = False  # 今天熊是否咆哮
    flutist_charmed_ids: list[int] = field(default_factory=list)  # 被吹笛人迷惑的玩家ID

    def get_alive_players(self) -> list[Player]:
        return [p for p in self.players.values() if p.is_alive()]

    def get_players_by_camp(self, camp: str) -> list[Player]:
        """获取阵营玩家"""
        return [p for p in self.get_alive_players() if p.role and p.role.camp.value == camp]

    # ===== 情侣相关方法 =====

    def set_lovers(self, id1: int, id2: int):
        """设置情侣关系"""
        self.lovers[id1] = id2
        self.lovers[id2] = id1

    def get_lover(self, player_id: int) -> Optional[int]:
        """获取情侣ID"""
        return self.lovers.get(player_id)

    def is_lover(self, id1: int, id2: int) -> bool:
        """判断两人是否是情侣"""
        return self.lovers.get(id1) == id2

    def get_love_type(self, player_id: int) -> Optional[str]:
        """获取情侣类型（人人链/狼狼链/人狼链）"""
        if player_id not in self.lovers:
            return None
        lover_id = self.lovers[player_id]
        p1 = self.players.get(player_id)
        p2 = self.players.get(lover_id)
        if not p1 or not p2 or not p1.role or not p2.role:
            return None
        p1_is_wolf = self._is_wolf_role(p1.role.name)
        p2_is_wolf = self._is_wolf_role(p2.role.name)
        if p1_is_wolf and p2_is_wolf:
            return "狼狼链"
        elif not p1_is_wolf and not p2_is_wolf:
            return "人人链"
        else:
            return "人狼链"

    # ===== 中立角色方法 =====

    def set_cupid_id(self, cupid_id: int):
        self.cupid_id = cupid_id

    def set_flutist_id(self, flutist_id: int):
        self.flutist_id = flutist_id

    def set_blood_moon_id(self, blood_moon_id: int):
        self.blood_moon_id = blood_moon_id

    def set_wild_child_id(self, wild_child_id: int):
        self.wild_child_id = wild_child_id

    def set_wild_child_idol(self, idol_id: int):
        self.wild_child_idol = idol_id

    def is_wild_child_converted(self) -> bool:
        """野孩子是否已转换阵营"""
        if self.wild_child_idol is None or self.wild_child_id is None:
            return False
        idol = self.players.get(self.wild_child_idol)
        return idol is not None and not idol.is_alive()

    def is_cursed(self, player_id: int) -> bool:
        """检查玩家是否被吹笛人迷惑"""
        return player_id in self.flutist_charmed_ids

    def get_wolves(self) -> list[Player]:
        return self.get_players_by_camp("wolf")

    def get_goods(self) -> list[Player]:
        return self.get_players_by_camp("good")

    def is_couple_wolf_human(self) -> bool:
        """检查是否有人狼链情侣"""
        return self.love_chain_type == "人狼链"

    def _is_wolf_role(self, role_name: str) -> bool:
        """检查是否为狼人角色"""
        if not role_name:
            return False
        wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
        return role_name in wolf_roles

    def is_love_chain_alone(self) -> bool:
        """检查情侣（人狼链）是否存活到最后"""
        if not self.is_couple_wolf_human():
            return False

        # 人狼链情侣存活状态检查（不要求丘比特存活）
        if not self.lovers:
            return False

        # 获取人狼链的情侣（丘比特和其伴侣）
        cupid_or_lover_ids = []
        for pid in self.lovers:
            p = self.players.get(pid)
            if p and p.role and p.role.name == "丘比特" and p.is_alive():
                cupid_or_lover_ids.append(pid)
                lover_id = self.get_lover(pid)
                if lover_id is not None:
                    cupid_or_lover_ids.append(lover_id)
                break

        # 如果没有存活的丘比特，尝试从情侣中找人狼链
        if not cupid_or_lover_ids:
            for pid in self.lovers:
                p = self.players.get(pid)
                lover_id = self.get_lover(pid)
                if p and lover_id is not None:
                    lover = self.players.get(lover_id)
                    if p.is_alive() and lover and lover.is_alive():
                        # 检查是否是人狼链（一人是狼，一人不是狼）
                        p_is_wolf = self._is_wolf_role(p.role.name) if p.role else False
                        l_is_wolf = self._is_wolf_role(lover.role.name) if lover.role else False
                        if p_is_wolf != l_is_wolf:
                            cupid_or_lover_ids = [pid, lover_id]
                            break

        if len(cupid_or_lover_ids) < 2:
            return False

        # 检查除了这对情侣外是否还有其他存活玩家
        alive = self.get_alive_players()
        other_alive = [p for p in alive if p.id not in cupid_or_lover_ids]
        return len(other_alive) == 0

    def to_god_view(self) -> dict:
        """上帝视角的完整信息"""
        return {
            "current_day": self.current_day,
            "current_phase": self.current_phase.value,
            "players": {pid: p.to_dict(hide_role=False) for pid, p in self.players.items()},
            "sheriff_id": self.sheriff_id,
            "round_history": [
                {
                    "day": r.day,
                    "phase": r.phase.value,
                    "actions": [
                        {"actor": a.actor_id, "target": a.target_id, "result": a.result, "success": a.success}
                        for a in r.actions
                    ],
                    "deaths": r.deaths,
                }
                for r in self.round_history
            ],
        }

    def to_player_view(self, player_id: int) -> dict:
        """玩家视角的信息（隐藏他人角色）"""
        players_info = {}
        for pid, p in self.players.items():
            if pid == player_id:
                players_info[pid] = p.to_dict(hide_role=False)  # 看到自己的角色
            else:
                players_info[pid] = p.to_dict(hide_role=True)  # 隐藏其他角色

        return {
            "current_day": self.current_day,
            "current_phase": self.current_phase.value,
            "players": players_info,
            "sheriff_id": self.sheriff_id,
            "my_id": player_id,
            "round_history": [
                {
                    "day": r.day,
                    "phase": r.phase.value,
                    "actions": r.actions,  # 根据权限过滤
                    "deaths": r.deaths,
                }
                for r in self.round_history[-3:]  # 只显示最近3轮
            ],
        }