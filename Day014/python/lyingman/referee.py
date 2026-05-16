"""裁判 - 负责角色分发和胜负判定"""
import random
from typing import Optional
from .game_state import GameState, Player, RoundPhase, PlayerStatus
from .roles import RoleType, Camp
from .good_roles import create_good_role
from .wolf_roles import create_wolf_role
from .neutral_roles import create_neutral_role


# 屠城局配置（5-8人）：狼人:好人 = 2:3
DEFAULT_5P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.VILLAGER],
    "neutral": [],
}

DEFAULT_6P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [],
}

DEFAULT_7P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.VILLAGER],
    "neutral": [],
}

DEFAULT_8P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.VILLAGER],
    "neutral": [],
}

# 标准局配置（9-15人）：狼人:神职:平民:中立 = 3:2:2:1
DEFAULT_9P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID],
}

DEFAULT_10P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID],
}

DEFAULT_11P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID],
}

DEFAULT_12P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID, RoleType.FLUTIST],
}

DEFAULT_13P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.IDIOT, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID],
}

DEFAULT_14P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.IDIOT, RoleType.VILLAGER, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID],
}

DEFAULT_15P_CONFIG = {
    "wolf": [RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF, RoleType.WEREWOLF],
    "good": [RoleType.PROPHET, RoleType.WITCH, RoleType.HUNTER, RoleType.GUARDIAN, RoleType.IDIOT, RoleType.KNIGHT, RoleType.VILLAGER, RoleType.VILLAGER],
    "neutral": [RoleType.CUPID, RoleType.FLUTIST],
}

ROLE_CONFIGS = {
    5: DEFAULT_5P_CONFIG,
    6: DEFAULT_6P_CONFIG,
    7: DEFAULT_7P_CONFIG,
    8: DEFAULT_8P_CONFIG,
    9: DEFAULT_9P_CONFIG,
    10: DEFAULT_10P_CONFIG,
    11: DEFAULT_11P_CONFIG,
    12: DEFAULT_12P_CONFIG,
    13: DEFAULT_13P_CONFIG,
    14: DEFAULT_14P_CONFIG,
    15: DEFAULT_15P_CONFIG,
}


# 特殊狼人配置：每局随机替换的特殊狼人数量
SPECIAL_WOLF_REPLACE = {
    9: 1,   # 3狼中1特殊
    10: 1,  # 4狼中1特殊
    11: 2,  # 4狼中2特殊
    12: 2,  # 4狼中2特殊
    13: 2,  # 5狼中2特殊
    14: 3,  # 5狼中3特殊
    15: 3,  # 5狼中3特殊
}

# 可选的特殊狼人池
SPECIAL_WOLF_POOL = [
    RoleType.WOLF_KING,       # 狼王：死亡带走一人
    RoleType.WHITE_WOLF_KING, # 白狼王：白天自爆带人
    RoleType.WOLF_BEAUTY,     # 狼美人：魅惑带人
    RoleType.EVIL_KNIGHT,     # 恶灵骑士：免疫狼刀/查验反伤
]

# 可选的中立角色池
NEUTRAL_ROLE_POOL = [
    RoleType.CUPID,       # 丘比特：指定情侣
    RoleType.FLUTIST,     # 吹笛人：迷惑玩家
    RoleType.WILD_CHILD,   # 野孩子：偶像死后变狼
    RoleType.FOX,          # 狐狸：查验3人
    RoleType.BEAR,         # 熊：相邻狼人咆哮
]


def get_role_description_summary() -> str:
    """获取角色描述摘要，供Player参考"""
    return """
=== 角色技能说明 ===

【狼人阵营】
- 普通狼人：每晚与狼人一起刀人
- 狼王：死亡时带走一人（被投票不能发动）
- 白狼王：白天可自爆带走一人
- 狼美人：每晚魅惑一人，死亡带走魅惑目标
- 恶灵骑士：免疫狼刀，被查验/毒杀会反伤

【好人阵营】
- 预言家：每晚查验一人（好人/狼人）
- 女巫：解药救人、毒药杀人（不能同晚使用）
- 猎人：死亡时开枪带走一人
- 守卫：每晚守护一人（不能连续守护同一人）
- 白痴：被投票出局可翻牌免死
- 骑士：白天质疑玩家，目标为狼人则狼人死
- 平民：无特殊技能

【中立阵营】
- 丘比特：第一晚指定情侣（人狼链第三方获胜）
- 吹笛人：每晚迷惑两人，迷惑所有人获胜
- 野孩子：第一晚选择偶像，偶像死后变狼人
- 狐狸：每晚查验3人，有狼人则失去技能
- 熊：相邻位置有狼人时咆哮
"""


def get_game_config_summary(player_count: int, config: dict) -> str:
    """获取本局阵容配置摘要"""
    wolf_count = len(config.get("wolf", []))
    good_count = len(config.get("good", []))
    neutral_count = len(config.get("neutral", []))
    special_wolf_count = SPECIAL_WOLF_REPLACE.get(player_count, 0)

    return f"""
=== 本局阵容配置 ===
本局共{player_count}人：
- 狼人阵营：{wolf_count}人（含{'' if special_wolf_count == 0 else special_wolf_count}个特殊狼人）
- 好人阵营：{good_count}人
- 中立阵营：{neutral_count}人
"""


class Referee:
    """裁判类"""

    def __init__(self, player_count: int = 8):
        self.player_count = player_count
        self.game_state: Optional[GameState] = None
        self.is_explosion_mode = player_count < 9  # 屠城局

    def is_explosion_game(self) -> bool:
        """是否为屠城局"""
        return self.is_explosion_mode

    def get_gods(self) -> list[Player]:
        """获取神职玩家（排除平民）"""
        if not self.game_state:
            return []
        god_roles = ["预言家", "女巫", "猎人", "白痴", "守卫", "骑士"]
        return [
            p for p in self.game_state.get_alive_players()
            if p.role and p.role.name in god_roles
        ]

    def get_villagers(self) -> list[Player]:
        """获取平民玩家"""
        if not self.game_state:
            return []
        return [
            p for p in self.game_state.get_alive_players()
            if p.role and p.role.name == "平民"
        ]

    def init_game(self, player_names: list[str]) -> GameState:
        """初始化游戏，分发角色"""
        players = {}
        for i, name in enumerate(player_names[:self.player_count]):
            players[i] = Player(id=i, name=name)

        self.game_state = GameState(players=players)
        self.distribute_roles()
        return self.game_state

    def distribute_roles(self):
        """分发角色（包含随机化逻辑）"""
        if not self.game_state:
            raise ValueError("游戏未初始化")

        if self.player_count in ROLE_CONFIGS:
            config = ROLE_CONFIGS[self.player_count].copy()
        else:
            config = DEFAULT_8P_CONFIG.copy()

        # ========== 随机化特殊狼人 ==========
        if self.player_count >= 9 and self.player_count in SPECIAL_WOLF_REPLACE:
            special_count = SPECIAL_WOLF_REPLACE[self.player_count]
            if special_count > 0:
                # 随机选择要替换的普通狼人数量
                config["wolf"] = self._randomize_wolves(config["wolf"], special_count)

        # ========== 随机化中立角色 ==========
        config["neutral"] = self._randomize_neutral(config["neutral"])

        all_roles = []
        all_roles.extend(config["wolf"])
        all_roles.extend(config["good"])
        all_roles.extend(config["neutral"])

        random.shuffle(all_roles)
        player_ids = list(self.game_state.players.keys())
        random.shuffle(player_ids)

        for i, player_id in enumerate(player_ids):
            role_type = all_roles[i]
            player = self.game_state.players[player_id]

            if role_type in [RoleType.WEREWOLF, RoleType.WOLF_KING,
                           RoleType.WHITE_WOLF_KING, RoleType.WOLF_BEAUTY,
                           RoleType.EVIL_KNIGHT]:
                player.role = create_wolf_role(role_type)
            elif role_type in [RoleType.CUPID, RoleType.WILD_CHILD,
                               RoleType.FLUTIST, RoleType.FOX,
                               RoleType.BEAR, RoleType.BLOOD_MOON]:
                player.role = create_neutral_role(role_type)
                # 记录中立角色ID到game_state
                if role_type == RoleType.CUPID:
                    self.game_state.set_cupid_id(player_id)
                elif role_type == RoleType.FLUTIST:
                    self.game_state.set_flutist_id(player_id)
                elif role_type == RoleType.BLOOD_MOON:
                    self.game_state.set_blood_moon_id(player_id)
                elif role_type == RoleType.WILD_CHILD:
                    self.game_state.set_wild_child_id(player_id)
            else:
                player.role = create_good_role(role_type)

    def _randomize_wolves(self, wolves: list, special_count: int) -> list:
        """随机将部分普通狼人替换为特殊狼人"""
        if special_count <= 0 or special_count >= len(wolves):
            return wolves

        # 随机选择要替换的位置
        replace_indices = random.sample(range(len(wolves)), special_count)

        # 随机选择特殊狼人类型
        special_wolves = random.sample(SPECIAL_WOLF_POOL, min(special_count, len(SPECIAL_WOLF_POOL)))

        new_wolves = wolves.copy()
        for i, idx in enumerate(replace_indices):
            if i < len(special_wolves):
                new_wolves[idx] = special_wolves[i]

        return new_wolves

    def _randomize_neutral(self, neutrals: list) -> list:
        """随机选择中立角色"""
        if not neutrals:
            return neutrals

        # 从角色池中随机选择
        selected = []
        available = NEUTRAL_ROLE_POOL.copy()
        random.shuffle(available)

        for _ in range(len(neutrals)):
            if available:
                selected.append(available.pop())

        return selected

    def set_cupid_link(self, lover1_id: int, lover2_id: int):
        """设置情侣关系"""
        self.game_state.set_lovers(lover1_id, lover2_id)
        self.game_state.love_chain_type = self.game_state.get_love_type(lover1_id)

    def wild_child_convert(self, idol_id: int):
        """野孩子选择偶像"""
        self.game_state.set_wild_child_idol(idol_id)

    def get_effective_camp(self, player_id: int) -> Camp:
        """
        获取玩家实际阵营（考虑野孩子转换）
        """
        player = self.game_state.players.get(player_id)
        if not player or not player.role:
            return Camp.GOOD

        role_name = player.role.name

        # 野孩子阵营判断（优先于其他判断）
        if role_name == "野孩子":
            if self.game_state.is_wild_child_converted():
                return Camp.WOLF
            return Camp.NEUTRAL

        # 狼人阵营判断
        wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
        if role_name in wolf_roles:
            return Camp.WOLF

        # 好人阵营判断
        good_roles = ["平民", "预言家", "女巫", "猎人", "白痴", "守卫", "骑士"]
        if role_name in good_roles:
            return Camp.GOOD

        return Camp.NEUTRAL

    def get_wolves(self) -> list[Player]:
        """获取狼人玩家（包含已转换的野孩子）"""
        wolves = []
        for player in self.game_state.players.values():
            if not player.is_alive():
                continue
            if self.get_effective_camp(player.id) == Camp.WOLF:
                wolves.append(player)
        return wolves

    def get_goods(self) -> list[Player]:
        """获取好人玩家"""
        goods = []
        for player in self.game_state.players.values():
            if not player.is_alive():
                continue
            camp = self.get_effective_camp(player.id)
            if camp == Camp.GOOD:
                goods.append(player)
        return goods

    def is_wolf_role(self, role_name: str) -> bool:
        """判断是否为狼人角色"""
        wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
        return role_name in wolf_roles

    def check_win_condition(self) -> Optional[str]:
        """
        检查胜负条件

        Returns:
            "good": 好人获胜
            "wolf": 狼人获胜
            "neutral": 中立获胜（情侣第三方）
            None: 游戏继续
        """
        if not self.game_state:
            return None

        wolves = self.get_wolves()
        goods = self.get_goods()
        alive_players = self.game_state.get_alive_players()
        total_alive = len(alive_players)

        if total_alive == 0:
            return "wolf"

        # ===== 中立角色获胜判定 =====

        # 1. 情侣第三方获胜判定（人狼链）- 使用is_love_chain_alone方法
        if self.game_state.is_love_chain_alone():
            return "neutral"

        # 2. 吹笛人获胜判定
        if self.game_state.flutist_id is not None:
            flutist = self.game_state.players.get(self.game_state.flutist_id)
            if flutist and flutist.is_alive():
                # 使用GameState的is_cursed方法检查被迷惑状态
                cursed_alive = [p for p in alive_players if self.game_state.is_cursed(p.id)]
                if len(cursed_alive) == total_alive:
                    return "neutral"

        # ===== 好人获胜条件 =====
        if len(wolves) == 0:
            return "good"

        # ===== 狼人获胜条件 =====

        if self.is_explosion_mode:
            # 屠城局：狼人必须杀光所有好人
            if len(goods) == 0:
                return "wolf"
            if len(wolves) >= total_alive / 2:
                return "wolf"
        else:
            # 标准局：狼人可以通过屠边获胜
            # 1. 好人全部死亡
            if len(goods) == 0:
                return "wolf"
            # 2. 屠神：神职全部死亡（预言家、女巫、猎人、守卫、骑士、白痴）
            alive_gods = self.get_gods()
            if len(alive_gods) == 0:
                return "wolf"
            # 3. 屠民：平民全部死亡
            alive_villagers = self.get_villagers()
            if len(alive_villagers) == 0:
                return "wolf"

        return None

    def handle_player_death(self, player_id: int, cause: str = "vote"):
        """
        处理玩家死亡

        Args:
            player_id: 死亡玩家ID
            cause: 死亡原因 (vote/night/poison/shoot)
        """
        if not self.game_state:
            return

        player = self.game_state.players.get(player_id)
        if not player or not player.is_alive():
            return

        # 白痴翻牌处理
        if cause == "vote":
            if player.role and player.role.name == "白痴":
                player.role.can_vote = False
                player.status = PlayerStatus.ALIVE
                return

        player.status = PlayerStatus.DEAD_NIGHT if cause == "night" else PlayerStatus.DEAD_DAY

        # 情侣殉情
        if self.game_state.get_lover(player_id) is not None:
            lover_id = self.game_state.get_lover(player_id)
            if lover_id is not None:
                lover = self.game_state.players.get(lover_id)
                if lover and lover.is_alive():
                    lover.status = PlayerStatus.DEAD_NIGHT
                    if self.game_state.round_history:
                        self.game_state.round_history[-1].deaths.append(lover_id)

        # 狼美人死亡带走魅惑目标
        if player.role and player.role.name == "狼美人":
            charm_target = self.game_state.wolf_beauty_charm_target
            if charm_target is not None:
                target = self.game_state.players.get(charm_target)
                if target and target.is_alive():
                    target.status = PlayerStatus.DEAD_NIGHT
                    if self.game_state.round_history:
                        self.game_state.round_history[-1].deaths.append(charm_target)

    def judge_vote_result(self) -> Optional[int]:
        """判定投票结果"""
        if not self.game_state:
            return None

        max_votes = 0
        candidates = []

        for player in self.game_state.players.values():
            if not player.is_alive():
                continue
            if player.vote_count > max_votes:
                max_votes = player.vote_count
                candidates = [player.id]
            elif player.vote_count == max_votes:
                candidates.append(player.id)

        if len(candidates) > 1:
            return None

        if candidates:
            return candidates[0]

        return None

    def eliminate_player(self, player_id: int, cause: str = "vote"):
        """处决玩家"""
        self.handle_player_death(player_id, cause)

    def get_role_distribution(self) -> dict:
        """获取角色分布信息"""
        if not self.game_state:
            return {}

        distribution = {}
        for player in self.game_state.players.values():
            if player.role:
                role_name = player.role.name
                distribution[role_name] = distribution.get(role_name, 0) + 1

        return distribution