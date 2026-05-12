"""角色基类和枚举定义"""
from enum import Enum


class Camp(Enum):
    """阵营"""
    GOOD = "good"      # 好人阵营
    WOLF = "wolf"      # 狼人阵营
    NEUTRAL = "neutral"  # 中立阵营


class RoleType(Enum):
    """角色类型"""

    # ===== 神职阵营（好人）=====
    VILLAGER = "villager"        # 平民
    PROPHET = "prophet"          # 预言家
    WITCH = "witch"             # 女巫
    HUNTER = "hunter"            # 猎人
    IDIOT = "idiot"              # 白痴
    GUARDIAN = "guardian"        # 守卫
    KNIGHT = "knight"            # 骑士

    # ===== 狼人阵营 =====
    WEREWOLF = "werewolf"        # 狼人
    WOLF_KING = "wolf_king"      # 狼王/狼枪
    WHITE_WOLF_KING = "white_wolf_king"  # 白狼王
    WOLF_BEAUTY = "wolf_beauty"  # 狼美人
    EVIL_KNIGHT = "evil_knight"  # 恶灵骑士

    # ===== 中立阵营 =====
    CUPID = "cupid"              # 丘比特
    WILD_CHILD = "wild_child"    # 野孩子
    FLUTIST = "flutist"          # 吹笛人
    FOX = "fox"                  # 狐狸
    BEAR = "bear"                # 熊
    BLOOD_MOON = "blood_moon"    # 血月使徒

    def get_camp(self) -> Camp:
        """获取角色阵营"""
        good_roles = {RoleType.VILLAGER, RoleType.PROPHET, RoleType.WITCH,
                      RoleType.HUNTER, RoleType.IDIOT, RoleType.GUARDIAN, RoleType.KNIGHT}
        wolf_roles = {RoleType.WEREWOLF, RoleType.WOLF_KING, RoleType.WHITE_WOLF_KING,
                      RoleType.WOLF_BEAUTY, RoleType.EVIL_KNIGHT}
        neutral_roles = {RoleType.CUPID, RoleType.WILD_CHILD, RoleType.FLUTIST,
                         RoleType.FOX, RoleType.BEAR, RoleType.BLOOD_MOON}

        if self in good_roles:
            return Camp.GOOD
        elif self in wolf_roles:
            return Camp.WOLF
        else:
            return Camp.NEUTRAL

    def is_good(self) -> bool:
        return self.get_camp() == Camp.GOOD

    def is_wolf(self) -> bool:
        return self.get_camp() == Camp.WOLF

    def is_neutral(self) -> bool:
        return self.get_camp() == Camp.NEUTRAL


# 角色显示名称映射
ROLE_DISPLAY_NAMES = {
    RoleType.VILLAGER: "平民",
    RoleType.PROPHET: "预言家",
    RoleType.WITCH: "女巫",
    RoleType.HUNTER: "猎人",
    RoleType.IDIOT: "白痴",
    RoleType.GUARDIAN: "守卫",
    RoleType.KNIGHT: "骑士",
    RoleType.WEREWOLF: "狼人",
    RoleType.WOLF_KING: "狼王",
    RoleType.WHITE_WOLF_KING: "白狼王",
    RoleType.WOLF_BEAUTY: "狼美人",
    RoleType.EVIL_KNIGHT: "恶灵骑士",
    RoleType.CUPID: "丘比特",
    RoleType.WILD_CHILD: "野孩子",
    RoleType.FLUTIST: "吹笛人",
    RoleType.FOX: "狐狸",
    RoleType.BEAR: "熊",
    RoleType.BLOOD_MOON: "血月使徒",
}