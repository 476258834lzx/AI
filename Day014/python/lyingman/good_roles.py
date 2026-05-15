"""好人阵营角色定义（神职+平民）"""
from .roles import RoleType, Camp, ROLE_DISPLAY_NAMES
from .game_state import Role


# ===== 神职角色 =====
def get_prophet_role() -> Role:
    """预言家 - 每晚查验一人是好人还是狼人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.PROPHET],
        description="每晚可以查验一名玩家的身份，查验结果是好人或狼人。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_witch_role() -> Role:
    """女巫 - 拥有解药和毒药"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WITCH],
        description="拥有一瓶解药和一瓶毒药。解药可以救人，毒药可以杀人。不能同晚使用。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_hunter_role() -> Role:
    """猎人 - 死亡时可带走一人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.HUNTER],
        description="死亡时可以开枪带走一名玩家，但如果被女巫毒死则不能开枪。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
        can_shoot_on_death=True,
    )


def get_idiot_role() -> Role:
    """白痴 - 被投票出局时翻牌免死"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.IDIOT],
        description="被投票出局时可以翻牌免死，继续存活但失去投票权，仍可发言。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=False,  # 白痴不能当警长
    )


def get_guardian_role() -> Role:
    """守卫 - 每晚守护一人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.GUARDIAN],
        description="每晚可以守护一名玩家，被守护者免疫狼刀。不能连续两晚守护同一人。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_knight_role() -> Role:
    """骑士 - 可质疑玩家"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.KNIGHT],
        description="白天可以质疑一名玩家。如果目标是狼人，狼人立即死亡；如果不是，骑士自己死亡。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
    )


# ===== 平民角色 =====
def get_villager_role() -> Role:
    """平民 - 无特殊技能"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.VILLAGER],
        description="没有任何特殊能力的普通村民。",
        camp=Camp.GOOD,
        can_vote=True,
        can_be_sheriff=True,
    )


# 角色创建函数映射
GOOD_ROLE_CREATORS = {
    RoleType.VILLAGER: get_villager_role,
    RoleType.PROPHET: get_prophet_role,
    RoleType.WITCH: get_witch_role,
    RoleType.HUNTER: get_hunter_role,
    RoleType.IDIOT: get_idiot_role,
    RoleType.GUARDIAN: get_guardian_role,
    RoleType.KNIGHT: get_knight_role,
}


def create_good_role(role_type: RoleType) -> Role:
    """创建好人阵营角色"""
    if role_type in GOOD_ROLE_CREATORS:
        return GOOD_ROLE_CREATORS[role_type]()
    raise ValueError(f"Unknown good role: {role_type}")