"""狼人阵营角色定义"""
from .roles import RoleType, Camp, ROLE_DISPLAY_NAMES
from .game_state import Role


def get_werewolf_role() -> Role:
    """狼人 - 每晚刀人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WEREWOLF],
        description="每晚可以与其他狼人一起选择刀杀一名玩家。",
        camp=Camp.WOLF,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_wolf_king_role() -> Role:
    """狼王 - 死亡带走一人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WOLF_KING],
        description="死亡时可以开枪带走一名玩家。被投票出局时无法发动技能。",
        camp=Camp.WOLF,
        can_vote=True,
        can_be_sheriff=True,
        can_shoot_on_death=True,
    )


def get_white_wolf_king_role() -> Role:
    """白狼王 - 自爆带人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WHITE_WOLF_KING],
        description="白天可以自爆，自爆时可以选择带走一名玩家。",
        camp=Camp.WOLF,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_wolf_beauty_role() -> Role:
    """狼美人 - 魅惑带人"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WOLF_BEAUTY],
        description="每晚可以魅惑一名玩家。死亡时自动带走被魅惑的玩家。",
        camp=Camp.WOLF,
        can_vote=True,
        can_be_sheriff=True,
    )


def get_evil_knight_role() -> Role:
    """恶灵骑士 - 不被查验，死亡随机带走"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.EVIL_KNIGHT],
        description="预言家查验结果显示为好人。死亡时随机带走一名玩家。",
        camp=Camp.WOLF,
        can_vote=True,
        can_be_sheriff=True,
        can_shoot_on_death=True,
    )


# 角色创建函数映射
WOLF_ROLE_CREATORS = {
    RoleType.WEREWOLF: get_werewolf_role,
    RoleType.WOLF_KING: get_wolf_king_role,
    RoleType.WHITE_WOLF_KING: get_white_wolf_king_role,
    RoleType.WOLF_BEAUTY: get_wolf_beauty_role,
    RoleType.EVIL_KNIGHT: get_evil_knight_role,
}


def create_wolf_role(role_type: RoleType) -> Role:
    """创建狼人阵营角色"""
    if role_type in WOLF_ROLE_CREATORS:
        return WOLF_ROLE_CREATORS[role_type]()
    raise ValueError(f"Unknown wolf role: {role_type}")