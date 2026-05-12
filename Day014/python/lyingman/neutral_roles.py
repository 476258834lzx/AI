"""中立阵营角色定义"""
from .roles import RoleType, Camp, ROLE_DISPLAY_NAMES
from .game_state import Role


def get_cupid_role() -> Role:
    """丘比特 - 指定情侣"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.CUPID],
        description="第一晚可以指定两名玩家成为情侣。情侣一方死亡，另一方殉情。如果情侣是人和狼，则成为第三方。",
        can_vote=True,
        can_be_sheriff=True,
    )


def get_wild_child_role() -> Role:
    """野孩子 - 选偶像"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.WILD_CHILD],
        description="第一晚选择一个偶像，之后跟随偶像的阵营。如果偶像死亡，野孩子变为狼人阵营。",
        can_vote=True,
        can_be_sheriff=True,
    )


def get_flutist_role() -> Role:
    """吹笛人 - 迷惑玩家"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.FLUTIST],
        description="每晚可以迷惑两名玩家。被迷惑的玩家失去投票权。全部存活玩家被迷惑时吹笛人获胜。",
        can_vote=True,
        can_be_sheriff=True,
    )


def get_fox_role() -> Role:
    """狐狸 - 查验3连坐"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.FOX],
        description="每晚可以查验3个相邻位置的玩家。如果其中有狼人，则失去技能；如果没有，狐狸免疫狼刀。",
        can_vote=True,
        can_be_sheriff=True,
    )


def get_bear_role() -> Role:
    """熊 - 相邻有狼则咆哮"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.BEAR],
        description="如果熊的相邻位置有狼人存活，法官会在白天宣布熊咆哮了。熊死亡后不再咆哮。",
        can_vote=True,
        can_be_sheriff=True,
    )


def get_blood_moon_role() -> Role:
    """血月使徒 - 狼人全死则血月获胜"""
    return Role(
        name=ROLE_DISPLAY_NAMES[RoleType.BLOOD_MOON],
        description="死亡时，如果场上狼人全部死亡，血月使徒复活并获胜。",
        can_vote=True,
        can_be_sheriff=True,
    )


# 角色创建函数映射
NEUTRAL_ROLE_CREATORS = {
    RoleType.CUPID: get_cupid_role,
    RoleType.WILD_CHILD: get_wild_child_role,
    RoleType.FLUTIST: get_flutist_role,
    RoleType.FOX: get_fox_role,
    RoleType.BEAR: get_bear_role,
    RoleType.BLOOD_MOON: get_blood_moon_role,
}


def create_neutral_role(role_type: RoleType) -> Role:
    """创建中立阵营角色"""
    if role_type in NEUTRAL_ROLE_CREATORS:
        return NEUTRAL_ROLE_CREATORS[role_type]()
    raise ValueError(f"Unknown neutral role: {role_type}")