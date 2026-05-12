"""狼人杀游戏"""
from .game_state import (
    Player, Role, GamePhase, RoundPhase, NightPhase, PlayerStatus,
    NightAction, DayAction, Round, GameState
)
from .roles import RoleType, Camp, ROLE_DISPLAY_NAMES
from .good_roles import create_good_role
from .wolf_roles import create_wolf_role
from .neutral_roles import create_neutral_role
from .player_agent import PlayerAgent, HumanPlayerAgent
from .referee import Referee
from .god import God
from .graph import WerewolfGraph
from .config import get_llm_config, get_embedding_config
from .knowledge_base import KnowledgeBase, init_knowledge_base, GAME_REPLAYS
from .emotion_skills import (
    EmotionSystem, get_player_emotion_system, create_emotion_system,
    SPEECH_STYLES, EMOTIONS, STRATEGIES,
    get_speech_guidance, adjust_emotion, apply_emotion_to_speech, get_strategy_recommendation
)
from .tools import get_available_tools, get_night_action_tools

__all__ = [
    # 状态
    "Player", "Role", "GamePhase", "RoundPhase", "NightPhase", "PlayerStatus",
    "NightAction", "DayAction", "Round", "GameState",
    # 角色
    "RoleType", "Camp", "ROLE_DISPLAY_NAMES",
    "create_good_role", "create_wolf_role", "create_neutral_role",
    # Agent
    "PlayerAgent", "HumanPlayerAgent",
    # 游戏组件
    "Referee", "God", "WerewolfGraph",
    # 配置
    "get_llm_config", "get_embedding_config",
    # 知识库
    "KnowledgeBase", "init_knowledge_base", "GAME_REPLAYS",
    # 情绪系统
    "EmotionSystem", "get_player_emotion_system", "create_emotion_system",
    "SPEECH_STYLES", "EMOTIONS", "STRATEGIES",
    "get_speech_guidance", "adjust_emotion", "apply_emotion_to_speech", "get_strategy_recommendation",
    # 技能
    "get_available_tools", "get_night_action_tools",
]