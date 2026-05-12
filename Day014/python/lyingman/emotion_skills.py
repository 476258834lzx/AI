"""情绪技能系统 - 集成qiqing-liuyu的七情六欲发言技能"""
from typing import Optional
from langchain_core.tools import tool


# 发言风格权重配置
SPEECH_STYLES = {
    "performance": {
        "name": "表演型",
        "description": "善于扮演角色，演技出色",
        "weight": 0.8,
    },
    "agitation": {
        "name": "煽动型",
        "description": "善于说服他人，影响力强",
        "weight": 0.9,
    },
    "pressure": {
        "name": "压力型",
        "description": "善于施压，让对手紧张",
        "weight": 0.7,
    },
    "pua": {
        "name": "PUA型",
        "description": "善于心理操控，话术精妙",
        "weight": 0.6,
    },
    "rational": {
        "name": "理性型",
        "description": "逻辑清晰，分析冷静",
        "weight": 0.85,
    },
}

# 情绪类型
EMOTIONS = {
    "joy": "开心",
    "anger": "愤怒",
    "sadness": "悲伤",
    "fear": "恐惧",
    "surprise": "惊讶",
    "disgust": "厌恶",
    "trust": "信任",
    "anticipation": "期待",
}

# 狼人杀策略类型
STRATEGIES = {
    "self_kill": {
        "name": "自刀",
        "description": "狼人第一晚刀自己，博取信任",
    },
    "reverse_hook": {
        "name": "倒钩",
        "description": "狼人假装好人，站边真预言家",
    },
    "mutual_protect": {
        "name": "互保",
        "description": "狼人之间互相保护",
    },
    "reverse_water": {
        "name": "反水立警",
        "description": "反对假预言家来获取好人信任",
    },
    "bluff": {
        "name": "悍跳",
        "description": "狼人跳神职与真神职对抗",
    },
    "calm": {
        "name": "潜水",
        "description": "低调发言，减少注意力",
    },
    "lead": {
        "name": "号票",
        "description": "主导投票方向",
    },
}


class EmotionSystem:
    """情绪技能系统"""

    def __init__(
        self,
        primary_style: str = "rational",
        secondary_style: Optional[str] = None,
        emotion_tendency: Optional[dict] = None,
        strategy_tendency: Optional[dict] = None,
    ):
        """
        初始化情绪系统

        Args:
            primary_style: 主要发言风格
            secondary_style: 次要发言风格
            emotion_tendency: 情绪倾向 {emotion: weight}
            strategy_tendency: 策略倾向 {strategy: weight}
        """
        self.primary_style = primary_style
        self.secondary_style = secondary_style
        self.emotion_tendency = emotion_tendency or {}
        self.strategy_tendency = strategy_tendency or {}

        # 初始化七情强度
        self.emotion_intensity = {
            emotion: 0.5 for emotion in EMOTIONS
        }

    def update_emotion(self, emotion: str, delta: float):
        """更新情绪强度"""
        if emotion in self.emotion_intensity:
            self.emotion_intensity[emotion] = max(
                0.0, min(1.0, self.emotion_intensity[emotion] + delta)
            )

    def get_speech_prompt(self, context: dict) -> str:
        """生成带情绪的发言提示"""
        style_info = SPEECH_STYLES[self.primary_style]

        prompt_parts = [
            f"你的发言风格是【{style_info['name']}】({style_info['description']})",
        ]

        if self.secondary_style:
            secondary_info = SPEECH_STYLES[self.secondary_style]
            prompt_parts.append(f"次要风格：{secondary_info['name']}")

        # 添加当前情绪状态
        active_emotions = [
            f"{EMOTIONS[em]}（{int(intensity*100)}%）"
            for em, intensity in self.emotion_intensity.items()
            if intensity > 0.6
        ]
        if active_emotions:
            prompt_parts.append(f"当前情绪状态：{', '.join(active_emotions)}")

        return "\n".join(prompt_parts)

    def generate_speech_with_emotion(
        self,
        base_speech: str,
        emotion: Optional[str] = None,
    ) -> str:
        """根据情绪调整发言"""
        if emotion:
            self.update_emotion(emotion, 0.2)

        # 根据情绪添加语气词
        emotion_markers = {
            "joy": "太好了！",
            "anger": "真是气死我了！",
            "sadness": "唉...",
            "fear": "我有点担心...",
            "surprise": "什么？！",
            "disgust": "真是恶心",
            "trust": "我相信你",
            "anticipation": "期待...",
        }

        if emotion and emotion in emotion_markers:
            prefix = emotion_markers[emotion]
            if not base_speech.startswith(prefix):
                base_speech = f"{prefix}{base_speech}"

        return base_speech

    def choose_strategy(self, context: dict) -> str:
        """根据情境选择策略"""
        # 根据当前游戏状态选择策略
        day = context.get("day", 1)
        is_wolf = context.get("is_wolf", False)
        is_prophet_alive = context.get("prophet_alive", True)

        # 狼人策略
        if is_wolf:
            if day == 1:
                # 第一天策略选择
                weights = {
                    "self_kill": 0.2,
                    "reverse_hook": 0.3,
                    "bluff": 0.3,
                    "calm": 0.2,
                }
            else:
                weights = {
                    "reverse_hook": 0.4,
                    "mutual_protect": 0.2,
                    "bluff": 0.3,
                    "calm": 0.1,
                }
        else:
            # 好人策略
            if not is_prophet_alive:
                weights = {
                    "reverse_water": 0.4,
                    "lead": 0.3,
                    "calm": 0.3,
                }
            else:
                weights = {
                    "lead": 0.4,
                    "reverse_water": 0.3,
                    "calm": 0.3,
                }

        # 根据倾向调整权重
        for strategy, tendency in self.strategy_tendency.items():
            if strategy in weights:
                weights[strategy] += tendency * 0.1

        # 选择策略
        import random
        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0
        for strategy, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return strategy
        return "calm"


# 全局情绪系统实例（按玩家ID存储）
_player_emotion_systems: dict[int, EmotionSystem] = {}


def get_player_emotion_system(player_id: int) -> EmotionSystem:
    """获取玩家情绪系统"""
    if player_id not in _player_emotion_systems:
        _player_emotion_systems[player_id] = EmotionSystem()
    return _player_emotion_systems[player_id]


def create_emotion_system(
    player_id: int,
    primary_style: str = "rational",
    secondary_style: Optional[str] = None,
) -> EmotionSystem:
    """为玩家创建情绪系统"""
    system = EmotionSystem(
        primary_style=primary_style,
        secondary_style=secondary_style,
    )
    _player_emotion_systems[player_id] = system
    return system


@tool
def get_speech_guidance(player_id: int) -> str:
    """
    获取当前玩家的发言指导。

    Args:
        player_id: 玩家ID

    Returns:
        包含发言风格和情绪状态的指导文本
    """
    system = get_player_emotion_system(player_id)
    context = {}  # 需要传入当前游戏上下文
    return system.get_speech_prompt(context)


@tool
def adjust_emotion(player_id: int, emotion: str, delta: float) -> str:
    """
    调整玩家情绪。

    Args:
        player_id: 玩家ID
        emotion: 情绪类型 (joy/anger/sadness/fear/surprise/disgust/trust/anticipation)
        delta: 情绪变化量 (-1.0 到 1.0)

    Returns:
        调整后的情绪状态
    """
    system = get_player_emotion_system(player_id)
    system.update_emotion(emotion, delta)
    active = [
        f"{EMOTIONS[em]}: {int(system.emotion_intensity[em]*100)}%"
        for em in system.emotion_intensity
        if system.emotion_intensity[em] > 0.5
    ]
    return f"当前活跃情绪: {', '.join(active)}" if active else "情绪平稳"


@tool
def apply_emotion_to_speech(player_id: int, speech: str, emotion: str) -> str:
    """
    根据情绪调整发言内容。

    Args:
        player_id: 玩家ID
        speech: 原始发言
        emotion: 要添加的情绪类型

    Returns:
        添加情绪后的发言
    """
    system = get_player_emotion_system(player_id)
    return system.generate_speech_with_emotion(speech, emotion)


@tool
def get_strategy_recommendation(player_id: int, context: str) -> str:
    """
    获取策略建议。

    Args:
        player_id: 玩家ID
        context: 当前游戏情境描述

    Returns:
        推荐的策略类型和描述
    """
    system = get_player_emotion_system(player_id)
    strategy = system.choose_strategy(eval(context) if context else {})
    strategy_info = STRATEGIES[strategy]
    return f"推荐策略【{strategy_info['name']}】：{strategy_info['description']}"
