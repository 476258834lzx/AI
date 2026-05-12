"""玩家Agent - 使用LangChain+ollama实现"""
import json
from typing import Optional, Callable
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from .game_state import GameState, Player
from .roles import RoleType
from .tools import get_available_tools
from .emotion_skills import get_player_emotion_system, create_emotion_system
from .config import get_llm_config


class PlayerAgent:
    """玩家Agent"""

    def __init__(
        self,
        player_id: int,
        player_name: str,
        model: str = None,
    ):
        self.player_id = player_id
        self.player_name = player_name
        self.model = model or get_llm_config()["model"]

        # 使用ollama
        llm_config = get_llm_config()
        self.llm = ChatOllama(
            model=llm_config["model"],
            base_url=llm_config["base_url"],
            temperature=llm_config.get("temperature", 0.7),
            keep_alive="5m",
        )

        self.conversation_history = []
        self.game_state: Optional[GameState] = None
        self.emotion_system = create_emotion_system(
            player_id,
            primary_style="rational",
        )

    def set_game_state(self, game_state: GameState):
        """设置游戏状态"""
        self.game_state = game_state

    def get_role(self) -> Optional[str]:
        """获取自己的角色"""
        if self.game_state:
            player = self.game_state.players.get(self.player_id)
            if player and player.role:
                return player.role.name
        return None

    def get_available_tools(self) -> list:
        """获取可用工具"""
        role = self.get_role()
        if role:
            return get_available_tools(role)
        return []

    def get_system_prompt(self) -> str:
        """生成系统提示"""
        role = self.get_role()
        role_desc = ""
        if role:
            role_desc = f"你的角色是【{role}】"

        return f"""你是狼人杀游戏中的一名玩家。

{role_desc}

你的任务是：
1. 根据当前游戏状态和你的角色做出最佳决策
2. 使用合适的技能（如果你是神职或狼人）
3. 在发言环节发表符合角色和策略的发言

游戏规则：
- 狼人阵营：消灭所有好人
- 好人阵营：找出并消灭所有狼人
- 中立阵营：根据角色技能达成特定胜利条件
- 每个人可以投票、发言
- 神职有特殊技能

发言时请注意：
- 保持角色一致性
- 适当使用情绪技巧增加感染力
- 考虑战术配合

{self.emotion_system.get_speech_prompt({})}"""

    def get_current_context(self) -> str:
        """获取当前游戏上下文"""
        if not self.game_state:
            return "游戏未开始"

        player = self.game_state.players.get(self.player_id)
        alive_players = self.game_state.get_alive_players()

        context = f"""=== 当前游戏状态 ===
第{self.game_state.current_day}天
阶段: {self.game_state.current_phase.value}
你的ID: {self.player_id}

存活玩家: {len(alive_players)}人
"""
        for p in alive_players:
            role_info = f"【{p.role.name}】" if p.role and p.id == self.player_id else ""
            context += f"  ID:{p.id} {p.name} {role_info}\n"

        return context

    def act(self, action_type: str, **kwargs) -> dict:
        """
        执行动作

        Args:
            action_type: 动作类型 (speak, vote, skill, etc.)
            **kwargs: 其他参数

        Returns:
            动作结果
        """
        if action_type == "speak":
            return self.speak(kwargs.get("prompt", ""))
        elif action_type == "vote":
            return self.vote(kwargs.get("candidates", []))
        elif action_type == "skill":
            return self.use_skill(kwargs.get("skill_name", ""), kwargs.get("target_id", None))
        else:
            return {"error": f"未知动作类型: {action_type}"}

    def speak(self, prompt: str = "") -> dict:
        """发言"""
        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
        ]

        if prompt:
            messages.append(("human", prompt))

        messages.append(("human", "请发表你的发言："))

        response = self.llm.invoke(messages)
        speech = response.content if hasattr(response, "content") else str(response)

        # 应用情绪调整
        speech = self.emotion_system.generate_speech_with_emotion(speech)

        return {
            "player_id": self.player_id,
            "speech": speech,
        }

    def vote(self, candidates: list[int]) -> int:
        """投票"""
        if not candidates:
            return -1

        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""请投票。候选人：
{chr(10).join(candidates_info)}

请选择你要投票的玩家ID（只回复数字）："""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # 解析投票目标
        try:
            # 提取数字
            import re
            numbers = re.findall(r'\d+', content)
            for num in numbers:
                target_id = int(num)
                if target_id in candidates:
                    return target_id
        except (ValueError, re.error):
            # 解析失败，默认选择第一个
            pass

        # 默认选择第一个
        return candidates[0]

    def use_skill(self, skill_name: str, target_id: Optional[int] = None) -> dict:
        """使用技能"""
        tools = self.get_available_tools()
        tool_map = {t.name.replace("_", " "): t for t in tools}

        if skill_name not in tool_map:
            return {"error": f"技能 {skill_name} 不可用"}

        tool_func = tool_map[skill_name]

        # 构建工具参数
        args = {"game_state": self.game_state.__dict__, "player_id": self.player_id}
        if target_id is not None:
            args["target_id"] = target_id

        try:
            result = tool_func.invoke(args)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def decide_sheriff_run(self) -> bool:
        """决定是否参加警长竞选"""
        role = self.get_role()

        # 强神通常上警
        if role in ["预言家", "女巫", "猎人"]:
            return True

        # 狼人可能悍跳
        if role and role.endswith("狼"):
            return True

        return False

    def decide_night_action(self, night_phase: str) -> Optional[int]:
        """决定夜间动作目标"""
        role = self.get_role()

        if night_phase == "wolf_kill":
            # 狼人选择刀人目标
            alive = self.game_state.get_alive_players()
            # 优先刀神
            for p in alive:
                if p.role and p.role.name in ["预言家", "女巫", "猎人"]:
                    return p.id
            return alive[0].id if alive else None

        elif night_phase == "seer_check":
            # 预言家查验
            alive = self.game_state.get_alive_players()
            for p in alive:
                if p.id != self.player_id:
                    return p.id
            return None

        elif night_phase == "witch_heal":
            # 女巫救人（如果有人被刀）
            if self.game_state.wolf_kill_target and not self.game_state.witch_heal_used:
                return self.game_state.wolf_kill_target
            return None

        elif night_phase == "guardian_protect":
            # 守卫守护
            alive = self.game_state.get_alive_players()
            # 优先守神
            for p in alive:
                if p.role and p.role.name in ["预言家", "女巫"]:
                    if p.id != self.game_state.guardian_last_protect:
                        return p.id
            return alive[0].id if alive else None

        elif night_phase == "witch_poison":
            # 女巫毒人（AI默认跳过）
            return None

        elif night_phase == "hunter_choice":
            # 猎人开枪（AI默认跳过）
            return None

        return None

    def think(self, question: str) -> str:
        """思考/分析"""
        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", f"问题: {question}\n请分析并回答："),
        ]

        response = self.llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)


class HumanPlayerAgent(PlayerAgent):
    """人类玩家Agent（用于游玩模式）"""

    def __init__(self, player_id: int, player_name: str, input_func: Optional[Callable] = None):
        super().__init__(player_id, player_name)
        self.input_func = input_func or input

    def speak(self, prompt: str = "") -> dict:
        """人类玩家发言"""
        print(f"\n[{self.player_name}] 请发言：")
        if prompt:
            print(f"提示: {prompt}")

        speech = self.input_func("> ")
        return {
            "player_id": self.player_id,
            "speech": speech,
        }

    def vote(self, candidates: list[int]) -> int:
        """人类玩家投票"""
        print(f"\n[{self.player_name}] 请投票：")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            try:
                choice = self.input_func("选择玩家ID: ")
                target_id = int(choice)
                if target_id in candidates:
                    return target_id
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def decide_sheriff_run(self) -> bool:
        """人类玩家决定是否上警"""
        role = self.get_role()
        print(f"\n[{self.player_name}] 警长竞选阶段")
        print(f"你的角色是: {role}")

        while True:
            choice = self.input_func("是否参加警长竞选？(y/n): ").lower()
            if choice in ["y", "yes", "是", "参加"]:
                return True
            elif choice in ["n", "no", "否", "不参加"]:
                return False
            print("无效输入")

    def decide_night_action(self, night_phase: str) -> Optional[int]:
        """人类玩家夜间动作"""
        role = self.get_role()
        print(f"\n[{self.player_name}] 夜间动作 - {night_phase}")
        print(f"你的角色是: {role}")

        alive = self.game_state.get_alive_players()
        print("存活玩家:")
        for p in alive:
            if p.id != self.player_id:
                print(f"  {p.id}: {p.name}")

        while True:
            choice = self.input_func(f"选择目标玩家ID（跳过请输入-1）: ")
            try:
                target_id = int(choice)
                if target_id == -1:
                    return None
                if any(p.id == target_id for p in alive):
                    return target_id
                print("无效选择")
            except ValueError:
                print("请输入数字")
