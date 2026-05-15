"""玩家Agent - 支持vllm/ollama/sglang多种后端"""
import json
from typing import Optional, Callable
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from .game_state import GameState, Player
from .roles import RoleType
from .tools import get_available_tools
from .emotion_skills import get_player_emotion_system, create_emotion_system
from .config import get_llm_config, LLMProvider


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
        self.conversation_history = []
        self.game_state: Optional[GameState] = None
        self.emotion_system = create_emotion_system(
            player_id,
            primary_style="rational",
        )

        self._init_llm()

    def _init_llm(self):
        """初始化LLM客户端"""
        llm_config = get_llm_config()
        provider = llm_config.get("provider", "ollama")

        if provider == "vllm" or provider == "sglang":
            self.llm = ChatOpenAI(
                model=llm_config["model"],
                base_url=llm_config["base_url"],
                api_key="EMPTY",
                temperature=llm_config.get("temperature", 0.7),
            )
        else:
            self.llm = ChatOllama(
                model=llm_config["model"],
                base_url=llm_config["base_url"],
                temperature=llm_config.get("temperature", 0.7),
                keep_alive="5m",
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

    def wolf_discuss(self, wolf_ids: list[int], non_wolf_targets: list[int],
                     other_wolf_suggestions: dict[int, int] = None) -> dict:
        """狼人夜间商讨（互认身份并讨论刀人目标）"""
        other_wolf_suggestions = other_wolf_suggestions or {}

        # 构建狼人队友信息
        wolf_info = []
        for wid in wolf_ids:
            if wid != self.player_id:
                wolf_player = self.game_state.players.get(wid)
                if wolf_player:
                    wolf_info.append(f"ID:{wid} {wolf_player.name}")

        # 构建其他狼人的提议
        suggestions_info = []
        for wid, target in other_wolf_suggestions.items():
            if wid != self.player_id:
                target_player = self.game_state.players.get(target)
                suggester = self.game_state.players.get(wid)
                if target_player and suggester:
                    suggestions_info.append(f"{suggester.name}提议刀{target_player.name}(ID:{target})")

        candidates_info = []
        for tid in non_wolf_targets:
            player = self.game_state.players.get(tid)
            if player:
                candidates_info.append(f"ID:{tid} {player.name}")

        prompt = f"""作为狼人，你需要与其他狼人商讨今晚刀谁。

=== 狼人队友 ===
{chr(10).join(wolf_info) if wolf_info else "你是唯一的狼人"}

=== 其他狼人的提议 ===
{chr(10).join(suggestions_info) if suggestions_info else "暂无提议"}

=== 可刀目标 ===
{chr(10).join(candidates_info)}

请分析局势，选择一个最合适的目标，并在回复中明确说出你选择刀哪个ID的玩家。

格式要求：回复中必须包含"我选择刀 ID:数字"，例如"我选择刀 ID:3"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # 解析目标ID
        import re
        match = re.search(r'ID[：:]\s*(\d+)', content)
        if match:
            target_id = int(match.group(1))
            if target_id in non_wolf_targets:
                return {"player_id": self.player_id, "target_id": target_id, "reason": content}

        # 默认选择第一个
        if non_wolf_targets:
            return {"player_id": self.player_id, "target_id": non_wolf_targets[0], "reason": content}

        return {"player_id": self.player_id, "target_id": None, "reason": content}

    def wolf_confirm_kill(self, target_id: int) -> dict:
        """狼人确认刀人"""
        target_player = self.game_state.players.get(target_id)
        target_name = target_player.name if target_player else f"ID:{target_id}"

        return {
            "player_id": self.player_id,
            "target_id": target_id,
            "confirmed": True,
            "message": f"确认刀 {target_name}"
        }

    def decide_hunter_shoot(self, candidates: list[int]) -> Optional[int]:
        """猎人选择开枪目标"""
        if not candidates:
            return None

        # 构建候选目标信息
        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""你是猎人，今晚你死亡了。你可以选择开枪带走一名玩家。

=== 可开枪目标 ===
{chr(10).join(candidates_info)}

请分析局势，选择一个最合适的目标带走。如果选择不开枪，请说"不开枪"。

格式要求：回复中必须包含"我选择开枪 ID:数字"或"我选择不开枪"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # 检查是否选择不开枪
        if "不开枪" in content:
            return None

        # 解析目标ID
        import re
        match = re.search(r'ID[：:]\s*(\d+)', content)
        if match:
            target_id = int(match.group(1))
            if target_id in candidates:
                return target_id

        # 默认不开枪
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

    def decide_cupid_link(self, candidates: list[int]) -> dict:
        """丘比特选择情侣"""
        if len(candidates) < 2:
            return {}

        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""你是丘比特，第一晚你需要指定两名玩家成为情侣。情侣一方死亡，另一方殉情。如果是人狼链，你们将成为第三方阵营获胜。

=== 可选玩家 ===
{chr(10).join(candidates_info)}

请选择两名玩家作为情侣。

格式要求：回复中必须包含"我选择情侣: ID1和ID2"，例如"我选择情侣: 2和5"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import re
        # 提取所有数字ID并转换为整数
        ids = []
        for x in re.findall(r'\d+', content):
            try:
                ids.append(int(x))
            except ValueError:
                pass

        if len(ids) >= 2:
            # 返回前两个有效的候选ID
            valid_ids = [cid for cid in ids if cid in candidates]
            if len(valid_ids) >= 2:
                return {"lover1": valid_ids[0], "lover2": valid_ids[1]}
            # 如果没有完全有效的，返回前两个并依赖后续检查
            return {"lover1": ids[0], "lover2": ids[1]}

        return {"lover1": candidates[0], "lover2": candidates[1]}

    def decide_wild_child_idol(self, candidates: list[int]) -> dict:
        """野孩子选择偶像"""
        if not candidates:
            return {}

        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""你是野孩子，第一晚你需要选择一个偶像。之后你会跟随偶像的阵营。如果偶像死亡，你将变成狼人阵营。

=== 可选玩家 ===
{chr(10).join(candidates_info)}

请分析局势，选择一个最合适的偶像。

格式要求：回复中必须包含"我选择偶像: ID"，例如"我选择偶像: 3"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import re
        match = re.search(r'偶像[：:]\s*(\d+)', content)
        if match:
            return {"idol": int(match.group(1))}

        return {"idol": candidates[0] if candidates else None}

    def decide_flutist_charm(self, candidates: list[int]) -> dict:
        """吹笛人选择迷惑目标"""
        if len(candidates) < 2:
            return {}

        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""你是吹笛人，每晚可以迷惑两名玩家。被迷惑的玩家失去投票权。当所有存活玩家都被迷惑时，你获胜。

=== 可选迷惑目标 ===
{chr(10).join(candidates_info)}

请分析局势，选择两个最合适的目标迷惑。

格式要求：回复中必须包含"我迷惑: ID1和ID2"，例如"我迷惑: 1和4"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import re
        ids = []
        for x in re.findall(r'\d+', content):
            try:
                ids.append(int(x))
            except ValueError:
                pass

        if len(ids) >= 2:
            valid_ids = [cid for cid in ids if cid in candidates]
            if len(valid_ids) >= 2:
                return {"target1": valid_ids[0], "target2": valid_ids[1]}
            return {"target1": ids[0], "target2": ids[1]}

        return {"target1": candidates[0], "target2": candidates[1] if len(candidates) > 1 else None}

    def decide_fox_check(self, candidates: list[int]) -> dict:
        """狐狸选择查验目标（3个相邻位置）"""
        if len(candidates) < 3:
            return {}

        candidates_info = []
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                candidates_info.append(f"ID:{cid} {player.name}")

        prompt = f"""你是狐狸，每晚可以查验3个位置的玩家。如果查验范围内有狼人，你将失去技能；如果没有，你免疫狼刀。

=== 可查验玩家 ===
{chr(10).join(candidates_info)}

请分析局势，选择3个最合适的位置查验。

格式要求：回复中必须包含"我查验: ID1、ID2和ID3"，例如"我查验: 1、2和3"
"""

        messages = [
            ("system", self.get_system_prompt()),
            ("human", self.get_current_context()),
            ("human", prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        import re
        ids = []
        for x in re.findall(r'\d+', content):
            try:
                ids.append(int(x))
            except ValueError:
                pass

        if len(ids) >= 3:
            valid_ids = [cid for cid in ids if cid in candidates]
            if len(valid_ids) >= 3:
                return {"target1": valid_ids[0], "target2": valid_ids[1], "target3": valid_ids[2]}
            return {"target1": ids[0], "target2": ids[1], "target3": ids[2]}

        return {"target1": candidates[0], "target2": candidates[1], "target3": candidates[2]}


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

    def wolf_discuss(self, wolf_ids: list[int], non_wolf_targets: list[int],
                     other_wolf_suggestions: dict[int, int] = None) -> dict:
        """人类玩家狼人商讨"""
        print(f"\n[{self.player_name}] 狼人夜间商讨阶段")
        print("你是狼人！请与其他狼人商讨今晚刀谁。")

        # 显示狼人队友
        wolf_info = []
        for wid in wolf_ids:
            if wid != self.player_id:
                wolf_player = self.game_state.players.get(wid)
                if wolf_player:
                    wolf_info.append(f"ID:{wid} {wolf_player.name}")
        print(f"狼人队友: {', '.join(wolf_info) if wolf_info else '无'}")

        # 显示可刀目标
        print("\n可刀目标:")
        for tid in non_wolf_targets:
            player = self.game_state.players.get(tid)
            if player:
                print(f"  {tid}: {player.name}")

        while True:
            choice = self.input_func("选择你要提议刀的玩家ID: ")
            try:
                target_id = int(choice)
                if target_id in non_wolf_targets:
                    target_player = self.game_state.players.get(target_id)
                    print(f"你提议刀 {target_player.name}")
                    return {"player_id": self.player_id, "target_id": target_id, "reason": "human input"}
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def wolf_confirm_kill(self, target_id: int) -> dict:
        """人类玩家确认刀人"""
        target_player = self.game_state.players.get(target_id)
        target_name = target_player.name if target_player else f"ID:{target_id}"

        print(f"\n[{self.player_name}] 狼人确认刀人")
        print(f"狼人决定今晚刀 {target_name}，是否确认？(y/n): ")

        choice = self.input_func("> ").lower()
        if choice in ["y", "yes", "是", "确认"]:
            return {
                "player_id": self.player_id,
                "target_id": target_id,
                "confirmed": True,
                "message": f"确认刀 {target_name}"
            }
        else:
            return {
                "player_id": self.player_id,
                "target_id": target_id,
                "confirmed": False,
                "message": f"不同意刀 {target_name}"
            }

    def decide_hunter_shoot(self, candidates: list[int]) -> Optional[int]:
        """人类玩家猎人选择开枪目标"""
        print(f"\n[{self.player_name}] 你是猎人，你死亡了！")
        print("你可以选择开枪带走一名玩家。")

        print("\n可开枪目标:")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择开枪目标ID（跳过请输入-1）: ")
            try:
                target_id = int(choice)
                if target_id == -1:
                    print("你选择不开枪")
                    return None
                if target_id in candidates:
                    target_player = self.game_state.players.get(target_id)
                    print(f"你选择开枪带走 {target_player.name}")
                    return target_id
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def decide_cupid_link(self, candidates: list[int]) -> dict:
        """人类玩家丘比特选择情侣"""
        print(f"\n[{self.player_name}] 你是丘比特！")
        print("请选择两名玩家作为情侣。")

        print("\n可选玩家:")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择第一个情侣ID: ")
            try:
                lover1 = int(choice)
                if lover1 in candidates:
                    break
                print("无效选择")
            except ValueError:
                print("请输入数字")

        candidates2 = [c for c in candidates if c != lover1]
        print(f"\n可选玩家(排除第一个):")
        for cid in candidates2:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择第二个情侣ID: ")
            try:
                lover2 = int(choice)
                if lover2 in candidates2:
                    print(f"你选择 {self.game_state.players.get(lover1).name} 和 {self.game_state.players.get(lover2).name} 为情侣")
                    return {"lover1": lover1, "lover2": lover2}
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def decide_wild_child_idol(self, candidates: list[int]) -> dict:
        """人类玩家野孩子选择偶像"""
        print(f"\n[{self.player_name}] 你是野孩子！")
        print("请选择一名玩家作为偶像。")

        print("\n可选玩家:")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择偶像ID: ")
            try:
                idol = int(choice)
                if idol in candidates:
                    print(f"你选择 {self.game_state.players.get(idol).name} 为偶像")
                    return {"idol": idol}
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def decide_flutist_charm(self, candidates: list[int]) -> dict:
        """人类玩家吹笛人选择迷惑目标"""
        print(f"\n[{self.player_name}] 你是吹笛人！")
        print("请选择两名玩家进行迷惑。")

        print("\n可选玩家:")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择第一个迷惑目标ID: ")
            try:
                target1 = int(choice)
                if target1 in candidates:
                    break
                print("无效选择")
            except ValueError:
                print("请输入数字")

        candidates2 = [c for c in candidates if c != target1]
        print(f"\n可选玩家(排除第一个):")
        for cid in candidates2:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        while True:
            choice = self.input_func("选择第二个迷惑目标ID: ")
            try:
                target2 = int(choice)
                if target2 in candidates2:
                    print(f"你选择迷惑 {self.game_state.players.get(target1).name} 和 {self.game_state.players.get(target2).name}")
                    return {"target1": target1, "target2": target2}
                print("无效选择")
            except ValueError:
                print("请输入数字")

    def decide_fox_check(self, candidates: list[int]) -> dict:
        """人类玩家狐狸选择查验目标"""
        print(f"\n[{self.player_name}] 你是狐狸！")
        print("请选择3个位置进行查验。")

        print("\n可选玩家:")
        for cid in candidates:
            player = self.game_state.players.get(cid)
            if player:
                print(f"  {cid}: {player.name}")

        selected = []
        for i in range(3):
            remaining = [c for c in candidates if c not in selected]
            print(f"\n已选择: {selected}")
            print(f"可选玩家:")
            for cid in remaining:
                player = self.game_state.players.get(cid)
                if player:
                    print(f"  {cid}: {player.name}")

            while True:
                choice = self.input_func(f"选择第{i+1}个查验目标ID: ")
                try:
                    target = int(choice)
                    if target in remaining:
                        selected.append(target)
                        break
                    print("无效选择")
                except ValueError:
                    print("请输入数字")

        print(f"\n你选择查验: {[self.game_state.players.get(cid).name for cid in selected]}")
        return {"target1": selected[0], "target2": selected[1], "target3": selected[2]}
