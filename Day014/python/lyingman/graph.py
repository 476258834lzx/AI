"""LangGraph游戏流程图"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from .game_state import GameState, RoundPhase, NightPhase, PlayerStatus
from .referee import Referee
from .god import God
from .player_agent import PlayerAgent, HumanPlayerAgent


class GameGraphState(TypedDict):
    """游戏图状态"""
    game_state: GameState
    referee: Referee
    god: God
    players: dict[int, PlayerAgent]
    current_phase: str
    current_day: int
    winner: Optional[str]
    messages: list


class WerewolfGraph:
    """狼人杀游戏LangGraph"""

    def __init__(self, player_count: int = 8):
        self.player_count = player_count
        self.graph = None

    def build(self) -> StateGraph:
        """构建游戏图"""
        workflow = StateGraph(GameGraphState)

        # 注册节点
        workflow.add_node("init_game", self.node_init_game)
        workflow.add_node("referee_judge", self.node_referee_judge)
        workflow.add_node("god_start_day", self.node_god_start_day)
        workflow.add_node("sheriff_election", self.node_sheriff_election)
        workflow.add_node("day_speech", self.node_day_speech)
        workflow.add_node("day_vote", self.node_day_vote)
        workflow.add_node("god_start_night", self.node_god_start_night)
        workflow.add_node("wolf_kill", self.node_wolf_kill)
        workflow.add_node("seer_check", self.node_seer_check)
        workflow.add_node("witch_heal", self.node_witch_heal)
        workflow.add_node("witch_poison", self.node_witch_poison)
        workflow.add_node("guardian_protect", self.node_guardian_protect)
        workflow.add_node("hunter_choice", self.node_hunter_choice)
        workflow.add_node("night_settle", self.node_night_settle)
        workflow.add_node("game_end", self.node_game_end)

        # 设置入口
        workflow.set_entry_point("init_game")

        # 设置边
        workflow.add_edge("init_game", "referee_judge")

        workflow.add_conditional_edges(
            "referee_judge",
            self.should_continue,
            {
                "continue": "god_start_day",
                "end": "game_end",
            }
        )

        workflow.add_edge("god_start_day", "sheriff_election")
        workflow.add_edge("sheriff_election", "day_speech")
        workflow.add_edge("day_speech", "day_vote")
        workflow.add_edge("day_vote", "referee_judge")

        workflow.add_edge("day_vote", "god_start_night")
        workflow.add_edge("god_start_night", "wolf_kill")
        workflow.add_edge("wolf_kill", "seer_check")
        workflow.add_edge("seer_check", "witch_heal")
        workflow.add_edge("witch_heal", "witch_poison")
        workflow.add_edge("witch_poison", "guardian_protect")
        workflow.add_edge("guardian_protect", "hunter_choice")
        workflow.add_edge("hunter_choice", "night_settle")
        workflow.add_edge("night_settle", "referee_judge")

        workflow.add_edge("game_end", END)

        self.graph = workflow.compile()
        return self.graph

    def should_continue(self, state: GameGraphState) -> str:
        """判断是否继续游戏"""
        winner = state["referee"].check_win_condition()
        if winner:
            state["winner"] = winner
            return "end"
        return "continue"

    def node_init_game(self, state: GameGraphState) -> GameGraphState:
        """初始化游戏"""
        player_names = [f"玩家{i+1}" for i in range(self.player_count)]
        game_state = state["referee"].init_game(player_names)
        state["game_state"] = game_state
        state["god"] = God(game_state)

        state["players"] = {
            i: PlayerAgent(player_id=i, player_name=player_names[i])
            for i in range(self.player_count)
        }

        for agent in state["players"].values():
            agent.set_game_state(game_state)

        state["current_day"] = 1
        state["current_phase"] = "init"
        state["messages"] = ["游戏初始化完成"]

        return state

    def node_referee_judge(self, state: GameGraphState) -> GameGraphState:
        """裁判判定"""
        winner = state["referee"].check_win_condition()
        if winner:
            state["winner"] = winner
            state["current_phase"] = "game_end"
        else:
            state["current_phase"] = "judge"
        return state

    def node_god_start_day(self, state: GameGraphState) -> GameGraphState:
        """上帝开启白天"""
        god = state["god"]
        game_state = state["game_state"]

        if state["current_day"] == 1:
            god.start_new_round(RoundPhase.SHERIFF_ELECTION)
        else:
            god.start_day()
            god.start_new_round(RoundPhase.DAY_SPEECH)

        state["current_phase"] = "day_start"
        state["messages"].append(f"第{game_state.current_day}天白天开始")

        return state

    def node_sheriff_election(self, state: GameGraphState) -> GameGraphState:
        """警长竞选（仅第一天）"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        if game_state.current_day != 1:
            state["current_phase"] = "sheriff_end"
            return state

        god.start_new_round(RoundPhase.SHERIFF_ELECTION)

        candidates = []
        for pid, agent in players.items():
            player = game_state.players.get(pid)
            if player and player.is_alive():
                if agent.decide_sheriff_run():
                    candidates.append(pid)
                    game_state.sheriff_candidate_ids.append(pid)

        state["messages"].append(f"警长竞选: {len(candidates)}人上警")

        if candidates:
            for pid in candidates:
                agent = players[pid]
                result = agent.speak("请发表竞选发言")
                god.record_day_action(pid, result["speech"])

            votes = {pid: 0 for pid in candidates}
            for pid, agent in players.items():
                player = game_state.players.get(pid)
                if player and player.is_alive():
                    vote_target = agent.vote(candidates)
                    if vote_target in votes:
                        votes[vote_target] += 1

            max_votes = max(votes.values())
            winners = [pid for pid, v in votes.items() if v == max_votes]

            if len(winners) == 1:
                game_state.sheriff_id = winners[0]
                player = game_state.players[winners[0]]
                state["messages"].append(f"{player.name}当选警长")
            else:
                state["messages"].append("警长竞选平票")

        state["current_phase"] = "sheriff_end"
        return state

    def node_day_speech(self, state: GameGraphState) -> GameGraphState:
        """白天发言"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        god.start_new_round(RoundPhase.DAY_SPEECH)

        speech_order = god.get_speech_order()

        for pid in speech_order:
            agent = players.get(pid)
            if not agent:
                continue

            player = game_state.players.get(pid)
            if not player or not player.is_alive():
                continue

            prompt = f"你是第{len(speech_order)}个发言"
            result = agent.speak(prompt)
            god.record_day_action(pid, result["speech"])

        state["current_phase"] = "speech_end"
        return state

    def node_day_vote(self, state: GameGraphState) -> GameGraphState:
        """白天投票"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        god.start_new_round(RoundPhase.DAY_VOTE)

        alive = game_state.get_alive_players()
        candidates = [p.id for p in alive]

        for p in game_state.players.values():
            p.vote_count = 0

        for pid in candidates:
            agent = players.get(pid)
            if not agent:
                continue

            player = game_state.players.get(pid)
            if not player or not player.can_vote():
                continue

            vote_target = agent.vote(candidates)
            if vote_target in candidates:
                vote_count = 1.5 if player.is_sheriff else 1
                game_state.players[vote_target].vote_count += vote_count

        max_votes = 0
        candidates_to_eliminate = []

        for pid in candidates:
            player = game_state.players.get(pid)
            if player and player.vote_count > max_votes:
                max_votes = player.vote_count
                candidates_to_eliminate = [pid]
            elif player and player.vote_count == max_votes:
                candidates_to_eliminate.append(pid)

        if len(candidates_to_eliminate) == 1:
            eliminated = candidates_to_eliminate[0]
            player = game_state.players[eliminated]
            player.status = PlayerStatus.DEAD_DAY
            state["messages"].append(f"{player.name}被投票出局")
            god.current_round.deaths.append(eliminated)
        else:
            state["messages"].append("投票平票，无人出局")

        god.add_to_history()
        state["current_phase"] = "vote_end"
        return state

    def node_god_start_night(self, state: GameGraphState) -> GameGraphState:
        """上帝开启夜晚"""
        god = state["god"]
        god.start_night()

        state["current_phase"] = "night"
        state["messages"].append("夜晚降临")

        return state

    def _get_role_players(self, state: GameGraphState, role_name: str) -> list[int]:
        """获取特定角色的存活玩家ID"""
        game_state = state["game_state"]
        role_players = []

        for pid, player in game_state.players.items():
            if player.is_alive() and player.role and player.role.name == role_name:
                role_players.append(pid)

        return role_players

    def _get_wolf_players(self, state: GameGraphState) -> list[int]:
        """获取狼人玩家ID"""
        wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
        game_state = state["game_state"]
        wolves = []

        for pid, player in game_state.players.items():
            if player.is_alive() and player.role and player.role.name in wolf_roles:
                wolves.append(pid)

        return wolves

    def node_wolf_kill(self, state: GameGraphState) -> GameGraphState:
        """狼人刀人"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        wolves = self._get_wolf_players(state)

        if not wolves:
            state["messages"].append("无存活狼人")
            return state

        targets = []

        for pid in wolves:
            agent = players.get(pid)
            if not agent:
                continue

            target = agent.decide_night_action("wolf_kill")
            if target:
                targets.append(target)

        if targets:
            target_counts = {}
            for t in targets:
                target_counts[t] = target_counts.get(t, 0) + 1

            max_count = max(target_counts.values())
            top_targets = [t for t, c in target_counts.items() if c == max_count]

            import random
            game_state.wolf_kill_target = random.choice(top_targets)
            target_name = game_state.players[game_state.wolf_kill_target].name
            state["messages"].append(f"狼人刀了{target_name}")

        god.record_action(
            NightPhase.WOLF_KILL,
            actor_id=wolves[0] if wolves else -1,
            target_id=game_state.wolf_kill_target,
            result=f"刀了{game_state.wolf_kill_target}",
            success=True,
        )

        state["current_phase"] = "wolf_kill_end"
        return state

    def node_seer_check(self, state: GameGraphState) -> GameGraphState:
        """预言家查验"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        seers = self._get_role_players(state, "预言家")

        if not seers:
            state["messages"].append("无存活预言家")
            return state

        seer_id = seers[0]
        agent = players.get(seer_id)

        if not agent:
            return state

        target = agent.decide_night_action("seer_check")

        if target:
            target_player = game_state.players.get(target)
            if target_player and target_player.role:
                if target_player.role.name == "恶灵骑士":
                    result = "好人"
                elif target_player.role.name.endswith("狼"):
                    result = "狼人"
                else:
                    result = "好人"
            else:
                result = "好人"

            game_state.seer_check_result = {
                "target_id": target,
                "result": result,
            }

            god.record_action(
                NightPhase.SEER_CHECK,
                actor_id=seer_id,
                target_id=target,
                result=result,
                success=True,
            )

        state["current_phase"] = "seer_check_end"
        return state

    def node_witch_heal(self, state: GameGraphState) -> GameGraphState:
        """女巫救人"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        witches = self._get_role_players(state, "女巫")

        if not witches:
            state["messages"].append("无存活女巫")
            return state

        witch_id = witches[0]
        agent = players.get(witch_id)

        if not agent:
            return state

        if game_state.wolf_kill_target is None:
            return state

        target = agent.decide_night_action("witch_heal")

        if target and not game_state.witch_heal_used:
            game_state.witch_heal_used = True
            game_state.revives.append(target)

            target_name = game_state.players[target].name
            state["messages"].append(f"女巫救了{target_name}")

            god.record_action(
                NightPhase.WITCH_HEAL,
                actor_id=witch_id,
                target_id=target,
                result="救人",
                success=True,
            )

        state["current_phase"] = "witch_heal_end"
        return state

    def node_witch_poison(self, state: GameGraphState) -> GameGraphState:
        """女巫毒人"""
        state["current_phase"] = "witch_poison_end"
        return state

    def node_guardian_protect(self, state: GameGraphState) -> GameGraphState:
        """守卫守护"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        guardians = self._get_role_players(state, "守卫")

        if not guardians:
            state["messages"].append("无存活守卫")
            return state

        guardian_id = guardians[0]
        agent = players.get(guardian_id)

        if not agent:
            return state

        target = agent.decide_night_action("guardian_protect")

        if target:
            if game_state.guardian_last_protect != target:
                game_state.guardian_last_protect = target
                game_state.players[target].is_protected = True

                target_name = game_state.players[target].name
                state["messages"].append(f"守卫守护了{target_name}")

                god.record_action(
                    NightPhase.GUARDIAN_PROTECT,
                    actor_id=guardian_id,
                    target_id=target,
                    result="守护",
                    success=True,
                )

        state["current_phase"] = "guardian_protect_end"
        return state

    def node_hunter_choice(self, state: GameGraphState) -> GameGraphState:
        """猎人选择"""
        state["messages"].append("猎人待命")
        state["current_phase"] = "hunter_choice_end"
        return state

    def node_night_settle(self, state: GameGraphState) -> GameGraphState:
        """夜晚结算"""
        god = state["god"]
        game_state = state["game_state"]

        deaths = god.night_settle()

        for pid in deaths:
            player = game_state.players.get(pid)
            if player:
                state["messages"].append(f"{player.name}夜间死亡")

                if player.role and player.role.name == "猎人" and not player.is_poisoned:
                    import random
                    alive = game_state.get_alive_players()
                    if alive:
                        target = random.choice(alive)
                        target.status = PlayerStatus.DEAD_NIGHT
                        state["messages"].append(f"猎人带走了{target.name}")

        god.add_to_history()
        state["current_phase"] = "night_end"
        return state

    def node_game_end(self, state: GameGraphState) -> GameGraphState:
        """游戏结束"""
        winner = state.get("winner")

        if winner == "good":
            state["messages"].append("===== 游戏结束：好人胜利 =====")
        elif winner == "wolf":
            state["messages"].append("===== 游戏结束：狼人胜利 =====")
        elif winner == "neutral":
            state["messages"].append("===== 游戏结束：第三方胜利 =====")
        else:
            state["messages"].append("===== 游戏结束 =====")

        state["current_phase"] = "game_end"
        return state

    def run(self) -> dict:
        """运行游戏"""
        if not self.graph:
            self.build()

        initial_state = GameGraphState(
            game_state=None,
            referee=Referee(self.player_count),
            god=None,
            players={},
            current_phase="init",
            current_day=1,
            winner=None,
            messages=[],
        )

        result = self.graph.invoke(initial_state)
        return result
