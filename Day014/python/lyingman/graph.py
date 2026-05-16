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
        workflow.add_node("white_wolf_explode", self.node_white_wolf_explode)
        workflow.add_node("day_vote", self.node_day_vote)
        workflow.add_node("god_start_night", self.node_god_start_night)
        workflow.add_node("wolf_kill", self.node_wolf_kill)
        workflow.add_node("wolf_beauty_charm", self.node_wolf_beauty_charm)
        workflow.add_node("cupid_link", self.node_cupid_link)
        workflow.add_node("wild_child_idol", self.node_wild_child_idol)
        workflow.add_node("seer_check", self.node_seer_check)
        workflow.add_node("fox_check", self.node_fox_check)
        workflow.add_node("witch_heal", self.node_witch_heal)
        workflow.add_node("witch_poison", self.node_witch_poison)
        workflow.add_node("guardian_protect", self.node_guardian_protect)
        workflow.add_node("flutist_charm", self.node_flutist_charm)
        workflow.add_node("hunter_choice", self.node_hunter_choice)
        workflow.add_node("night_settle", self.node_night_settle)
        workflow.add_node("bear_roar_check", self.node_bear_roar_check)
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
                "night": "god_start_night",  # 新增：第一晚进入夜晚
            }
        )

        # 第一晚特殊流程（仅第一天）
        workflow.add_conditional_edges(
            "wolf_kill",
            self.check_first_night,
            {
                "cupid": "cupid_link",
                "wild_child": "wild_child_idol",
                "continue": "seer_check",
            }
        )

        workflow.add_edge("god_start_day", "sheriff_election")
        workflow.add_edge("sheriff_election", "day_speech")
        workflow.add_edge("day_speech", "white_wolf_explode")
        workflow.add_edge("white_wolf_explode", "day_vote")
        workflow.add_edge("day_vote", "referee_judge")

        workflow.add_edge("cupid_link", "wild_child_idol")
        workflow.add_edge("wild_child_idol", "seer_check")
        workflow.add_edge("seer_check", "fox_check")
        workflow.add_edge("fox_check", "witch_heal")
        workflow.add_edge("witch_heal", "witch_poison")
        workflow.add_edge("witch_poison", "guardian_protect")
        workflow.add_edge("guardian_protect", "flutist_charm")
        workflow.add_edge("flutist_charm", "hunter_choice")
        workflow.add_edge("hunter_choice", "night_settle")
        workflow.add_edge("night_settle", "bear_roar_check")
        workflow.add_edge("bear_roar_check", "referee_judge")

        # 白天投票后进入夜晚
        workflow.add_edge("day_vote", "god_start_night")

        workflow.add_edge("game_end", END)

        self.graph = workflow.compile()
        return self.graph

    def should_continue(self, state: GameGraphState) -> str:
        """判断是否继续游戏"""
        # 初始化后进入第一晚
        if state["current_phase"] == "init":
            return "night"

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
                god.record_day_action(pid, result["speech"], players)

            votes = {pid: 0 for pid in candidates}
            for pid, agent in players.items():
                player = game_state.players.get(pid)
                if player and player.is_alive():
                    # 上警玩家不能参与投票
                    if pid in candidates:
                        continue
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
            god.record_day_action(pid, result["speech"], players)

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

    def _get_neutral_role_player(self, state: GameGraphState, role_name: str) -> Optional[int]:
        """获取特定中立角色的存活玩家ID"""
        game_state = state["game_state"]
        for pid, player in game_state.players.items():
            if player.is_alive() and player.role and player.role.name == role_name:
                return pid
        return None

    def check_first_night(self, state: GameGraphState) -> str:
        """检查是否第一晚，返回下一节点"""
        if state["current_day"] == 1:
            # 第一晚需要先处理丘比特和野孩子
            cupid_id = self._get_neutral_role_player(state, "丘比特")
            if cupid_id is not None:
                return "cupid"
            wild_child_id = self._get_neutral_role_player(state, "野孩子")
            if wild_child_id is not None:
                return "wild_child"
        return "continue"

    def node_white_wolf_explode(self, state: GameGraphState) -> GameGraphState:
        """白狼王白天自爆"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        # 查找白狼王
        white_wolf_ids = self._get_role_players(state, "白狼王")

        if not white_wolf_ids:
            state["messages"].append("无存活白狼王")
            state["current_phase"] = "white_wolf_end"
            return state

        white_wolf_id = white_wolf_ids[0]
        agent = players.get(white_wolf_id)

        if not agent:
            state["current_phase"] = "white_wolf_end"
            return state

        # 白狼王决定是否自爆
        # 这里简化处理：白狼王有50%概率自爆
        import random
        if random.random() < 0.5:
            # 白狼王自爆，选择带走一人
            alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive() if pid != white_wolf_id]
            if alive_ids:
                target_id = random.choice(alive_ids)
                target = game_state.players.get(target_id)
                target_name = target.name if target else f"ID:{target_id}"

                # 白狼王死亡
                white_wolf = game_state.players.get(white_wolf_id)
                white_wolf.status = PlayerStatus.DEAD_DAY
                god.current_round.deaths.append(white_wolf_id)

                # 带走目标
                if target:
                    target.status = PlayerStatus.DEAD_DAY
                    god.current_round.deaths.append(target_id)

                state["messages"].append(f"白狼王自爆，带走了{target_name}！")

                # 白狼王自爆后直接进入夜晚
                game_state.current_phase = RoundPhase.NIGHT_START

        state["current_phase"] = "white_wolf_end"
        return state

    def node_wolf_beauty_charm(self, state: GameGraphState) -> GameGraphState:
        """狼美人魅惑"""
        game_state = state["game_state"]
        players = state["players"]

        # 查找狼美人
        wolf_beauty_ids = self._get_role_players(state, "狼美人")

        if not wolf_beauty_ids:
            state["messages"].append("无存活狼美人")
            state["current_phase"] = "wolf_beauty_end"
            return state

        wolf_beauty_id = wolf_beauty_ids[0]
        agent = players.get(wolf_beauty_id)

        if not agent:
            state["current_phase"] = "wolf_beauty_end"
            return state

        # 狼美人选择魅惑目标
        alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive() if pid != wolf_beauty_id]

        if alive_ids:
            import random
            charm_target = random.choice(alive_ids)
            game_state.wolf_beauty_charm_target = charm_target

            target = game_state.players.get(charm_target)
            target_name = target.name if target else f"ID:{charm_target}"
            state["messages"].append(f"狼美人魅惑了{target_name}")

        state["current_phase"] = "wolf_beauty_end"
        return state

    def node_cupid_link(self, state: GameGraphState) -> GameGraphState:
        """丘比特连线"""
        game_state = state["game_state"]
        players = state["players"]

        cupid_id = self._get_neutral_role_player(state, "丘比特")

        if cupid_id is None:
            state["messages"].append("无存活丘比特")
            state["current_phase"] = "cupid_end"
            return state

        # 检查是否已经连线
        if game_state.lovers:
            state["messages"].append("丘比特已连线")
            state["current_phase"] = "cupid_end"
            return state

        agent = players.get(cupid_id)
        if not agent:
            state["current_phase"] = "cupid_end"
            return state

        # 丘比特选择情侣
        alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive()]

        result = agent.decide_cupid_link(alive_ids)
        lover1 = result.get("lover1")
        lover2 = result.get("lover2")

        if lover1 and lover2 and lover1 in alive_ids and lover2 in alive_ids:
            game_state.set_lovers(lover1, lover2)
            game_state.love_chain_type = game_state.get_love_type(lover1)

            lover1_name = game_state.players[lover1].name
            lover2_name = game_state.players[lover2].name
            state["messages"].append(f"丘比特指定{lover1_name}和{lover2_name}为情侣（{game_state.love_chain_type}）")

        state["current_phase"] = "cupid_end"
        return state

    def node_wild_child_idol(self, state: GameGraphState) -> GameGraphState:
        """野孩子选择偶像"""
        game_state = state["game_state"]
        players = state["players"]

        wild_child_id = self._get_neutral_role_player(state, "野孩子")

        if wild_child_id is None:
            state["messages"].append("无存活野孩子")
            state["current_phase"] = "wild_child_end"
            return state

        # 检查是否已经选择偶像
        if game_state.wild_child_idol is not None:
            state["messages"].append("野孩子已选择偶像")
            state["current_phase"] = "wild_child_end"
            return state

        agent = players.get(wild_child_id)
        if not agent:
            state["current_phase"] = "wild_child_end"
            return state

        # 野孩子选择偶像
        alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive() if pid != wild_child_id]

        result = agent.decide_wild_child_idol(alive_ids)
        idol_id = result.get("idol")

        if idol_id and idol_id in alive_ids:
            game_state.set_wild_child_idol(idol_id)
            idol_name = game_state.players[idol_id].name
            state["messages"].append(f"野孩子选择了{idol_name}作为偶像")

        state["current_phase"] = "wild_child_end"
        return state

    def node_fox_check(self, state: GameGraphState) -> GameGraphState:
        """狐狸查验"""
        game_state = state["game_state"]
        players = state["players"]

        fox_id = self._get_neutral_role_player(state, "狐狸")

        if fox_id is None:
            state["messages"].append("无存活狐狸")
            state["current_phase"] = "fox_check_end"
            return state

        # 狐狸失去技能后不能再查验
        if game_state.fox_lost_skill:
            state["messages"].append("狐狸已失去技能")
            state["current_phase"] = "fox_check_end"
            return state

        agent = players.get(fox_id)
        if not agent:
            state["current_phase"] = "fox_check_end"
            return state

        # 狐狸查验3个相邻位置
        alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive()]

        result = agent.decide_fox_check(alive_ids)
        target1 = result.get("target1")
        target2 = result.get("target2")
        target3 = result.get("target3")

        # 检查是否有狼人
        has_wolf = False
        targets = [t for t in [target1, target2, target3] if t is not None]

        for tid in targets:
            target = game_state.players.get(tid)
            if target and target.role:
                if "狼" in target.role.name:
                    has_wolf = True
                    break

        if has_wolf:
            game_state.fox_lost_skill = True
            game_state.fox_last_check_has_wolf = True
            state["messages"].append("狐狸查验范围内有狼人，技能失效")
        else:
            game_state.fox_lost_skill = False
            game_state.fox_last_check_has_wolf = False
            state["messages"].append("狐狸查验范围内无狼人，技能生效")

        state["current_phase"] = "fox_check_end"
        return state

    def node_flutist_charm(self, state: GameGraphState) -> GameGraphState:
        """吹笛人迷惑"""
        game_state = state["game_state"]
        players = state["players"]

        flutist_id = self._get_neutral_role_player(state, "吹笛人")

        if flutist_id is None:
            state["messages"].append("无存活吹笛人")
            state["current_phase"] = "flutist_charm_end"
            return state

        agent = players.get(flutist_id)
        if not agent:
            state["current_phase"] = "flutist_charm_end"
            return state

        # 吹笛人选择迷惑目标
        alive_ids = [pid for pid, p in game_state.players.items() if p.is_alive() if pid != flutist_id]

        result = agent.decide_flutist_charm(alive_ids)
        target1 = result.get("target1")
        target2 = result.get("target2")

        if target1:
            game_state.flutist_charmed_ids.append(target1)
        if target2:
            game_state.flutist_charmed_ids.append(target2)

        charmed_names = []
        for tid in [t for t in [target1, target2] if t]:
            if tid in game_state.players:
                charmed_names.append(game_state.players[tid].name)

        if charmed_names:
            state["messages"].append(f"吹笛人迷惑了{'和'.join(charmed_names)}")

        state["current_phase"] = "flutist_charm_end"
        return state

    def node_bear_roar_check(self, state: GameGraphState) -> GameGraphState:
        """熊咆哮检查"""
        game_state = state["game_state"]

        bear_id = self._get_neutral_role_player(state, "熊")

        if bear_id is None:
            state["messages"].append("无存活熊")
            state["current_phase"] = "bear_roar_end"
            return state

        # 获取熊相邻位置的玩家
        player_ids = list(game_state.players.keys())
        player_ids.sort()

        bear_index = player_ids.index(bear_id)
        n = len(player_ids)

        # 相邻位置（左右各一个）
        left_id = player_ids[(bear_index - 1) % n]
        right_id = player_ids[(bear_index + 1) % n]

        # 检查相邻位置是否有狼人存活
        has_wolf_nearby = False
        for adjacent_id in [left_id, right_id]:
            adjacent = game_state.players.get(adjacent_id)
            if adjacent and adjacent.is_alive() and adjacent.role:
                if "狼" in adjacent.role.name:
                    has_wolf_nearby = True
                    break

        if has_wolf_nearby:
            game_state.bear_roared_today = True
            state["messages"].append("熊咆哮了！")
        else:
            game_state.bear_roared_today = False
            state["messages"].append("今晚平安无事")

        state["current_phase"] = "bear_roar_end"
        return state

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
                    # 恶灵骑士：查验显示狼人，但次日反伤预言家
                    result = "狼人"
                elif target_player.role.name.endswith("狼"):
                    result = "狼人"
                else:
                    result = "好人"
            else:
                result = "好人"

            game_state.seer_check_result = {
                "target_id": target,
                "result": result,
                "seer_id": seer_id,
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
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        witches = self._get_role_players(state, "女巫")

        if not witches:
            state["messages"].append("无存活女巫")
            state["current_phase"] = "witch_poison_end"
            return state

        witch_id = witches[0]
        agent = players.get(witch_id)

        if not agent:
            state["current_phase"] = "witch_poison_end"
            return state

        # 检查毒药是否已用
        if game_state.witch_poison_used:
            state["messages"].append("女巫毒药已用")
            state["current_phase"] = "witch_poison_end"
            return state

        # 女巫决定是否毒人
        target = agent.decide_night_action("witch_poison")

        if target:
            game_state.witch_poison_used = True
            target_player = game_state.players.get(target)
            if target_player:
                target_player.is_poisoned = True
                target_name = target_player.name
                state["messages"].append(f"女巫毒了{target_name}")

                god.record_action(
                    NightPhase.WITCH_POISON,
                    actor_id=witch_id,
                    target_id=target,
                    result="毒杀",
                    success=True,
                )

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
        """猎人选择标记节点

        注意：猎人实际开枪逻辑在 node_night_settle 中通过 hunter_callback 处理
        这里只标记阶段，因为猎人死亡时需要在夜晚结算时立即决定是否开枪
        """
        state["messages"].append("猎人待命")
        state["current_phase"] = "hunter_choice_end"
        return state

    def node_night_settle(self, state: GameGraphState) -> GameGraphState:
        """夜晚结算"""
        god = state["god"]
        game_state = state["game_state"]
        players = state["players"]

        def hunter_shoot_callback(hunter_id: int, candidates: list) -> Optional[int]:
            """猎人开枪回调"""
            agent = players.get(hunter_id)
            if agent:
                return agent.decide_hunter_shoot(candidates)
            return None

        deaths = god.night_settle(hunter_callback=hunter_shoot_callback)

        for pid in deaths:
            player = game_state.players.get(pid)
            if player:
                state["messages"].append(f"{player.name}夜间死亡")

                if player.role and player.role.name == "猎人" and not player.is_poisoned:
                    # 猎人已经在night_settle中处理开枪，这里只记录日志
                    pass

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
