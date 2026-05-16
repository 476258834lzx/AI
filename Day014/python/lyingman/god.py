"""上帝 - 负责回合管理和信息记录"""
from typing import Optional, TYPE_CHECKING
from .game_state import GameState, Round, RoundPhase, NightPhase, NightAction, PlayerStatus
if TYPE_CHECKING:
    from .player_agent import PlayerAgent


class God:
    """上帝类"""

    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.current_round: Optional[Round] = None

    def start_new_round(self, phase: RoundPhase) -> Round:
        """开启新回合"""
        self.current_round = Round(
            day=self.game_state.current_day,
            phase=phase,
            actions=[],
            deaths=[],
            revives=[],
        )
        self.game_state.current_phase = phase
        return self.current_round

    def start_day(self) -> RoundPhase:
        """开启白天"""
        self.game_state.current_day += 1
        return self.start_new_round(RoundPhase.DAY_START)

    def start_night(self) -> RoundPhase:
        """开启夜晚"""
        return self.start_new_round(RoundPhase.NIGHT_START)

    def record_action(
        self,
        phase: NightPhase,
        actor_id: int,
        target_id: Optional[int] = None,
        result: Optional[str] = None,
        success: bool = False,
    ):
        """记录夜间动作"""
        if self.current_round:
            action = NightAction(
                phase=phase,
                actor_id=actor_id,
                target_id=target_id,
                result=result,
                success=success,
            )
            self.current_round.actions.append(action)
            self.game_state.night_actions.append(action)

    def record_day_action(self, speaker_id: int, content: str, players: dict = None):
        """记录白天发言并广播给所有玩家"""
        if self.current_round:
            from .game_state import DayAction
            action = DayAction(
                phase=self.current_round.phase,
                speaker_id=speaker_id,
                content=content,
            )
            self.current_round.actions.append(action)
            self.game_state.day_actions.append(action)

        # 广播发言给所有玩家的对话历史
        if players:
            speech_record = {"player_id": speaker_id, "speech": content}
            for pid, agent in players.items():
                if pid != speaker_id:
                    agent.conversation_history.append(speech_record)

    def get_night_action_result(self, phase: NightPhase) -> Optional[dict]:
        """获取某夜间阶段的动作结果"""
        for action in self.game_state.night_actions:
            if action.phase == phase:
                return {
                    "actor_id": action.actor_id,
                    "target_id": action.target_id,
                    "result": action.result,
                }
        return None

    def night_settle(self, hunter_callback=None):
        """夜晚结算

        Args:
            hunter_callback: 猎人死亡时的回调函数(hunter_id, candidates) -> target_id or None
                             用于让猎人的agent立即选择开枪目标
        """
        deaths = []

        # ===== 1. 处理预言家查验恶灵骑士反伤 =====
        seer_result = self.game_state.seer_check_result
        if seer_result:
            target_id = seer_result.get("target_id")
            target = self.game_state.players.get(target_id) if target_id else None
            if target and target.role and target.role.name == "恶灵骑士":
                # 恶灵骑士被查验，次日反伤预言家
                seer_id = seer_result.get("seer_id")
                if seer_id:
                    self.game_state.evil_knight_checked_by = seer_id
                    print(f"【{target.name}】是恶灵骑士，预兆家查验后触发反伤！")

        # ===== 2. 处理女巫毒杀恶灵骑士反伤 =====
        for player in self.game_state.players.values():
            if player.is_poisoned and player.role and player.role.name == "恶灵骑士":
                # 恶灵骑士被毒，女巫死亡
                witch_id = self._find_witch_id()
                if witch_id:
                    self.game_state.evil_knight_poisoned_by = witch_id
                    print(f"【{player.name}】是恶灵骑士，女巫毒杀触发反伤！")
                    if witch_id not in deaths:
                        deaths.append(witch_id)

        # ===== 3. 处理狼刀 =====
        wolf_target = self.game_state.wolf_kill_target
        if wolf_target is not None:
            target = self.game_state.players.get(wolf_target)
            if target and target.is_alive():
                # 恶灵骑士免疫狼刀
                if target.role and target.role.name == "恶灵骑士":
                    print(f"【{target.name}】是恶灵骑士，免疫狼刀！")
                elif not target.is_protected:
                    deaths.append(wolf_target)

        # ===== 4. 处理女巫救人 =====
        if wolf_target in self.game_state.revives:
            if wolf_target in deaths:
                deaths.remove(wolf_target)

        # ===== 5. 处理女巫毒人 =====
        for player in self.game_state.players.values():
            if player.is_poisoned:
                # 恶灵骑士被毒已经在上一步处理
                if player.role and player.role.name == "恶灵骑士":
                    continue
                if player.id not in deaths:
                    deaths.append(player.id)

        # ===== 6. 处理恶灵骑士反伤死亡 =====
        # 如果预言家查验了恶灵骑士，预言家次日死亡
        if self.game_state.evil_knight_checked_by:
            prophet_id = self.game_state.evil_knight_checked_by
            prophet = self.game_state.players.get(prophet_id)
            if prophet and prophet.is_alive() and prophet_id not in deaths:
                deaths.append(prophet_id)
                print(f"【{prophet.name}】被恶灵骑士反伤死亡！")

        # ===== 7. 处理需要立即开枪的猎人 =====
        for player_id in deaths[:]:  # 使用切片避免修改列表时出问题
            player = self.game_state.players.get(player_id)
            if player and player.role and player.role.can_shoot_on_death:
                # 猎人被女巫毒死不能开枪
                if player.is_poisoned:
                    print(f"【{player.name}】被女巫毒死，无法开枪")
                else:
                    # 立即调用LLM让猎人选择开枪目标
                    print(f"【{player.name}】死亡，正在决定是否开枪...")
                    if hunter_callback:
                        # 获取可开枪目标（排除自己和已死亡玩家）
                        candidates = [pid for pid, p in self.game_state.players.items()
                                    if p.is_alive() and pid != player_id]
                        target_id = hunter_callback(player_id, candidates)
                        if target_id is not None:
                            if target_id not in deaths:
                                deaths.append(target_id)
                                target = self.game_state.players.get(target_id)
                                if target:
                                    print(f"【{player.name}】开枪带走了【{target.name}】")
                        else:
                            print(f"【{player.name}】选择不开枪")

        # ===== 8. 执行死亡 =====
        for player_id in deaths:
            player = self.game_state.players.get(player_id)
            if player:
                player.status = PlayerStatus.DEAD_NIGHT
                self.current_round.deaths.append(player_id)

        # ===== 9. 处理情侣殉情 =====
        self._handle_couple_death(deaths)

        # ===== 10. 重置夜间状态 =====
        self.game_state.wolf_kill_target = None
        self.game_state.seer_check_result = None
        # 注意：witch_heal_used 和 witch_poison_used 是永久消耗标志，不能重置
        self.game_state.evil_knight_checked_by = None
        self.game_state.evil_knight_poisoned_by = None
        # 清理狼人商讨状态
        self.game_state.wolf_discuss_proposals.clear()
        self.game_state.wolf_awaiting_confirm.clear()
        self.game_state.wolf_consensus_target = None
        for player in self.game_state.players.values():
            player.is_protected = False
            player.is_poisoned = False
            player.vote_count = 0

        return deaths

    def _find_witch_id(self) -> Optional[int]:
        """查找女巫ID"""
        for pid, player in self.game_state.players.items():
            if player.is_alive() and player.role and player.role.name == "女巫":
                return pid
        return None

    def get_round_summary(self) -> str:
        """获取回合摘要"""
        if not self.current_round:
            return "无回合信息"

        summary_parts = [f"第{self.game_state.current_day}天 {self.current_round.phase.value}"]

        if self.current_round.deaths:
            death_names = [
                self.game_state.players[pid].name
                for pid in self.current_round.deaths
                if pid in self.game_state.players
            ]
            summary_parts.append(f"死亡: {', '.join(death_names)}")

        if self.current_round.revives:
            revive_names = [
                self.game_state.players[pid].name
                for pid in self.current_round.revives
                if pid in self.game_state.players
            ]
            summary_parts.append(f"救人: {', '.join(revive_names)}")

        return "\n".join(summary_parts)

    def get_full_history(self) -> list[dict]:
        """获取完整历史"""
        return [
            {
                "day": r.day,
                "phase": r.phase.value,
                "deaths": r.deaths,
                "revives": r.revives,
            }
            for r in self.game_state.round_history
        ]

    def add_to_history(self):
        """将当前回合添加到历史"""
        if self.current_round:
            self.game_state.round_history.append(self.current_round)
            self.current_round = None

    def get_speech_order(self) -> list[int]:
        """获取发言顺序（警长优先，其他随机）"""
        alive_players = self.game_state.get_alive_players()

        if not alive_players:
            return []

        sheriff = self.game_state.sheriff_id
        if sheriff and any(p.id == sheriff for p in alive_players):
            order = [sheriff]
            order.extend(p.id for p in alive_players if p.id != sheriff)
        else:
            order = [p.id for p in alive_players]

        return order

    def _handle_couple_death(self, deaths: list[int]):
        """处理情侣殉情（只有存在情侣时才执行）"""
        if not self.game_state.lovers:
            return

        new_deaths = []
        for dead_id in deaths:
            lover_id = self.game_state.get_lover(dead_id)
            if lover_id is not None:
                lover = self.game_state.players.get(lover_id)
                if lover and lover.is_alive() and lover_id not in deaths and lover_id not in new_deaths:
                    new_deaths.append(lover_id)
                    lover.status = PlayerStatus.DEAD_NIGHT
                    if self.current_round:
                        self.current_round.deaths.append(lover_id)
                    print(f"【{lover.name}】因情侣殉情死亡")

        deaths.extend(new_deaths)

    def get_night_action_order(self) -> list[tuple[NightPhase, Optional[str]]]:
        """获取夜间动作顺序和对应角色"""
        return [
            (NightPhase.WOLF_DISCUSS, "狼人"),      # 狼人先商讨
            (NightPhase.WOLF_KILL, "狼人"),          # 达成共识后刀人
            (NightPhase.SEER_CHECK, "预言家"),
            (NightPhase.WITCH_HEAL, "女巫"),
            (NightPhase.WITCH_POISON, "女巫"),
            (NightPhase.GUARDIAN_PROTECT, "守卫"),
            (NightPhase.HUNTER_CHOICE, "猎人"),
        ]

    def notify_player(self, player_id: int, message: str):
        """通知玩家信息（由外部消息系统实现）"""
        # 实际实现需要消息队列或事件系统
        pass

    def get_player_info(self, player_id: int, hide_role: bool = True) -> dict:
        """获取玩家可见信息"""
        player = self.game_state.players.get(player_id)
        if not player:
            return {}

        info = player.to_dict(hide_role=hide_role)
        return info