"""上帝 - 负责回合管理和信息记录"""
from typing import Optional
from .game_state import GameState, Round, RoundPhase, NightPhase, NightAction, PlayerStatus


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

    def record_day_action(self, speaker_id: int, content: str):
        """记录白天发言"""
        if self.current_round:
            from .game_state import DayAction
            action = DayAction(
                phase=self.current_round.phase,
                speaker_id=speaker_id,
                content=content,
            )
            self.current_round.actions.append(action)
            self.game_state.day_actions.append(action)

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

    def night_settle(self):
        """夜晚结算"""
        deaths = []

        # 1. 处理狼刀
        wolf_target = self.game_state.wolf_kill_target
        if wolf_target is not None:
            target = self.game_state.players.get(wolf_target)
            if target and target.is_alive():
                if not target.is_protected:
                    deaths.append(wolf_target)

        # 2. 处理女巫救人
        if wolf_target in self.game_state.revives:
            if wolf_target in deaths:
                deaths.remove(wolf_target)

        # 3. 处理女巫毒人
        for player in self.game_state.players.values():
            if player.is_poisoned:
                if player.id not in deaths:
                    deaths.append(player.id)

        # 4. 执行死亡
        for player_id in deaths:
            player = self.game_state.players.get(player_id)
            if player:
                player.status = PlayerStatus.DEAD_NIGHT
                self.current_round.deaths.append(player_id)

        # 5. 重置夜间状态
        self.game_state.wolf_kill_target = None
        self.game_state.seer_check_result = None
        self.game_state.witch_heal_used = False
        self.game_state.witch_poison_used = False
        for player in self.game_state.players.values():
            player.is_protected = False
            player.is_poisoned = False
            player.vote_count = 0

        return deaths

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

    def get_night_action_order(self) -> list[tuple[NightPhase, Optional[str]]]:
        """获取夜间动作顺序和对应角色"""
        return [
            (NightPhase.WOLF_KILL, "狼人"),
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