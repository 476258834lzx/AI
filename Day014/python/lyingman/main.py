"""狼人杀游戏主入口"""
import os
from typing import Optional

from .game_state import GameState
from .referee import Referee
from .god import God
from .graph import WerewolfGraph
from .player_agent import PlayerAgent, HumanPlayerAgent
from .prompts import get_role_prompt, get_phase_prompt
from .emotion_skills import create_emotion_system, SPEECH_STYLES, STRATEGIES


class WerewolfGame:
    """狼人杀游戏主类"""

    def __init__(
        self,
        player_count: int = 8,
        mode: str = "watch",
        human_player_id: Optional[int] = None,
    ):
        self.player_count = player_count
        self.mode = mode
        self.human_player_id = human_player_id
        self.game_state: Optional[GameState] = None
        self.referee: Optional[Referee] = None
        self.god: Optional[God] = None
        self.players: dict[int, PlayerAgent] = {}
        self.human_player: Optional[HumanPlayerAgent] = None

    def setup_game(self):
        """设置游戏"""
        self.referee = Referee(self.player_count)
        player_names = [f"玩家{i+1}" for i in range(self.player_count)]

        for i in range(self.player_count):
            if self.mode == "play" and i == self.human_player_id:
                self.human_player = HumanPlayerAgent(player_id=i, player_name=player_names[i])
                self.players[i] = self.human_player
            else:
                import random
                styles = list(SPEECH_STYLES.keys())
                primary = random.choice(styles)
                secondary = random.choice([s for s in styles if s != primary])

                agent = PlayerAgent(player_id=i, player_name=player_names[i])
                create_emotion_system(i, primary_style=primary, secondary_style=secondary)
                self.players[i] = agent

        self.game_state = self.referee.init_game(player_names)
        self.god = God(self.game_state)

        for agent in self.players.values():
            agent.set_game_state(self.game_state)

    def print_welcome(self):
        """打印欢迎信息"""
        print("=" * 50)
        print("          狼人杀游戏 Werewolf")
        print("=" * 50)
        print(f"玩家数量: {self.player_count}")
        print(f"游戏模式: {'观战模式' if self.mode == 'watch' else '游玩模式'}")
        if self.mode == "play" and self.human_player:
            print(f"你的角色: {self.human_player.get_role()}")
        print("=" * 50)
        print()

    def print_winner(self, winner: str):
        """打印获胜方"""
        print("\n" + "=" * 50)
        if winner == "good":
            print("           游戏结束：好人胜利！")
        elif winner == "wolf":
            print("           游戏结束：狼人胜利！")
        elif winner == "neutral":
            print("           游戏结束：第三方胜利！")
        else:
            print("           游戏结束")
        print("=" * 50)

        print("\n存活玩家:")
        for p in self.game_state.players.values():
            if p.is_alive():
                print(f"  {p.name}: {p.role.name if p.role else '未知'}")

    def run_watch_mode(self):
        """运行观战模式"""
        print("观战模式启动...\n")
        self.setup_game()
        self.print_welcome()

        print("\n=== 角色分配 ===")
        for p in self.game_state.players.values():
            print(f"{p.name}: {p.role.name if p.role else '未知'}")

        self.run_game_loop()

    def run_play_mode(self):
        """运行游玩模式"""
        if self.human_player_id is None:
            self.human_player_id = 0

        print("游玩模式启动...\n")
        self.setup_game()
        self.print_welcome()

        if self.human_player:
            role = self.human_player.get_role()
            print(f"\n你的角色是: 【{role}】")
            print(get_role_prompt(role))

        self.run_game_loop()

    def run_game_loop(self):
        """运行游戏循环"""
        day = 1
        max_days = 20

        while day <= max_days:
            self.game_state.current_day = day

            if day == 1:
                print("\n=== 警长竞选 ===")
                self.handle_sheriff_election()

            print(f"\n=== 第{day}天白天发言 ===")
            self.handle_day_speech()

            print(f"\n=== 第{day}天放逐投票 ===")
            self.handle_day_vote()

            winner = self.referee.check_win_condition()
            if winner:
                self.print_winner(winner)
                break

            print(f"\n=== 第{day}天夜晚 ===")
            self.handle_night()

            print("\n=== 天亮了 ===")
            self.god.add_to_history()

            if self.god.current_round and self.god.current_round.deaths:
                print("\n死亡玩家:")
                for pid in self.god.current_round.deaths:
                    player = self.game_state.players.get(pid)
                    if player:
                        print(f"  - {player.name}")
            else:
                print("昨晚是平安夜")

            winner = self.referee.check_win_condition()
            if winner:
                self.print_winner(winner)
                break

            day += 1

    def handle_sheriff_election(self):
        """处理警长竞选"""
        if not self.game_state or not self.god:
            return

        candidates = []
        for pid, agent in self.players.items():
            player = self.game_state.players.get(pid)
            if not player or not player.is_alive():
                continue
            if isinstance(agent, HumanPlayerAgent):
                if agent.decide_sheriff_run():
                    candidates.append(pid)
            else:
                if agent.decide_sheriff_run():
                    candidates.append(pid)

        print(f"上警玩家: {len(candidates)}人")
        for pid in candidates:
            print(f"  - {self.game_state.players[pid].name}")

        if not candidates:
            print("无人上警")
            return

        for pid in candidates:
            agent = self.players.get(pid)
            player = self.game_state.players.get(pid)
            result = agent.speak("请发表竞选发言")
            print(f"\n【{player.name}】竞选发言:")
            print(result["speech"])
            self.god.record_day_action(pid, result["speech"])

        print("\n=== 投票 ===")
        votes = {pid: 0 for pid in candidates}

        for pid, agent in self.players.items():
            player = self.game_state.players.get(pid)
            if not player or not player.is_alive():
                continue
            vote_target = agent.vote(candidates)
            if vote_target in votes:
                votes[vote_target] += 1

        print("\n票数统计:")
        for pid, count in votes.items():
            print(f"  {self.game_state.players[pid].name}: {count}票")

        max_votes = max(votes.values())
        winners = [pid for pid, v in votes.items() if v == max_votes]

        if len(winners) == 1:
            self.game_state.sheriff_id = winners[0]
            print(f"\n{self.game_state.players[winners[0]].name} 当选警长！")
        else:
            print("\n警长竞选平票，无人当选")

    def handle_day_speech(self):
        """处理白天发言"""
        if not self.game_state or not self.god:
            return

        speech_order = self.god.get_speech_order()
        print(f"发言顺序: {[self.game_state.players[pid].name for pid in speech_order]}")

        for i, pid in enumerate(speech_order):
            agent = self.players.get(pid)
            player = self.game_state.players.get(pid)
            if not player or not player.is_alive():
                continue

            result = agent.speak(f"你是第{i+1}个发言")
            print(f"\n【{player.name}】发言:")
            print(result["speech"])
            self.god.record_day_action(pid, result["speech"])

    def handle_day_vote(self):
        """处理白天投票"""
        if not self.game_state or not self.god:
            return

        alive = self.game_state.get_alive_players()
        candidates = [p.id for p in alive]

        for p in self.game_state.players.values():
            p.vote_count = 0

        print("投票中...")
        for pid in candidates:
            agent = self.players.get(pid)
            player = self.game_state.players.get(pid)
            if not player or not player.can_vote():
                continue
            vote_target = agent.vote(candidates)
            if vote_target in candidates:
                vote_count = 1.5 if player.is_sheriff else 1
                self.game_state.players[vote_target].vote_count += vote_count

        print("\n票数统计:")
        for p in self.game_state.players.values():
            if p.is_alive() and p.vote_count > 0:
                print(f"  {p.name}: {p.vote_count}票")

        max_votes = 0
        candidates_to_eliminate = []
        for p in self.game_state.players.values():
            if p.is_alive() and p.vote_count > max_votes:
                max_votes = p.vote_count
                candidates_to_eliminate = [p.id]
            elif p.is_alive() and p.vote_count == max_votes:
                candidates_to_eliminate.append(p.id)

        if len(candidates_to_eliminate) == 1:
            eliminated = candidates_to_eliminate[0]
            player = self.game_state.players[eliminated]
            player.status = "dead_day"
            print(f"\n{player.name} 被投票出局！")
        else:
            print("\n投票平票，无人出局")

    def handle_night(self):
        """处理夜晚流程"""
        if not self.game_state or not self.god:
            return

        self.god.start_night()

        print("\n[狼人刀人阶段]")
        wolves = self._get_wolf_players()
        if wolves:
            wolf_targets = []
            for pid in wolves:
                agent = self.players.get(pid)
                target = agent.decide_night_action("wolf_kill")
                if target:
                    wolf_targets.append(target)

            if wolf_targets:
                from collections import Counter
                target_counts = Counter(wolf_targets)
                max_count = max(target_counts.values())
                top_targets = [t for t, c in target_counts.items() if c == max_count]
                import random
                self.game_state.wolf_kill_target = random.choice(top_targets)
                print(f"狼人刀了 {self.game_state.players[self.game_state.wolf_kill_target].name}")

        print("\n[预言家查验阶段]")
        seers = self._get_role_players("预言家")
        if seers:
            seer_id = seers[0]
            agent = self.players.get(seer_id)
            target = agent.decide_night_action("seer_check")
            if target:
                target_player = self.game_state.players.get(target)
                if target_player and target_player.role:
                    if target_player.role.name == "恶灵骑士":
                        result = "好人"
                    elif target_player.role.name.endswith("狼"):
                        result = "狼人"
                    else:
                        result = "好人"
                else:
                    result = "好人"
                self.game_state.seer_check_result = {"target_id": target, "result": result}
                print(f"预言家查验了 {target_player.name}: {result}")

        print("\n[女巫阶段]")
        witches = self._get_role_players("女巫")
        if witches:
            witch_id = witches[0]
            agent = self.players.get(witch_id)
            if self.game_state.wolf_kill_target and not self.game_state.witch_heal_used:
                target = agent.decide_night_action("witch_heal")
                if target:
                    self.game_state.witch_heal_used = True
                    self.game_state.revives.append(target)
                    print(f"女巫救了 {self.game_state.players[target].name}")

        print("\n[守卫守护阶段]")
        guardians = self._get_role_players("守卫")
        if guardians:
            guardian_id = guardians[0]
            agent = self.players.get(guardian_id)
            target = agent.decide_night_action("guardian_protect")
            if target and self.game_state.guardian_last_protect != target:
                self.game_state.guardian_last_protect = target
                self.game_state.players[target].is_protected = True
                print(f"守卫守护了 {self.game_state.players[target].name}")

        self.god.night_settle()

    def _get_role_players(self, role_name: str) -> list[int]:
        if not self.game_state:
            return []
        return [pid for pid, p in self.game_state.players.items()
                if p.is_alive() and p.role and p.role.name == role_name]

    def _get_wolf_players(self) -> list[int]:
        wolf_roles = ["狼人", "狼王", "白狼王", "狼美人", "恶灵骑士"]
        if not self.game_state:
            return []
        return [pid for pid, p in self.game_state.players.items()
                if p.is_alive() and p.role and p.role.name in wolf_roles]


def main():
    """主函数"""
    print("欢迎来到狼人杀游戏！\n")
    print("请选择游戏模式:")
    print("1. 观战模式 - 观看AI之间的对战")
    print("2. 游玩模式 - 亲自参与游戏")

    mode_choice = input("请输入选项 (1/2): ").strip()

    if mode_choice == "1":
        mode = "watch"
    elif mode_choice == "2":
        mode = "play"
    else:
        print("无效选择，默认观战模式")
        mode = "watch"

    print("\n请输入玩家数量（默认8人）:")
    try:
        count = int(input("> ").strip())
        if count < 5:
            print("玩家数量必须大于5人，默认为8人")
            count = 8
    except ValueError:
        count = 8

    game = WerewolfGame(player_count=count, mode=mode, human_player_id=None)

    if mode == "watch":
        game.run_watch_mode()
    else:
        game.run_play_mode()


if __name__ == "__main__":
    main()
