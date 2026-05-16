"""狼人杀游戏主入口 - 基于 LangGraph 实现"""
import os
from typing import Optional

from .game_state import GameState, PlayerStatus
from .referee import Referee
from .god import God
from .graph import WerewolfGraph, GameGraphState
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
        self.graph: Optional[WerewolfGraph] = None

    def setup_game(self):
        """设置游戏 - 使用 LangGraph"""
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

        # 构建 LangGraph
        self.graph = WerewolfGraph(self.player_count)
        self.graph.build()

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

    def print_role_distribution(self):
        """打印角色分布"""
        print("\n=== 角色分配 ===")
        for p in self.game_state.players.values():
            print(f"{p.name}: {p.role.name if p.role else '未知'}")

    def run(self):
        """运行游戏 - 使用 LangGraph"""
        self.setup_game()
        self.print_welcome()

        if self.mode == "watch":
            print("\n观战模式启动...\n")
        else:
            print("\n游玩模式启动...\n")
            if self.human_player:
                role = self.human_player.get_role()
                print(f"\n你的角色是: 【{role}】")
                print(get_role_prompt(role))

        # 打印角色分配
        self.print_role_distribution()

        # 构建初始状态
        initial_state = GameGraphState(
            game_state=self.game_state,
            referee=self.referee,
            god=self.god,
            players=self.players,
            current_phase="init",
            current_day=1,
            winner=None,
            messages=[],
        )

        # 运行 LangGraph
        print("\n" + "=" * 50)
        print("游戏开始")
        print("=" * 50)

        result = self.graph.graph.invoke(initial_state)

        # 打印游戏结果
        print("\n" + "=" * 50)
        for msg in result.get("messages", []):
            print(msg)
        print("=" * 50)

        # 打印获胜方
        winner = result.get("winner")
        if winner:
            self.print_winner(winner)

        return result


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

    game.run()


if __name__ == "__main__":
    main()
