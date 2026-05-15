"""狼人杀游戏主入口"""
import os
from typing import Optional

from .game_state import GameState, PlayerStatus
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

            # 宣布夜晚死亡信息
            if self.god.current_round and self.god.current_round.deaths:
                print("\n死亡玩家:")
                for pid in self.god.current_round.deaths:
                    player = self.game_state.players.get(pid)
                    if player:
                        print(f"  - {player.name}")
            else:
                print("昨晚是平安夜")

            # 宣布熊咆哮
            self._announce_bear_roar()

            # 处理恶灵骑士反伤
            self._handle_evil_knight_revenge()

            # 检查第三方获胜（只有存在丘比特且有人狼链情侣时才判定）
            if self.game_state.cupid_id is not None and self.game_state.lovers:
                if self.game_state.is_couple_wolf_human() and self.game_state.is_love_chain_alone():
                    print("\n=== 第三方获胜 ===")
                    print("人狼链情侣存活到最后获胜！")
                    self.print_winner("neutral")
                    break

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

        # ===== 第一晚特殊：丘比特、野孩子 =====
        if self.game_state.current_day == 1:
            self._handle_cupid_link()
            self._handle_wild_child_choice()

        # ===== 狼人商讨阶段 =====
        print("\n[狼人商讨阶段]")
        wolves = self._get_wolf_players()
        if wolves:
            self._handle_wolf_discuss(wolves)

        # ===== 狼人刀人阶段（共识后） =====
        if self.game_state.wolf_kill_target is not None:
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
                        # 恶灵骑士：查验显示狼人，但次日反伤预言家
                        result = "狼人"
                        self.game_state.seer_check_result = {"target_id": target, "result": result}
                        self.game_state.evil_knight_checked_by = seer_id
                        print(f"预言家查验了 {target_player.name}: {result}")
                        print(f"  （实际为恶灵骑士，次日将反伤预言家）")
                    elif target_player.role.name.endswith("狼"):
                        result = "狼人"
                        self.game_state.seer_check_result = {"target_id": target, "result": result}
                        print(f"预言家查验了 {target_player.name}: {result}")
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

            # 女巫毒人
            if not self.game_state.witch_poison_used:
                poison_target = agent.decide_night_action("witch_poison")
                if poison_target:
                    target_player = self.game_state.players.get(poison_target)
                    if target_player and target_player.role:
                        if target_player.role.name == "恶灵骑士":
                            # 恶灵骑士反伤：女巫毒恶灵骑士，女巫自己死，恶灵骑士不死
                            self.game_state.evil_knight_poisoned_by = witch_id
                            witch = self.game_state.players.get(witch_id)
                            if witch and witch.is_alive():
                                witch.status = PlayerStatus.DEAD_NIGHT
                                print(f"女巫试图毒 {target_player.name}，但被恶灵骑士反伤！女巫死亡！")
                        else:
                            target_player.is_poisoned = True
                            self.game_state.witch_poison_used = True
                            print(f"女巫毒了 {target_player.name}")

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

        # ===== 每晚：吹笛人、狐狸 =====
        self._handle_flutist_charm()
        self._handle_fox_check()

        # 传入猎人回调函数
        self.god.night_settle(hunter_callback=self._hunter_shoot_callback)

    def _hunter_shoot_callback(self, hunter_id: int, candidates: list[int]) -> int:
        """猎人开枪回调 - 让猎人的agent选择开枪目标"""
        agent = self.players.get(hunter_id)
        if not agent:
            return None
        return agent.decide_hunter_shoot(candidates)

    def _handle_wolf_discuss(self, wolves: list[int]):
        """处理狼人夜间商讨（共识机制）"""
        # 检查狼人列表
        if not wolves:
            print("没有存活的狼人")
            return

        alive = self.game_state.get_alive_players()
        non_wolf_targets = [p for p in alive if p.id not in wolves]

        if not non_wolf_targets:
            print("没有可刀的目标")
            return

        # 1. 狼人商讨阶段：每个狼人发言讨论
        print("狼人密谋中...")
        wolf_discussions = {}
        for pid in wolves:
            agent = self.players.get(pid)
            player = self.game_state.players.get(pid)
            result = agent.wolf_discuss(
                wolf_ids=wolves,
                non_wolf_targets=[p.id for p in non_wolf_targets],
                other_wolf_suggestions=self.game_state.wolf_discuss_proposals
            )
            wolf_discussions[pid] = result
            target_id = result.get('target_id')
            target_name = self.game_state.players.get(target_id).name if target_id is not None else "未知"
            print(f"  {player.name} 提议刀 {target_name}")

        # 2. 统计提议，尝试达成共识
        from collections import Counter
        proposals = [d['target_id'] for d in wolf_discussions.values() if d.get('target_id') is not None]
        proposal_counts = Counter(proposals)

        # 如果没有狼人或没有有效提议
        if not wolves or not proposals:
            print("没有狼人或无法达成刀人共识")
            return

        # 找出最多提议
        max_count = max(proposal_counts.values())
        top_targets = [tid for tid, cnt in proposal_counts.items() if cnt == max_count]

        # 如果有唯一最高票，达成共识
        if len(top_targets) == 1:
            self.game_state.wolf_consensus_target = top_targets[0]
            self.game_state.wolf_kill_target = top_targets[0]
            print(f"狼人达成共识！今晚刀 {self.game_state.players[top_targets[0]].name}")
        else:
            # 平票/多方案：使用多数投票结果
            import random
            self.game_state.wolf_kill_target = random.choice(top_targets)
            print(f"狼人意见不一，按多数决定刀 {self.game_state.players[self.game_state.wolf_kill_target].name}")

        # 3. 狼人确认刀人（条件边）
        for pid in wolves:
            agent = self.players.get(pid)
            result = agent.wolf_confirm_kill(self.game_state.wolf_kill_target)
            print(f"  {self.game_state.players[pid].name} 确认刀人")

        # 清理状态
        self.game_state.wolf_discuss_proposals.clear()
        self.game_state.wolf_awaiting_confirm.clear()
        self.game_state.wolf_consensus_target = None

    def _handle_cupid_link(self):
        """处理丘比特指定情侣"""
        cupids = self._get_role_players("丘比特")
        if not cupids:
            return

        cupid_id = cupids[0]
        agent = self.players.get(cupid_id)
        if not agent:
            return

        print("\n[丘比特阶段]")
        alive = self.game_state.get_alive_players()
        candidates = [p.id for p in alive if p.id != cupid_id]

        if len(candidates) < 2:
            print("存活玩家不足，丘比特无法指定情侣")
            return

        # 调用丘比特选择情侣
        result = agent.decide_cupid_link(candidates)
        if result and 'lover1' in result and 'lover2' in result:
            lover1_id = result['lover1']
            lover2_id = result['lover2']

            # 防御性检查
            if lover1_id is None or lover2_id is None:
                print("丘比特选择无效")
                return
            if lover1_id not in candidates or lover2_id not in candidates:
                print("丘比特选择了无效玩家")
                return

            self.game_state.set_lovers(lover1_id, lover2_id)
            self.game_state.love_chain_type = self.game_state.get_love_type(lover1_id)
            self.game_state.cupid_id = cupid_id

            lover1_name = self.game_state.players.get(lover1_id).name
            lover2_name = self.game_state.players.get(lover2_id).name
            print(f"丘比特指定情侣: {lover1_name} 和 {lover2_name}")
            print(f"情侣类型: {self.game_state.love_chain_type}")

    def _handle_wild_child_choice(self):
        """处理野孩子选择偶像"""
        wild_children = self._get_role_players("野孩子")
        if not wild_children:
            return

        wild_child_id = wild_children[0]
        # 野孩子只在第一晚选择偶像
        if self.game_state.current_day > 1:
            if self.game_state.wild_child_idol is None:
                print(f"野孩子尚未选择偶像（将在第一晚选择）")
            return

        agent = self.players.get(wild_child_id)
        if not agent:
            return

        print("\n[野孩子选择偶像阶段]")
        alive = self.game_state.get_alive_players()
        candidates = [p.id for p in alive if p.id != wild_child_id]

        if not candidates:
            return

        # 如果还没选择偶像
        if self.game_state.wild_child_idol is None:
            self.game_state.set_wild_child_id(wild_child_id)
            result = agent.decide_wild_child_idol(candidates)
            if result and 'idol' in result:
                idol_id = result['idol']
                self.game_state.set_wild_child_idol(idol_id)
                idol_name = self.game_state.players.get(idol_id).name
                print(f"野孩子选择偶像: {idol_name}")

                # 检查是否偶像已死亡（立即转换）
                if self.game_state.is_wild_child_converted():
                    print(f"偶像已死亡，野孩子立即转换为狼人阵营！")

    def _handle_flutist_charm(self):
        """处理吹笛人迷惑"""
        flutists = self._get_role_players("吹笛人")
        if not flutists:
            return

        flutist_id = flutists[0]
        agent = self.players.get(flutist_id)
        if not agent:
            return

        print("\n[吹笛人迷惑阶段]")
        alive = self.game_state.get_alive_players()
        candidates = [p.id for p in alive if p.id != flutist_id and not self.game_state.is_cursed(p.id)]

        if len(candidates) < 2:
            print("无可迷惑目标")
            return

        result = agent.decide_flutist_charm(candidates)
        if result and 'target1' in result and 'target2' in result:
            target1_id = result.get('target1')
            target2_id = result.get('target2')

            # 获取玩家名称
            target1 = self.game_state.players.get(target1_id) if target1_id else None
            target2 = self.game_state.players.get(target2_id) if target2_id else None

            # 防御性检查
            if target1_id is None or target2_id is None:
                print("吹笛人选择无效")
                return

            if target1 and target1_id not in self.game_state.flutist_charmed_ids and target1_id in candidates:
                self.game_state.flutist_charmed_ids.append(target1_id)
            if target2 and target2_id not in self.game_state.flutist_charmed_ids and target2_id in candidates:
                self.game_state.flutist_charmed_ids.append(target2_id)

            target1_name = target1.name if target1 else "未知"
            target2_name = target2.name if target2 else "未知"
            print(f"吹笛人迷惑了: {target1_name} 和 {target2_name}")

    def _handle_fox_check(self):
        """处理狐狸查验"""
        foxes = self._get_role_players("狐狸")
        if not foxes:
            return

        fox_id = foxes[0]
        agent = self.players.get(fox_id)
        if not agent:
            return

        # 狐狸失去技能后不再查验
        if self.game_state.fox_lost_skill:
            print("\n[狐狸] 技能已失去，无法查验")
            return

        print("\n[狐狸查验阶段]")
        alive = self.game_state.get_alive_players()
        candidates = [p.id for p in alive if p.id != fox_id]

        if len(candidates) < 3:
            print("存活玩家不足，狐狸无法查验")
            return

        result = agent.decide_fox_check(candidates)
        if result and 'target1' in result and 'target2' in result and 'target3' in result:
            target_ids = [result['target1'], result['target2'], result['target3']]
            has_wolf = False

            for tid in target_ids:
                target = self.game_state.players.get(tid)
                if target and target.role and target.role.name.endswith("狼"):
                    has_wolf = True
                    break

            self.game_state.fox_last_check_has_wolf = has_wolf
            if has_wolf:
                self.game_state.fox_lost_skill = True
                print(f"狐狸查验范围内有狼人，技能失去！")
            else:
                print(f"狐狸查验范围内无狼人，今晚免疫狼刀")

    def _announce_bear_roar(self):
        """宣布熊咆哮信息"""
        bears = self._get_role_players("熊")
        if not bears:
            return

        bear_id = bears[0]

        # 检查熊的相邻位置是否有狼人存活
        alive = self.game_state.get_alive_players()
        player_ids = [p.id for p in alive]
        bear_idx = player_ids.index(bear_id) if bear_id in player_ids else -1

        if bear_idx >= 0:
            prev_idx = (bear_idx - 1) % len(player_ids)
            next_idx = (bear_idx + 1) % len(player_ids)
            prev_id = player_ids[prev_idx]
            next_id = player_ids[next_idx]

            # 检查相邻位置是否有狼人
            prev_player = self.game_state.players.get(prev_id)
            next_player = self.game_state.players.get(next_id)

            prev_is_wolf = prev_player and prev_player.role and prev_player.role.name.endswith("狼")
            next_is_wolf = next_player and next_player.role and next_player.role.name.endswith("狼")

            if prev_is_wolf or next_is_wolf:
                self.game_state.bear_roared_today = True
                print(f"\n【法官宣布】熊咆哮了！")
            else:
                self.game_state.bear_roared_today = False

    def _handle_evil_knight_revenge(self):
        """处理恶灵骑士反伤 - 在白天宣布"""
        # 查验反伤
        if self.game_state.evil_knight_checked_by is not None:
            seer = self.game_state.players.get(self.game_state.evil_knight_checked_by)
            if seer and seer.is_alive():
                seer.status = PlayerStatus.DEAD_DAY
                if self.god.current_round:
                    self.god.current_round.deaths.append(self.game_state.evil_knight_checked_by)
                print(f"\n【法官宣布】{seer.name}因查验恶灵骑士被反伤死亡！")
            self.game_state.evil_knight_checked_by = None

        # 毒杀反伤
        if self.game_state.evil_knight_poisoned_by is not None:
            witch = self.game_state.players.get(self.game_state.evil_knight_poisoned_by)
            if witch and witch.is_alive():
                witch.status = PlayerStatus.DEAD_DAY
                if self.god.current_round:
                    self.god.current_round.deaths.append(self.game_state.evil_knight_poisoned_by)
                print(f"\n【法官宣布】{witch.name}因毒杀恶灵骑士被反伤死亡！")
            self.game_state.evil_knight_poisoned_by = None

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
