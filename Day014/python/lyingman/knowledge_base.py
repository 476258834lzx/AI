"""RAG知识库模块 - 基于本地文件的向量存储"""
import os
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeItem:
    """知识条目"""
    content: str
    metadata: dict
    embedding: Optional[np.ndarray] = None


class KnowledgeBase:
    """本地知识库"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "knowledge_data")

        self.data_dir = data_dir
        self.knowledge_file = os.path.join(data_dir, "knowledge.npy")
        self.metadata_file = os.path.join(data_dir, "metadata.npy")

        os.makedirs(data_dir, exist_ok=True)
        self.items: List[KnowledgeItem] = []
        self._load()

    def _load(self):
        """加载知识库"""
        if os.path.exists(self.knowledge_file):
            try:
                data = np.load(self.knowledge_file, allow_pickle=True)
                metadata = np.load(self.metadata_file, allow_pickle=True) if os.path.exists(self.metadata_file) else []

                self.items = []
                for i, (content, meta) in enumerate(zip(data, metadata)):
                    self.items.append(KnowledgeItem(
                        content=content,
                        metadata=meta if isinstance(meta, dict) else {},
                    ))
            except Exception:
                self.items = []

    def _save(self):
        """保存知识库"""
        if not self.items:
            if os.path.exists(self.knowledge_file):
                os.remove(self.knowledge_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            return

        contents = [item.content for item in self.items]
        metadata = [item.metadata for item in self.items]

        np.save(self.knowledge_file, np.array(contents, dtype=object))
        np.save(self.metadata_file, np.array(metadata, dtype=object))

    def add(self, content: str, metadata: dict = None):
        """添加知识条目"""
        self.items.append(KnowledgeItem(
            content=content,
            metadata=metadata or {},
        ))
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        搜索知识库（简单关键词匹配）

        实际生产环境应使用embedding模型进行向量检索
        """
        results = []
        query_lower = query.lower()

        for item in self.items:
            # 简单关键词匹配
            score = 0
            keywords = query_lower.split()

            for keyword in keywords:
                if keyword in item.content.lower():
                    score += 1
                for key in item.metadata:
                    if keyword in str(item.metadata[key]).lower():
                        score += 0.5

            if score > 0:
                results.append({
                    "content": item.content,
                    "metadata": item.metadata,
                    "score": score,
                })

        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all(self) -> List[dict]:
        """获取所有知识"""
        return [
            {"content": item.content, "metadata": item.metadata}
            for item in self.items
        ]

    def clear(self):
        """清空知识库"""
        self.items = []
        self._save()


# ===== 狼人杀比赛复盘数据 =====

GAME_REPLAYS = [
    # ===== 经典局复盘 =====
    {
        "content": """【第一局】12人标准局复盘

角色配置：4狼、4神（预言家、女巫、猎人、白痴）、4民

第一天：
- 1号跳预言家发2号金水，5号跳预言家发3号金水
- 狼队选择让5号悍跳，配合8号倒钩
- 1号拿到警徽，2号被抗推（狼队冲票）
- 女巫开药救了自己

第二天：
- 预言家1号报2号金水，5号报3号金水
- 场上出现真假预言家对跳
- 狼队8号倒钩投票1号，试图混淆好人
- 猎人带走8号，暴露狼队

第三天：
- 好人认清5号是狼人
- 投出3号和5号
- 狼队崩盘

【战术分析】
1. 悍跳狼5号发后置位3号金水，试图抢身份
2. 倒钩狼8号站边真预言家1号，获取信任
3. 好人通过行为分析识别狼队

【关键点】
- 预言家要敢报查验，1号报2号金水是正确的
- 好人要关注票型，8号倒钩冲票暴露身份""",
        "metadata": {
            "type": "比赛复盘",
            "player_count": 12,
            "roles": "标准局",
            "tags": ["悍跳", "倒钩", "预言家对跳"],
            "difficulty": "高级",
        }
    },

    {
        "content": """【第二局】9人速推局复盘

角色配置：3狼、3神、3民

第一天警上只有预言家一人，拿到警徽。
预言家第一晚查到6号是狼人，第二天白天报6号查杀。
6号被迫自爆，进入夜晚。

第二天：
- 女巫毒了7号（狼队队友）
- 场上存活7人，狼队只剩1狼

第三天：
- 好人全票投出剩余狼人
- 游戏结束，好人胜利

【战术分析】
1. 预言家首夜查验直接找到狼人
2. 女巫根据银水和发言判断7号是狼人
3. 狼队节奏崩盘

【关键点】
- 预言家查验是狼人杀最关键的信息
- 女巫毒人需要抿身份能力""",
        "metadata": {
            "type": "比赛复盘",
            "player_count": 9,
            "roles": "速推局",
            "tags": ["预言家首验", "女巫毒人", "速推"],
            "difficulty": "中级",
        }
    },

    {
        "content": """【第三局】狼王局复盘

角色配置：狼王、2普通狼人、预言家、女巫、猎人、白痴、守卫、4民

关键转折：
- 狼王白天自爆，带走预言家
- 预言家死前查验10号是狼人
- 女巫毒了另一个狼人

狼队崩盘原因：
1. 狼王自爆时机过早
2. 带走预言家反而让真查验信息公开
3. 剩余狼人无法对抗

【战术分析】
1. 狼王自爆时机选择很重要
2. 不应该带神职，应该带抿身份能力强的玩家
3. 好人利用遗言信息分析狼坑

【关键点】
- 狼王自爆是双刃剑
- 预言家遗言是重要信息
- 要考虑自爆带来的信息交换""",
        "metadata": {
            "type": "比赛复盘",
            "player_count": 12,
            "roles": "狼王局",
            "tags": ["狼王自爆", "预言家遗言", "双刃剑"],
            "difficulty": "高级",
        }
    },

    {
        "content": """【第四局】白痴翻牌局复盘

关键对局：
- 4号被抗推，翻牌白痴
- 好人认为4号是狼人抗推

第二天分析：
- 4号发言：我是白痴，昨天被抗推
- 好人开始怀疑
- 5号发言：4号可能是真白痴

第三天：
- 6号被抗推
- 4号白痴继续存活

结果：好人认错，但最终仍获胜

【战术分析】
1. 白痴翻牌后继续发言干扰狼人
2. 白痴要利用翻牌机会跳明身份带队
3. 好人要学会分辨白痴翻牌和狼人骗身份

【关键点】
- 白痴翻牌是强防御技能
- 翻牌后要强势发言
- 好人要保持冷静判断""",
        "metadata": {
            "type": "比赛复盘",
            "player_count": 12,
            "roles": "标准局",
            "tags": ["白痴翻牌", "身份识别", "带队"],
            "difficulty": "中级",
        }
    },

    {
        "content": """【第五局】守卫守人局复盘

关键操作：
- 第一晚守卫守预言家
- 狼人刀了预言家（空刀）

第二天：
- 预言家存活，场上平安夜
- 预言家报查验信息
- 好人分析：狼队刀空，预判狼队刀法

第三晚：
- 守卫继续守预言家
- 狼队再次空刀

第四天：
- 好人领先，狼队崩盘

【战术分析】
1. 守卫第一晚守预言家是标准开局
2. 狼队空刀说明抿身份能力差
3. 好人利用空刀信息分析

【关键点】
- 守卫首夜守预言家是共识
- 空刀是好人的信息优势
- 要学会利用一切信息分析""",
        "metadata": {
            "type": "比赛复盘",
            "player_count": 12,
            "roles": "标准局",
            "tags": ["守卫守人", "空刀", "预判"],
            "difficulty": "初级",
        }
    },

    # ===== 高级战术 =====
    {
        "content": """【高级战术】自刀骗药

适用场景：狼人第一晚刀自己博取信任

操作流程：
1. 狼人第一晚刀自己
2. 女巫救人（银水）
3. 狼人拿银水做身份
4. 后期跳神职或带队

成功条件：
- 女巫必须救人
- 狼人发言要像好人
- 不能太早暴露

风险：
- 女巫不救直接崩盘
- 被抿出来会背大锅

【案例】
1号狼人自刀，2号女巫救人。1号发言：我是好人，第一天被刀了，应该是好人刀我。后期跳预言家发2号金水，成功骗取信任。""",
        "metadata": {
            "type": "高级战术",
            "tags": ["自刀", "骗药", "银水"],
            "difficulty": "高级",
        }
    },

    {
        "content": """【高级战术】阴阳倒钩

定义：表面上站边真预言家，实际上帮助狼队

操作流程：
1. 狼队悍跳发后置位金水
2. 狼人倒钩站边真预言家
3. 真预言家阵营误以为狼人是好人
4. 倒钩狼引导投票抗推真好人

识别方法：
- 观察票型，倒钩狼通常会冲票
- 分析发言逻辑
- 关注倒钩狼的行为一致性

【案例】
真预言家1号，假预言家5号（狼人）。狼队安排8号倒钩站边1号，8号发言：我觉得1号是真的，5号发后置位金水太刻意。实际8号引导投票投了2号（好人）。""",
        "metadata": {
            "type": "高级战术",
            "tags": ["倒钩", "阴阳倒钩", "伪装"],
            "difficulty": "高级",
        }
    },

    {
        "content": """【高级战术】冲票战术

定义：狼队集体投票某目标，强行抗推好人

操作流程：
1. 悍跳狼发假查验
2. 狼队集体冲票某个好人
3. 好人被抗推
4. 悍跳狼拿警徽带队

反制方法：
1. 统计票型，找出狼队
2. 真预言家坚持报查验
3. 好人不要被带节奏

【案例】
狼队安排悍跳5号发1号金水，狼队8、9、10号集体冲票2号（好人），2号被抗推。后期好人发现票型异常，开始分析狼队。""",
        "metadata": {
            "type": "高级战术",
            "tags": ["冲票", "悍跳", "抗推"],
            "difficulty": "中级",
        }
    },

    {
        "content": """【高级战术】深水狼

定义：完全不悍跳，低调潜水到最后

操作流程：
1. 第一天不跳任何身份
2. 发言中规中矩，不引起注意
3. 关键时刻倒钩站边
4. 后期引导投票

成功条件：
- 场上没有明显狼人
- 发言要像好人
- 不能被抿出来

风险：
- 如果其他狼人被出，可能暴露
- 容易被抗推

【案例】
狼人8号全程不跳身份，发言：我就是好人，过。投票跟随大多数。最终存活到最后，狼队获胜。""",
        "metadata": {
            "type": "高级战术",
            "tags": ["深水狼", "潜水", "低调"],
            "difficulty": "高级",
        }
    },

    {
        "content": """【角色教学】预言家篇

核心职责：报查验，带队投狼

第一天策略：
1. 上警拿警徽
2. 报查验（通常查验警上）
3. 留好警徽流

查验优先级：
1. 警上后置位
2. 发言像狼的
3. 投票异常的

常见错误：
1. 不敢报查验
2. 查验不报
3. 警徽流乱留

【进阶技巧】
1. 警徽流可以留警下疑点大的
2. 查验要结合发言分析
3. 遗言要清晰报信息""",
        "metadata": {
            "type": "角色教学",
            "role": "预言家",
            "tags": ["预言家", "查验", "带队"],
            "difficulty": "初级",
        }
    },

    {
        "content": """【角色教学】女巫篇

核心职责：救人、毒人、报银水

用药策略：
- 解药：第一晚通常救人（除非想打自刀骗药）
- 毒药：谨慎使用，宁可错杀不可放过

银水分析：
1. 银水可能是好人
2. 也可能是狼人自刀骗药
3. 要结合发言判断

毒人优先级：
1. 铁狼
2. 抿身份抿不出来的
3. 疑似狼人的

常见错误：
1. 第一晚不救人
2. 毒药乱用
3. 不报银水

【进阶技巧】
1. 第一晚救人可以守预言家
2. 毒药可以追轮次
3. 银水要结合发言分析""",
        "metadata": {
            "type": "角色教学",
            "role": "女巫",
            "tags": ["女巫", "银水", "毒人"],
            "difficulty": "初级",
        }
    },

    {
        "content": """【角色教学】狼人篇

核心职责：隐藏身份、配合刀人、悍跳或潜水

狼队配合：
1. 统一刀人目标
2. 决定悍跳还是潜水
3. 安排倒钩狼

悍跳策略：
1. 发后置位金水
2. 发言要像真神职
3. 留好警徽流

潜水策略：
1. 发言中规中矩
2. 关键时刻倒钩
3. 不引起注意

刀人优先级：
1. 预言家
2. 女巫
3. 其他神职
4. 抿身份强的

【进阶技巧】
1. 狼队要有分工
2. 悍跳狼和倒钩狼配合
3. 注意刀人顺序""",
        "metadata": {
            "type": "角色教学",
            "role": "狼人",
            "tags": ["狼人", "悍跳", "配合"],
            "difficulty": "初级",
        }
    },
]


def init_knowledge_base():
    """初始化知识库"""
    kb = KnowledgeBase()

    # 如果知识库为空，添加预设数据
    if not kb.items:
        for replay in GAME_REPLAYS:
            kb.add(
                content=replay["content"],
                metadata=replay["metadata"],
            )

        print(f"知识库已初始化，包含 {len(GAME_REPLAYS)} 条记录")

    return kb


if __name__ == "__main__":
    kb = init_knowledge_base()
    print(f"知识库条目数: {len(kb.items)}")

    # 测试搜索
    results = kb.search("预言家对跳")
    print(f"\n搜索'预言家对跳'结果:")
    for r in results:
        print(f"  - {r['metadata']}")
