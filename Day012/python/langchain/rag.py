from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory,RunnableLambda,RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)
strOutputParser = StrOutputParser()

gangtiexia="""
钢铁侠（Iron Man）是漫威宇宙中最具代表性的超级英雄之一，由斯坦·李、拉里·利伯、唐·赫克和杰克·科比共同创作，首次登场于1963年的《悬疑故事》第39期。以下是漫威宇宙观下钢铁侠的完整介绍：
人物背景与生平
托尼·斯塔克（Tony Stark） 出生于美国纽约长岛，是霍华德·斯塔克（Howard Stark）和玛丽亚·斯塔克（Maria Stark）的独子。斯塔克家族是军火工业的巨头，霍华德·斯塔克是传奇发明家和企业家。
关键人生节点：
早年与转变
少年时期即展现天才级智商，17岁毕业于麻省理工学院
父母因车祸去世后，21岁继承斯塔克工业
在一次前往越南（后改为阿富汗）的武器展示途中，被恐怖分子"十戒帮"绑架
在囚禁期间，胸部被弹片重伤，与何银森（Ho Yinsen）合作制造了第一代方舟反应堆和马克1号装甲逃生
何银森为掩护托尼牺牲，这一事件深刻改变了托尼的人生观
英雄生涯
回归后宣布斯塔克工业停止武器生产，转型能源与科技领域
不断改进装甲技术，从马克1号发展至马克85号（终局之战）
创立复仇者联盟，提供资金、技术和总部（斯塔克大厦/复仇者基地）
与佩珀·波茨（Pepper Potts）结婚，育有女儿摩根·斯塔克
牺牲与传承
在《复仇者联盟4：终局之战》中，使用无限宝石打响指消灭灭霸及其大军
因无限宝石的能量反噬而牺牲，成为拯救宇宙的英雄
其遗产由妻子佩珀和女儿摩根继承，技术传承给蜘蛛侠彼得·帕克
核心能力
1. 超级装甲（Iron Man Armor）
托尼的核心战斗力来自自主研发的动力装甲系列：
基础系统：
方舟反应堆（Arc Reactor）：胸口的小型冷核聚变装置，为装甲提供近乎无限的能源，早期也用于维持磁铁防止弹片进入心脏
贾维斯/星期五（JARVIS/FRIDAY）：人工智能辅助系统，负责装甲控制、战术分析和通讯
纳米技术（Nano-Tech）：从马克50号开始采用纳米粒子构建装甲，可实时修复、变形出各种武器
装甲武器系统：
脉冲光束（Repulsor Rays）：手掌和胸部的能量武器
Uni-Beam：胸部集束炮，威力最大
微型导弹、激光切割、声波武器
飞行能力（通过脚底和手掌推进器）
力场护盾、电磁脉冲
特殊装甲型号：
反浩克装甲（Hulkbuster）：专门对抗绿巨人
太空装甲、深海装甲、隐形装甲等环境特化型
血边装甲（Bleeding Edge）：与托尼身体融合，从骨骼中生成
2. 天才级智力
地球上最聪明的人之一，与里德·理查兹、汉克·皮姆等齐名
精通物理学、工程学、计算机科学、人工智能等多个领域
快速学习能力，能在极端压力下即兴发明创造
3. 商业与政治影响力
斯塔克工业的CEO，曾经的世界首富
曾任美国国防部长、神盾局局长
在内战期间推动《超级英雄注册法案》
主要事迹与故事线
电影宇宙（MCU）关键事件：
《钢铁侠》（2008）
起源故事，制造马克1-3号装甲，击败铁芒果（Obadiah Stane）
《钢铁侠2》（2010）
应对钯元素中毒，开发新元素替代，与鞭索（Whiplash）和战争机器合作
《复仇者联盟》（2012）
纽约之战，扛着核弹穿越虫洞摧毁齐塔瑞母舰，奠定团队核心地位
《钢铁侠3》（2013）
应对满大人（Mandarin）和绝境病毒（Extremis），摧毁所有装甲后暂时退役
《复仇者联盟2：奥创纪元》（2015）
创造奥创（Ultron）试图保护地球，结果引发索科维亚危机，导致团队分裂
《美国队长3：内战》（2016）
支持超级英雄注册法案，与美国队长决裂，因冬兵杀害父母真相与队长和巴基激战
《复仇者联盟3：无限战争》（2018）
泰坦星与灭霸激战，被刺穿腹部，见证小蜘蛛化为灰烬，深受打击
《复仇者联盟4：终局之战》（2019）
发现时间劫持方案，与队长和解，最终戴上无限手套打响指，说出"我是钢铁侠"后牺牲
漫画中的经典故事：
《瓶中恶魔》（Demon in a Bottle）：经典的酗酒问题故事线，展现托尼的人性脆弱
《装甲战争》（Armor Wars）：托尼发现装甲技术被盗，独自追剿所有盗版技术
《内战》（Civil War）：漫画版内战的核心人物，支持注册派，最终导致美国队长"死亡"
《黑暗王朝》与《围城》：成为神盾局局长，应对诺曼·奥斯本的黑暗统治
人物特质与遗产
性格特点：
表面傲慢、自负、爱出风头，被称为"天才、亿万富翁、花花公子、慈善家"
内心深受创伤，有强烈的责任感和赎罪心理
焦虑、创伤后应激障碍（PTSD）患者，尤其在纽约之战后
保护欲极强，将队友尤其是彼得·帕克视为家人
核心主题：
钢铁侠的故事核心是一个赎罪与责任的旅程——从武器商人到和平守护者，从自我中心到为他人牺牲。他的装甲既是保护壳也是牢笼，象征着他在脆弱人性与超级英雄责任之间的挣扎。
托尼·斯塔克证明了英雄主义不在于超能力，而在于选择——选择用天赋保护而非伤害，选择在恐惧中前行，选择为更大的善牺牲自我。他的遗产不仅在于技术，更在于他留下的英雄精神，激励着新一代守护者继续前行。
"""
shandianxia="""
闪电侠（The Flash）是DC宇宙中最具标志性的超级英雄之一，以"极速者"（Speedster）的身份闻名。以下是DC宇宙观下闪电侠的完整介绍：
人物背景与生平
DC漫画历史上有多位闪电侠，最著名的是杰伊·加里克（Jay Garrick）、巴里·艾伦（Barry Allen）和沃利·韦斯特（Wally West）。其中巴里·艾伦是现代闪电侠的代表，也是DC宇宙的核心英雄之一。
巴里·艾伦（Barry Allen）
起源故事：
出生于中心城（Central City），父亲亨利·艾伦是医生，母亲诺拉·艾伦被谋杀
童年时目睹母亲被黄色闪电（逆闪电）杀害，父亲被诬陷为凶手入狱
成为中心城警察局（CCPD）的刑事鉴定科学家，性格内向、守时、热爱科学
一次实验室事故中，被闪电击中并泼洒化学药品，获得连接"神速力"（Speed Force）的能力
关键人生节点：
成为闪电侠
获得超能力后，以闪电侠身份保护中心城，成为"世界上最快的人"
与艾瑞斯·韦斯特（Iris West）相爱并结婚，她是中心城记者，也是他的情感支柱
发现侄子沃利·韦斯特同样被闪电击中获得能力，成为其导师，沃利后来成为第三代闪电侠
危机与牺牲
《无限地球危机》（Crisis on Infinite Earths，1985）：为拯救多元宇宙，巴里以超光速奔跑，摧毁反物质大炮，自身消散于神速力中，成为英雄牺牲的经典时刻
死亡后，沃利·韦斯特继承闪电侠 mantle
重生与回归
《闪电侠：重生》（Flash: Rebirth，2009）：巴里从神速力中回归，发现逆闪电（Eobard Thawne）是杀害母亲的真凶
《闪点》（Flashpoint，2011）：为拯救母亲，巴里穿越时空改变历史，导致末日般的闪点时间线，最终不得不让逆闪电再次杀死母亲以修复时间线
新52与DC重生：在新52重启后，巴里继续作为闪电侠活跃，并发现沃利被困在神速力中，将其救出
家庭与传承
与艾瑞斯结婚后，未来双胞胎子女唐（Don）和道恩（Dawn）成为龙卷风双胞胎（Tornado Twins）
孙子巴特·艾伦（Bart Allen）成为脉冲（Impulse）和第四代闪电侠
核心能力
1. 神速力（Speed Force）连接
神速力是DC宇宙中的超维度能量场，是速度、时间和空间的源头。闪电侠是神速力的"创造者"和主要导管：
基础速度能力：
超高速移动：可超越光速，甚至达到瞬移（teleportation）效果
反应时间：可在纳秒级别感知和反应，看到子弹静止
振动分子：通过高速振动穿过固体物质（穿墙），或使物体分子不稳定而解体
时间旅行：通过宇宙跑步机（Cosmic Treadmill）或达到特定速度穿越时空
维度穿越：进入神速力维度，甚至穿越到平行宇宙
高级神速力技巧：
无限质量拳（Infinite Mass Punch）：以接近光速出拳，产生相当于白矮星质量的冲击力
速度窃取/赋予：从物体或人身上窃取动能使其静止，或将速度赋予他人
创造残影：制造实体速度残影（Speed Mirage），可同时出现在多个地点
思维加速：大脑以超高速运转，快速学习、计算和感知
闪电投射：从身体释放神速力闪电攻击敌人
治愈因子：超速新陈代谢使伤口愈合极快，对毒素和药物免疫（但也导致无法醉酒或正常进食）
2. 科学天才
刑事鉴定科学家背景，擅长证据分析和逻辑推理
神速力思维加速使他能在瞬间完成复杂计算
3. 团队领导能力
正义联盟的创始成员之一
多次担任联盟领导者，以冷静、理性的决策著称
主要事迹与故事线
经典漫画故事：
《闪电侠：重生》（Flash: Rebirth，2009）
巴里从神速力回归，发现逆闪电的阴谋，揭示神速力的真正起源与巴里的关系
《闪点》（Flashpoint，2011）
巴里为救母改变历史，导致托马斯·韦恩成为蝙蝠侠、神奇女侠与海王爆发战争、超人被政府囚禁
最终与托马斯·韦恩合作修复时间线，但造成新52宇宙重启
托马斯留下"别成为我，成为更好的人"的遗言，深刻影响巴里
《闪电侠：前进》（Flash Forward，2019-2020）
沃利·韦斯特成为主角，探索神速力的黑暗面——"静止力"（Still Force）
《无限边际》（Infinite Frontier，2021）
巴里发现多元宇宙的新结构，成为连接所有现实的"闪电侠"
危机事件中的关键角色：
《无限地球危机》（1985）：牺牲自己拯救多元宇宙
《最终危机》（Final Crisis，2008）：巴里与沃利共同击败达克赛德，巴里成为"死亡黑色跑者"（Black Racer）的宿主
《黑暗之夜：金属》（Dark Nights: Metal，2017）：揭示神速力与黑暗多元宇宙的联系，对抗黑暗蝙蝠侠军团
反派与宿敌：
逆闪电（Reverse-Flash / Eobard Thawne）
来自25世纪，崇拜闪电侠的科学家，通过整容成为"巴里·艾伦"
发现自己是巴里的宿敌后精神崩溃，穿越时空不断折磨巴里，包括杀害其母亲
神速力负能量的化身，无法被真正杀死
其他极速者反派：
zoom（亨特·佐罗门）：沃利的宿敌，操纵时间而非速度
神速（Godspeed）：腐败的警察，能同时存在于多个地点
思考者（Thinker）、寒冷队长（Captain Cold）、镜像大师（Mirror Master）等无赖帮（Rogues）成员
人物特质与核心主题
性格特点：
乐观、正直、富有同情心，被称为"希望的象征"
守时、有条理，源于科学家背景和对母亲死亡的执念
内疚感深重，对母亲的死和闪点事件感到自责
将每个人都视为值得拯救的，即使面对逆闪电也试图理解而非单纯憎恨
核心主题：
速度作为隐喻
闪电侠的故事核心是时间与选择——速度赋予他改变过去的能力，但他学会接受无法拯救每个人，真正的英雄主义在于向前奔跑而非沉溺过去。
家庭的纽带
与其他孤独的英雄不同，闪电侠强调家庭的重要性——与艾瑞斯的爱情、与沃利的师徒关系、与正义联盟的兄弟情谊，都是他的力量源泉。
希望的象征
在黑暗的DC宇宙中，闪电侠代表着乐观与希望。绿灯侠哈尔·乔丹曾说："巴里·艾伦是我认识的最善良的人。"即使在最黑暗的时刻，巴里相信人性的光明面。
遗产与传承
闪电侠 mantle的传递（杰伊→巴里→沃利→巴特）象征着英雄精神的延续，每位闪电侠都在前人基础上成长，形成独特的"闪电侠家族"。
闪电侠不仅是最快的男人 alive，更是DC宇宙道德核心的体现——他用速度跨越时空，但用心灵连接每一个人。
"""

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)

docs=[
    Document(page_content=gangtiexia),
    Document(page_content=shandianxia)
]


model_name = "/data/Workspace/airelearn/Day012/python/vllm/cache/BAAI/bge-large-zh-v1___5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
emb_insert_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
emb_insert_model.query_instruction = "为这个句子生成表示以用于检索相关文章："

splits=text_splitter.split_documents(docs)
vector_store=Chroma.from_documents(splits,emb_insert_model)

result=vector_store.similarity_search_with_score("托尼史塔克",k=2)#rag，传入消息
print(len(result))
print(result)

prompt="""
参考以下内容:
{ref}
回答以下问题:
{question}
"""

messages=ChatPromptTemplate([
    ("system","你是一个影评专家，帮助用户了解他们想了解的电影知识"),
    ("human",prompt)
])

#retriever=vector_store.as_retriever().bind(k=2)#转成链
#将函数类型转换为runable
retriever=RunnableLambda(vector_store.similarity_search_with_score).bind(k=2)

chain={"question":RunnablePassthrough(),"ref":retriever} |messages| llm |strOutputParser #langchain矛盾，生成消息需要拼接模版，生成模版需要参考rag，发出rag请求需要消息进行提问

result=chain.invoke("钢铁侠全名叫什么")
print(result)