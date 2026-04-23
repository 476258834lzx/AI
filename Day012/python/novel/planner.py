from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from typing import Annotated
from utils import WebSearch
from langchain_core.runnables import RunnableLambda


class timeline_node(BaseModel):
    current_time:str=Field(description="当前时间")
    background:str=Field(description="主角所处的局部背景")
    state:str=Field(description="主角自身的状态")
    end:str=Field(description="事件结局")

class branch(BaseModel):
    branch_storier: str = Field(description="分支剧情的故事总纲")
    key_role:list[str]=Field(description="关键角色列表")

class Storier(BaseModel):
    system_background:str=Field(description="所有角色共同的大背景")
    level_system:str=Field(description="支撑剧情发展详细的等级体系")
    combat_power_system:str=Field(description="不同等级对应的战斗力")
    unit_time:str=Field(description="以故事大背景为基础的独特历法")
    protagonist: str = Field(description="主角名字")
    storier: str = Field(description="全文讲述的故事总纲")
    roles:list[str]=Field(description="尽可能多的关键角色列表")
    main_storyline: str = Field(description="基于故事总纲和关键角色构建的详细对应每个等级的主线剧情")
    branchs:list[branch]=Field(description="尽可能多的支线剧情列表")
    protagonist_timeline: list[timeline_node] = Field(description="按独特历法顺序发展的主线剧情列表")
    
def extract_last_message(state):
    """从图的状态字典中取出最后一条 AI 消息的文本内容"""
    return state["messages"][-1].content

@tool
def web_search(query: Annotated[str, "互联网查询内容"]):
    """通过web_search工具查询互联网上的信息"""
    websearch = WebSearch()
    return websearch(query)

job_introduction="""
您是一位顶级的长篇网络小说作家,与您的编剧助手合作。
如果您没有思路,使用工具,查询小说榜单上的经典作品的桥段和设定。
您需要根据用户的需求,设计故事情节,围绕情节编写合理大纲,给出多个关键角色和多个剧情支线。
严格按照编剧的反馈进行修改,这将有助于你取得进展。尽你所能取得进展。
直接返回可用的剧情数据,不要包含任何schema定义($defs,properties等)。
所有字段必须填写实际内容,不要留空。
输出严格满足格式:{format_instructions}。
您可以访问以下工具：{tools}。
"""
task="""
故事:{content}。
编剧建议:{suggestion}。
上一次编写的大背景{system_background}。
上一次编写的等级体系{level_system}。
上一次编写的战力体系{combat_power_system}。
上一次编写的小说历法{unit_time}。
上一次编写的主角名字{protagonist}。
上一次编写的故事总纲{storier}。
上一次编写的关键角色列表{roles}。
上一次编写的主线剧情{main_storyline}。
上一次编写的支线剧情列表{branchs}。
上一次编写的主线剧情列表{protagonist_timeline}。
"""

class Planner:
    def __init__(self,llm):
        _tools = [web_search,]
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)

        self.prompt=ChatPromptTemplate.from_messages([
            ("system",job_introduction),
            ("human",task)
        ])

        self.jsonOutputParser = JsonOutputParser(pydantic_object=Storier)

        #单步调试
        self.prompt=self.prompt.partial(format_instructions=self.jsonOutputParser.get_format_instructions(),tools=",".join([_tool.name for _tool in _tools]))
        self.chain=self.prompt|_llm_with_tools_agent|RunnableLambda(extract_last_message)|self.jsonOutputParser

    def __call__(self,state):
        #单步调试
        return self.chain.invoke(state)

if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        # api_key="",
        openai_api_key="ollama",
        model="qwen3.5:latest",
        base_url="http://127.0.0.1:11434/v1"  # ollamaserve的地址/v1 协议版本
    )
    planner=Planner(llm=llm)
    result=planner({"content":"暗黑风末日求生文","suggestion":"主角达到五阶之后关键人物全部退场,成为主角的独角戏,剧情过于单一,需铺垫添加另外的关键角色,主角从五阶到六阶的剧情过于简单,需添加剧情设定",'system_background': '世界末日后，文明崩塌，异种生物肆虐，人类残存于辐射废土。旧世界科技碎片被重新解析，却激活了禁忌力量体系。幸存者必须不断进化才能适应环境，但所有强化都伴随着无法逆转的代价。', 'level_system': '一阶：腐化适应期（基础生存）/二阶：异化觉醒期（获得异能）/三阶：进化稳定期（力量成型）/四阶：临界崩坏期（代价显现）/五阶：终焉融合期（代价完全爆发）/六阶：超越序列（失去人性）/七阶：禁忌完全态（非人存在）', 'combat_power_system': '一阶战力1-5吨，二阶战力5-20吨，三阶战力20-100吨，四阶战力100吨以上开始侵蚀心智，五阶战力千吨以上伴随肉体异化，六阶战力万吨级别失去情感，七阶战力无上限但已是异种存在', 'unit_time': '末日纪元（RE）第001年至第999年，每年12个辐射月，每月28个生存日，以辐射尘污染程度划分季节：灰尘季、锈铁季、血色季、沉寂季', 'protagonist': '谢渊', 'storier': '谢渊从废土废墟中苏醒，获得禁忌进化能力。每次突破境界，都会付出更惨重的代价：亲情、人性、记忆、肉体。他没有超能力，只有不断累积的诅咒。旧世界的真相是：世界从未毁灭，而是所有人类被改造成了某种存在的容器。谢渊最终要面对的不是外界敌人，而是自身存在的意义。', 'roles': ['谢渊-主角，初始人类男性，性格冷漠但坚守底线。每次进化都承受精神与肉体双重诅咒', '老鬼-谢渊导师兼挚友，人类抵抗组织领袖，知晓禁忌力量真相，为保护谢渊最终自毁', '林婉-谢渊前女友，被改造为异化者，最终成为谢渊最大的痛苦来源，象征失去与挣扎', '黑蛇-敌对变异体，曾是人类精英，被强制改造后成为最危险敌人', '旧日之声-贯穿全文的神秘存在，给予谢渊进化的力量，实则是更高维度的存在在收割', '陈默-二阶阶段出现的同伴，象征谢渊曾经的人性光辉，在四阶阶段死去', '白霜-四阶阶段出现的异化人类，拥有完整情感，却被迫承受异变', '审判官-人类联合政府的最终裁决者，代表旧秩序', '灰烬议会-废土新秩序，由进化者组成，冷酷无情，视人类为蝼蚁'], 'main_storyline': '谢渊觉醒于RE001年废土，从一阶开始生存。二阶获得异能，目睹挚友老鬼为保护自己牺牲。三阶稳定力量，失去对亲情的记忆。四阶突破临界点但心智开始崩坏，陈默牺牲为他争取时间。五阶代价全面爆发，失去人性情感，被旧日之声控制。六阶超越序列，彻底失去人类身份。七阶最终面对真相与自我存在意义，可选择毁灭世界或成为新神，最终选择毁灭自身以保护残余人类文明。', 'branchs': [{'branch_storier': '二阶觉醒支线：谢渊在获得异能时，发现力量源自对亲人的吞噬。每一次使用能力，都会在脑海中浮现被吞噬亲人的痛苦记忆。最终他在幻觉中面对已死妻子的幻象，必须选择是否彻底遗忘过去。代价是失去最珍贵的记忆，但获得短暂力量爆发。', 'key_role': ['谢渊', '李雪-谢渊已故妻子', '老鬼', '陈默']}, {'branch_storier': '三阶稳定期支线：谢渊进入稳定期后获得控制异化生物的能力。黑蛇出现，揭露人类改造真相。抵抗组织内部出现内战，谢渊被迫站在审判台上为自己辩护。代价是目睹林婉彻底异化，失去最重要的人。此阶段他获得力量但代价是失去所有情感羁绊，成为孤家寡人。', 'key_role': ['谢渊', '黑蛇', '林婉', '灰烬议会']}, {'branch_storier': '四阶崩坏期支线：谢渊突破临界点，开始失去理智。白霜出现，是唯一能理解他的人。他必须在白霜异化前拯救她，代价是自己被旧日之声控制，暂时失去自主。白霜最终为他牺牲，成为他最大的精神支柱。', 'key_role': ['谢渊', '白霜', '旧日之声']}, {'branch_storier': '五阶终焉支线：谢渊在审判官和灰烬议会夹缝中生存。发现自身进化是旧日之声音收割。他必须选择：接受最终进化成为新神，或摧毁进化源。代价是彻底失去存在，成为虚无。最终他选择自我毁灭，阻止旧日计划。', 'key_role': ['谢渊', '审判官', '旧日之声', '灰烬议会']}, {'branch_storier': '六阶超越支线：超越序列意味着失去人类身份。谢渊在幻觉与现实间徘徊，看到自己从未存在过的过去与未来。他意识到自己可能只是更高维度存在的实验品。面对真相，他选择回归原点，以自身意志对抗命运。这是绝望中的反抗。', 'key_role': ['谢渊', '幻觉中的谢渊', '旧日之声']}], 'protagonist_timeline': [{'current_time': 'RE001年灰尘季', 'background': '废土废墟中苏醒，遭遇异种生物袭击，被老鬼所救', 'state': '一阶基础生存状态，身体虚弱，开始适应辐射环境', 'end': '在一阶生存中存活，得知世界末日的初步真相'}, {'current_time': 'RE001年锈铁季', 'background': '进入人类抵抗组织，接触旧日之力', 'state': '二阶觉醒期，获得异能，但开始出现精神波动', 'end': '为保护同伴获得异能，目睹第一次牺牲'}, {'current_time': 'RE002年锈铁季', 'background': '力量稳定但代价显现', 'state': '三阶稳定期，失去对过去的记忆', 'end': '进入稳定期，遗忘部分重要记忆'}, {'current_time': 'RE004年血色季', 'background': '黑蛇揭露真相，抵抗组织内战', 'state': '四阶临界期，心智开始崩坏', 'end': '被迫选择立场，陈默牺牲，获得短暂力量爆发'}, {'current_time': 'RE006年沉寂季', 'background': '与旧日之声对抗', 'state': '五阶终焉态，人性几乎完全丧失', 'end': '白霜为拯救他牺牲，他获得最终力量'}, {'current_time': 'RE007年灰尘季', 'background': '面对最终抉择', 'state': '六阶超越序列，失去人类身份', 'end': '选择自我毁灭，阻止旧日计划，保护人类文明'}, {'current_time': '终焉时刻', 'background': '真相大白后的虚无', 'state': '七阶禁忌存在，最终消散', 'end': '以自身存在作为祭品，换取世界重新开始'}]})
    print(result)