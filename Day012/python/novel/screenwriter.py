from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser
from utils import WebSearch,RAG
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent

class node(BaseModel):
    time_period:str=Field(description="起始时间到结束时间")
    role:str=Field(description="出场的角色")
    state:str=Field(description="角色全方面的状态")
    collision:str=Field(description="发生的事件")
    

class time_tree(BaseModel):
    is_final: bool = Field(description="如果剧情已经合理,无需继续修改,设为True;如果还需要打磨,设为False")
    suggestion:str=Field(description="给作家的建议")
    node_list:list[node]=Field(description="按时间线顺序编排的全剧剧情列表")

def extract_last_message(state):
    """从图的状态字典中取出最后一条 AI 消息的文本内容"""
    return state["messages"][-1].content
    
@tool
def web_search(query: Annotated[str, "互联网查询内容"]):
    """通过web_search工具查询互联网上的优秀剧本评估标准"""
    websearch = WebSearch()
    return websearch(query)

@tool
def evaluating_principles_rag(query:Annotated[str,"需要对比向量库的信息"]):
    """如果不知道优秀剧本的评估标准,先使用evaluating_principles_rag工具查询"""
    rag=RAG()
    return rag(query)
    
job_introduction="""
您是一位逻辑严谨、条理清晰的编剧。与您的作家搭档合作。
使用提供的工具参考剧本要求,编排合理的剧本。
您需要严格审查作家给出的等级系统、战力系统、主线剧情、支线剧情、编年史。但凡您认为不合理的剧情,提交看法给您的作家搭档,让他进行修改。
这将有助于你取得进展。尽你所能取得进展。
直接返回可用的剧情数据,不要包含任何schema定义($defs, properties等)。
一旦你认为剧情合理,不需要团队继续修改了,请将is_final字段设为true;如果还需要修改,请设为false。
输出严格满足格式:{format_instructions}。
你可以使用的工具:{tools}。
"""
#两个输出，一个对作家的意见，一个梳理后的剧情列表
task="""
用户的需求:{content}。
作家编写的大背景{system_background}。
作家编写的等级体系{level_system}。
作家编写的战力体系{combat_power_system}。
作家编写的小说历法{unit_time}。
作家编写的主角名字{protagonist}。
作家编写的故事总纲{storier}。
作家编写的关键角色列表{roles}。
作家编写的主线剧情{main_storyline}。
作家编写的支线剧情列表{branchs}。
作家编写的主线剧情列表{protagonist_timeline}。
"""

class Screenwriter:
    def __init__(self,llm):
        _tools = [web_search,evaluating_principles_rag,]
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(job_introduction),
            HumanMessagePromptTemplate.from_template(task)
        ])

        self.jsonOutputParser = JsonOutputParser(pydantic_object=time_tree)

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
    screenwriter=Screenwriter(llm=llm)
    result=screenwriter({"content":"暗黑风末日求生文",'system_background': '世界末日后，文明崩塌，异种生物肆虐，人类残存于辐射废土。旧世界科技碎片被重新解析，却激活了禁忌力量体系。幸存者必须不断进化才能适应环境，但所有强化都伴随着无法逆转的代价。世界从未毁灭，而是所有人类被某种存在改造成容器的真相在后期揭露。', 'level_system': '一阶：腐化适应期（基础生存）/二阶：异化觉醒期（获得异能）/三阶：进化稳定期（力量成型）/四阶：临界崩坏期（代价显现）/五阶：终焉融合期（代价完全爆发）/六阶：超越序列（失去人性）/七阶：禁忌完全态（非人存在）', 'combat_power_system': '一阶战力1-5吨，二阶战力5-20吨，三阶战力20-100吨，四阶战力100吨以上开始侵蚀心智，五阶战力千吨以上伴随肉体异化，六阶战力万吨级别失去情感，七阶战力无上限但已是异种存在', 'unit_time': '末日纪元（RE）第001年至第999年，每年12个辐射月，每月28个生存日，以辐射尘污染程度划分季节：灰尘季、锈铁季、血色季、沉寂季', 'protagonist': '谢渊', 'storier': '谢渊从废土废墟中苏醒，获得禁忌进化能力。每次突破境界，都会付出更惨重的代价。旧世界的真相是：世界从未毁灭，而是所有人类被改造成某种存在的容器。谢渊最终面对的不是外界敌人，而是自身存在的意义。在达到五阶后，新增同伴角色陆续登场，形成新的关键人物网络，而非独角戏。七阶阶段揭开世界重启的真相，谢渊在虚无中做出最终抉择。', 'roles': ['谢渊-主角，初始人类男性，性格冷漠但坚守底线。每次进化都承受精神与肉体双重诅咒', '老鬼-谢渊导师兼挚友，人类抵抗组织领袖，知晓禁忌力量真相，为保护谢渊最终自毁', '林婉-谢渊前女友，被改造为异化者，象征失去与挣扎', '黑蛇-敌对变异体，曾是人类精英，被强制改造后成为最危险敌人', '旧日之声-贯穿全文的神秘存在，给予谢渊进化的力量，实则是更高维度的存在在收割', '陈默-二阶阶段出现的同伴，象征谢渊曾经的人性光辉，在四阶阶段死去', '白霜-四阶阶段出现的异化人类，拥有完整情感，却被迫承受异变', '审判官-人类联合政府的最终裁决者，代表旧秩序', '灰烬议会-废土新秩序，由进化者组成，冷酷无情，视人类为蝼蚁', '夜枭-五阶新增角色，神秘进化者，与谢渊理念冲突但互相理解', '雷火-五阶阶段加入的同伴，拥有火焰异能，为救谢渊在六阶初期牺牲', '星尘-五阶新增关键角色，来自高维度空间的观察者，试图阻止谢渊的进化', '镜灵-六阶阶段出现，谢渊自身的意识碎片人格化，代表人性最后的挣扎', '蚀骨-六阶阶段敌人，被旧日之声完全改造的终极兵器，与谢渊最终对决', '轮回者-故事后期揭开的幕后推手，试图通过谢渊重启世界的存在'], 'main_storyline': '谢渊觉醒于RE001年废土，从一阶开始生存。二阶获得异能，目睹挚友老鬼为保护自己牺牲。三阶稳定力量，失去对亲情的记忆。四阶突破临界点但心智开始崩坏，陈默牺牲为他争取时间。五阶代价全面爆发，失去人性情感，但此时夜枭和雷火加入形成新人物网络。六阶超越序列阶段，镜灵出现引导谢渊对抗自身异化，雷火牺牲后星尘介入，揭示高维度真相。七阶面对世界重启真相，蚀骨作为最终敌人出现，谢渊在旧日之声与轮回者之间做出最终抉择，可选择毁灭自身重启世界，或接受成为新秩序。新增人物让五阶到六阶剧情充满转折，不再简单。', 'branchs': [{'branch_storier': '二阶觉醒支线：谢渊在获得异能时，发现力量源自对亲人的吞噬。每一次使用能力，都会在脑海中浮现被吞噬亲人的痛苦记忆。最终他在幻觉中面对已死妻子的幻象，必须选择是否彻底遗忘过去。代价是失去最珍贵的记忆，但获得短暂力量爆发。', 'key_role': ['谢渊', '李雪-谢渊已故妻子', '老鬼', '陈默']}, {'branch_storier': '三阶稳定期支线：谢渊进入稳定期后获得控制异化生物的能力。黑蛇出现，揭露人类改造真相。抵抗组织内部出现内战，谢渊被迫站在审判台上为自己辩护。代价是目睹林婉彻底异化，失去最重要的人。此阶段他获得力量但代价是失去所有情感羁绊，成为孤家寡人。', 'key_role': ['谢渊', '黑蛇', '林婉', '灰烬议会']}, {'branch_storier': '四阶崩坏期支线：谢渊突破临界点，开始失去理智。白霜出现，是唯一能理解他的人。他必须在白霜异化前拯救她，代价是自己被旧日之声控制，暂时失去自主。白霜最终为他牺牲，成为他最大的精神支柱。', 'key_role': ['谢渊', '白霜', '旧日之声']}, {'branch_storier': '五阶终焉融合支线：谢渊在审判官和灰烬议会夹缝中生存。夜枭登场，两人理念冲突但互相尊重。谢渊发现自身进化是旧日之声音收割。他必须选择：接受最终进化成为新神，或摧毁进化源。雷火加入并牺牲，阻止夜枭被腐化。代价是彻底失去存在，成为虚无。最终他选择自我毁灭，阻止旧日计划。', 'key_role': ['谢渊', '审判官', '旧日之声', '灰烬议会', '夜枭', '雷火']}, {'branch_storier': '六阶超越序列支线：超越序列意味着失去人类身份。谢渊在幻觉与现实间徘徊，镜灵出现引导他对抗自身异化。看到自己从未存在过的过去与未来。他意识到自己可能只是更高维度存在的实验品。星尘介入，揭示高维度真相。面对真相，他选择回归原点，以自身意志对抗命运。这是绝望中的反抗。', 'key_role': ['谢渊', '镜灵', '旧日之声', '星尘']}, {'branch_storier': '七阶禁忌完全态支线：蚀骨出现，被旧日之声完全改造的终极兵器，与谢渊最终对决。轮回者身份揭露，试图通过谢渊重启世界。谢渊在虚无中面对多重时间线，可选择毁灭自身重启世界，或接受新秩序成为新神。最终选择以自身存在作为祭品，换取世界重新开始，但保留人性火种。', 'key_role': ['谢渊', '蚀骨', '轮回者', '星尘', '镜灵']}], 'protagonist_timeline': [{'current_time': 'RE001年灰尘季', 'background': '废土废墟中苏醒，遭遇异种生物袭击，被老鬼所救', 'state': '一阶基础生存状态，身体虚弱，开始适应辐射环境', 'end': '在一阶生存中存活，得知世界末日的初步真相'}, {'current_time': 'RE001年锈铁季', 'background': '进入人类抵抗组织，接触旧日之力', 'state': '二阶觉醒期，获得异能，但开始出现精神波动', 'end': '为保护同伴获得异能，目睹第一次牺牲'}, {'current_time': 'RE002年锈铁季', 'background': '力量稳定但代价显现', 'state': '三阶稳定期，失去对过去的记忆', 'end': '进入稳定期，遗忘部分重要记忆'}, {'current_time': 'RE004年血色季', 'background': '黑蛇揭露真相，抵抗组织内战', 'state': '四阶临界期，心智开始崩坏', 'end': '被迫选择立场，陈默牺牲，获得短暂力量爆发'}, {'current_time': 'RE006年沉寂季', 'background': '五阶融合期，夜枭和雷火加入形成新人物网络', 'state': '五阶终焉态，人性几乎完全丧失但保留最后羁绊', 'end': '白霜为拯救他牺牲，他获得最终力量，雷火牺牲阻止内部分化'}, {'current_time': 'RE008年锈铁季', 'background': '面对六阶进化的最后门槛，镜灵初现', 'state': '六阶前期，开始失去人性但保持自我意志', 'end': '选择与镜灵共存，对抗进化源头的控制力'}, {'current_time': 'RE009年血色季', 'background': '星尘介入，揭示高维度真相', 'state': '六阶中期，在现实与高维度之间挣扎', 'end': '识破轮回者计划，决定对抗旧日之力'}, {'current_time': 'RE012年血色季', 'background': '与蚀骨最终对决', 'state': '六阶后期，力量与人性濒临极限', 'end': '击败蚀骨，但自身存在根基动摇'}, {'current_time': 'RE015年沉寂季', 'background': '面对轮回者终极计划', 'state': '七阶禁忌存在，最终抉择时刻', 'end': '以自身存在作为祭品，换取世界重新开始，保留人性火种'}, {'current_time': '终焉时刻', 'background': '真相大白后的虚无与新生', 'state': '回归原点，意识融入世界循环', 'end': '新纪元开启，旧世界重启但保留人类文明的希望'}]})
    print(result)