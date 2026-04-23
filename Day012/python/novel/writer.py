from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import StrOutputParser
from typing import List
        
    
job_introduction="""
您是一位情感细腻辞藻丰富的长篇小说作家。
您需要根据故事背景、关键角色列表、等级系统、战力系统、故事大纲、情节节点、前情结尾等参考信息,详细描写剧情。
要求:自动生成龙套角色。
只写小说正文,不要写任何标题。
前情结尾仅作参考,不要写入正文。
大量篇幅补全角色成长的详细历程,辞藻丰富。
添加转场修辞手法,减少因剧情突变带来的撕裂感。
细致描写心理活动,代入感强。
关键角色人设清晰,名梗不断。
"""

task="""
故事背景:{system_background}。
关键角色列表{roles}。
等级系统:{level_system}。
战力系统:{combat_power_system}。
故事大纲:{storier}。
情节节点:{current_node}。
前情结尾:{previous_end}。
"""

class Writer:
    def __init__(self,llm):
        self.llm=llm
        self.prompt=ChatPromptTemplate.from_messages([
            SystemMessage(job_introduction),
            HumanMessage(task)
        ])
        self.prompt = ChatPromptTemplate.from_messages([
            # 使用 SystemMessagePromptTemplate 包装字符串，并声明 'format_instructions' 变量
            SystemMessagePromptTemplate.from_template(job_introduction),
            # 使用 HumanMessagePromptTemplate 包装字符串，并声明 'content' 变量
            HumanMessagePromptTemplate.from_template(task)
        ])

        self.strOutputParser=StrOutputParser()
        self.chain=self.prompt|self.llm|self.strOutputParser
    
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
    
    writer=Writer(llm=llm)
    result=writer({'system_background': '世界末日后，文明崩塌，异种生物肆虐，人类残存于辐射废土。旧世界科技碎片被重新解析，却激活了禁忌力量体系。幸存者必须不断进化才能适应环境，但所有强化都伴随着无法逆转的代价。世界从未毁灭，而是所有人类被某种存在改造成容器的真相在后期揭露。','roles': ['谢渊-主角，初始人类男性，性格冷漠但坚守底线。每次进化都承受精神与肉体双重诅咒', '老鬼-谢渊导师兼挚友，人类抵抗组织领袖，知晓禁忌力量真相，为保护谢渊最终自毁', '林婉-谢渊前女友，被改造为异化者，象征失去与挣扎', '黑蛇-敌对变异体，曾是人类精英，被强制改造后成为最危险敌人', '旧日之声-贯穿全文的神秘存在，给予谢渊进化的力量，实则是更高维度的存在在收割', '陈默-二阶阶段出现的同伴，象征谢渊曾经的人性光辉，在四阶阶段死去', '白霜-四阶阶段出现的异化人类，拥有完整情感，却被迫承受异变', '审判官-人类联合政府的最终裁决者，代表旧秩序', '灰烬议会-废土新秩序，由进化者组成，冷酷无情，视人类为蝼蚁', '夜枭-五阶新增角色，神秘进化者，与谢渊理念冲突但互相理解', '雷火-五阶阶段加入的同伴，拥有火焰异能，为救谢渊在六阶初期牺牲', '星尘-五阶新增关键角色，来自高维度空间的观察者，试图阻止谢渊的进化', '镜灵-六阶阶段出现，谢渊自身的意识碎片人格化，代表人性最后的挣扎', '蚀骨-六阶阶段敌人，被旧日之声完全改造的终极兵器，与谢渊最终对决', '轮回者-故事后期揭开的幕后推手，试图通过谢渊重启世界的存在'],'level_system': '一阶：腐化适应期（基础生存）/二阶：异化觉醒期（获得异能）/三阶：进化稳定期（力量成型）/四阶：临界崩坏期（代价显现）/五阶：终焉融合期（代价完全爆发）/六阶：超越序列（失去人性）/七阶：禁忌完全态（非人存在）', 'combat_power_system': '一阶战力1-5吨，二阶战力5-20吨，三阶战力20-100吨，四阶战力100吨以上开始侵蚀心智，五阶战力千吨以上伴随肉体异化，六阶战力万吨级别失去情感，七阶战力无上限但已是异种存在','storier': '谢渊从废土废墟中苏醒，获得禁忌进化能力。每次突破境界，都会付出更惨重的代价。旧世界的真相是：世界从未毁灭，而是所有人类被改造成某种存在的容器。谢渊最终面对的不是外界敌人，而是自身存在的意义。在达到五阶后，新增同伴角色陆续登场，形成新的关键人物网络，而非独角戏。七阶阶段揭开世界重启的真相，谢渊在虚无中做出最终抉择。','current_node':{'time_period': 'RE001年锈铁季', 'role': ['谢渊', '李雪(幻象)', '老鬼', '陈默'], 'state': ['二阶异化觉醒期，获得异能，出现吞噬亲人记忆'], 'collision': ['觉醒异能目睹吞噬亲人记忆，获得能力代价']}})
    print(result)