from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

class Sub_Storier(BaseModel):
    sub_storier:str=Field(description="按情节节点详细描写的剧情")
        
    
job_introduction=r"""
您是一位情感细腻辞藻丰富的情景写手。
职责是根据故事背景、等级系统、战力系统、故事总纲、时间度量、情节列表详细描写剧情。
要求:自动生成龙套角色制造冲突。
细致的心理活动描写,代入感强。
穿插角色第一视角和旁白第三视角进行描写。
关键角色各有特色,名梗不断。
关键角色与主角并行成长,不能让关键角色成为静态NPC。
一个镜头大于300字描写,一条剧情支线大于2000字描写。
输出:{format_instructions}。
你可以使用的工具:{tools}。
"""

task=r"""
故事背景:{system_background}。
等级系统:{level_system}。
战力系统:{combat_power_system}。
故事总纲:{storier}。
时间度量:{unit_time}。
情节节点:{node}。
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

        self.jsonOutputParser = JsonOutputParser(pydantic_object=Sub_Storier)

        #单步调试
        self.prompt=self.prompt.partial(format_instructions=self.jsonOutputParser.get_format_instructions(),tools="")
        self.chain=self.prompt|self.llm|self.jsonOutputParser
    
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
    result=writer({"system_background":"古界域崩塌后，灵气复苏形成三大修真界，万宗林立，强者为尊，资源匮乏，弱者如草芥。界域内有九大天域，各天域由顶级宗门统治，凡人如蝼蚁。","level_system":"炼气初期可御物，炼气后期可御气;筑基可破石;金丹可御器;元婴可御阵;化神可御城;炼虚可御国;合体可渡劫;大乘近乎不死;渡劫飞升成仙。","combat_power_system":"炼气初期战力10-50，炼气后期50-200;筑基初期200-500，筑基后期500-1000;金丹初期1000-3000，金丹后期3000-5000;元婴初期5000-8000，元婴后期8000-12000;化神初期12000-20000，化神后期20000-30000;炼虚初期30000-50000，炼虚后期50000-100000;合体初期100000-200000，合体后期200000-500000;大乘境界500000-100万;渡劫飞升100万以上。","unit_time":"修真历，每百年为一个甲子轮回，当前为修真历三百一十六年","storier":"陆风从废柴少年开始，被未婚妻林冰瑶当众羞辱退婚，获得神秘传承系统，修为一日千里，加入三大天域试炼，历经九九八十一难，最终成就无上道果。主线围绕打脸羞辱、宗门大比、秘境探索、仇家复仇多线并行，感情线多位女性角色情感纠葛，最终建立自己的仙朝帝国，超脱界域。","node":{'time_period': '修真历三百二十一年九月至三百二十五年三月之后', 'role': '陆风，叶轻云，血修罗，南宫雪', 'background': '三大天域争霸，上古遗迹探索获机缘，飞升前夕机缘，叶轻云转世为道脉引路人', 'state': '元婴后期渡化神，战力超越五百，建立道门势力，最终超脱界限，准备飞升成仙，叶轻云元神归位', 'collision': '在界域之巅与血修罗决战确立霸权，叶轻云元神转世延续传承，陆风建立仙朝帝国，飞升前夕叶轻云旧魂以引路者身份与陆风作别，完成终极使命'}})
    print(result)