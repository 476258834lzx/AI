from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

class node(BaseModel):
    time_period:str=Field(description="起始时间到结束时间")
    role:str=Field(description="角色")
    personality: str = Field(description="角色性格")
    background:str=Field(description="角色所处的局部背景")
    state:str=Field(description="角色自身的状态")
    collision:str=Field(description="与主线剧情碰撞的关键剧情")
    

class time_tree(BaseModel):
    node_list:list[node]=Field(description="按时间线顺序编排的全剧剧情列表")
    
    
job_introduction=r"""
您是一位逻辑严谨承上启下的编剧。
职责是根据故事背景、等级系统、战力系统、主线剧情、时间度量、剧情分支、主线剧情时间线,合理编排剧情,构造时间线分叉树。
要求:详细展开支线剧情,编写至少3个步骤的情节脉络,包含分支背景铺垫、剧情发展和关键人物成长、和主线剧情的碰撞。
剧情跌宕起伏,充满博弈,多分之间使用蒙太奇手法。
输出严格满足格式:{format_instructions}。
你可以使用的工具:{tools}。
"""

task=r"""
故事背景:{system_background}。
等级系统:{level_system}。
战力系统:{combat_power_system}。
时间度量:{unit_time}。
主线剧情:{main_storyline}。
剧情分支:{branchs}。
主线剧情时间线:{protagonist_timeline}。
好结局:{happy_end}。
坏结局:{bad_end}。
"""

class Screenwriter:
    def __init__(self,llm):
        self.llm=llm

        self.prompt = ChatPromptTemplate.from_messages([
            # 使用 SystemMessagePromptTemplate 包装字符串，并声明 'format_instructions' 变量
            SystemMessagePromptTemplate.from_template(job_introduction),
            # 使用 HumanMessagePromptTemplate 包装字符串，并声明 'content' 变量
            HumanMessagePromptTemplate.from_template(task)
        ])

        self.jsonOutputParser = JsonOutputParser(pydantic_object=time_tree)

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
    screenwriter=Screenwriter(llm=llm)
    result=screenwriter({"system_background":"古界域崩塌后，灵气复苏形成三大修真界，万宗林立，强者为尊，资源匮乏，弱者如草芥。界域内有九大天域，各天域由顶级宗门统治，凡人如蝼蚁。","level_system":"炼气初期可御物，炼气后期可御气;筑基可破石;金丹可御器;元婴可御阵;化神可御城;炼虚可御国;合体可渡劫;大乘近乎不死;渡劫飞升成仙。","combat_power_system":"炼气初期战力10-50，炼气后期50-200;筑基初期200-500，筑基后期500-1000;金丹初期1000-3000，金丹后期3000-5000;元婴初期5000-8000，元婴后期8000-12000;化神初期12000-20000，化神后期20000-30000;炼虚初期30000-50000，炼虚后期50000-100000;合体初期100000-200000，合体后期200000-500000;大乘境界500000-100万;渡劫飞升100万以上。","unit_time":"修真历，每百年为一个甲子轮回，当前为修真历三百一十六年","main_storyline":"陆风从废柴少年开始，被未婚妻林冰瑶当众羞辱退婚，获得神秘传承系统，修为一日千里，加入三大天域试炼，历经九九八十一难，最终成就无上道果。主线围绕打脸羞辱、宗门大比、秘境探索、仇家复仇多线并行，感情线多位女性角色情感纠葛，最终建立自己的仙朝帝国，超脱界域。","branchs":[{'storier': '林冰瑶支线：林冰瑶为青云宗二小姐，从小与陆风青梅竹马。后因被家族逼迫与陆风退婚，心生怨恨。三年后与家族联姻，成就筑基。十年后陆风崛起，她悔恨并多次试探，陆风淡然处之。二十年后，她为家族利益背叛陆风，被揭露后自尽。', 'key_role': ['陆风', '林冰瑶', '林老爷', '青云宗长老']}, {'storier': '叶轻云支线：叶轻云为神秘老者，实为陆风师尊。传授陆风修仙之道，后揭示自己乃上古剑仙转世。十年后为保护陆风不惜重伤陨落。二十年后转世于陆风道脉中，作为引路人持续辅助陆风成长。', 'key_role': ['陆风', '叶轻云', '南宫雪', '墨尘']}, {'storier': '苏红袖支线：苏红袖为落云门掌门之女，性格洒脱，与陆风在秘境中相遇，成为知己。陆风为她挡下追杀，两人产生情愫。后苏红袖家族背叛，陆风出手相救，被世人称颂。', 'key_role': ['陆风', '苏红袖', '落云门、雷震天', '夜千魂']}, {'storier': '萧炎支线：萧炎为邻宗天才少年，与陆风多次竞争又合作。初期被陆风头压，心怀不甘，后在秘境中领悟大道，与陆风联手对抗强敌。最终在飞升大典上与陆风结拜兄弟。', 'key_role': ['陆风', '萧炎', '南宫雪', '云仙子']}, {'storier': '三大天域争霸支线：九大天域争夺界域霸权，陆风从青云宗杂役开始，逐步融入宗门大比，参与秘境试炼，对抗天域霸主，建立势力。在三大天域争霸中，陆风创立道门，成为超脱者。', 'key_role': ['陆风', '赵无极', '风无痕', '血修罗', '南宫雪']}, {'storier': '上古遗迹探索支线：陆风进入上古遗迹，获得机缘，与各方势力争夺。在遗迹中揭露上古秘辛，获得传承，提升修为。在探索中多次与林冰瑶家族、叶轻云徒弟等角色发生剧情碰撞。', 'key_role': ['陆风', '墨尘', '风无痕', '夜千魂']}, {'storier': '飞升仙界支线：飞升前夕，陆风获得渡劫机缘。飞升后建立仙朝，与仙界势力周旋。最终超脱一切，成为界域守护神。', 'key_role': ['陆风', '苏红袖', '夜千魂', '风无痕']}],"protagonist_timeline":[{'current_time': '修真历三百一十六年三月', 'background': '青云宗杂役处', 'state': '废柴被羞辱，修为炼气一层', 'end': '获得神秘传承，被宗门驱逐'}, {'current_time': '修真历三百一十七年七月', 'background': '青云城外荒郊', 'state': '修为筑基初期，开始重建道心', 'end': '在荒野中领悟基础功法，修为精进'}, {'current_time': '修真历三百一十九年正月', 'background': '青云宗外门', 'state': '修为金丹初期，参加宗门大比', 'end': '夺得外门第一，成为内门弟子'}, {'current_time': '修真历三百二十一年九月', 'background': '秘境试炼之地', 'state': '修为元婴初期，遭遇上古凶兽', 'end': '击杀凶兽，获得上古传承'}, {'current_time': '修真历三百二十五年三月', 'background': '青云宗', 'state': '修为化神初期，被赵无极围攻', 'end': '以弱胜强，击杀赵无极'}]})
    print(result)