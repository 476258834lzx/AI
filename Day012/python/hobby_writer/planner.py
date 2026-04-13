from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

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
    happy_end: str = Field(description="皆大欢喜的结局和影响")
    bad_end: str = Field(description="令人哀叹心有不甘的结局和影响")

job_introduction=r"""
您是一位顶级的网络小说作家。
职责是根据任务需求,构建故事情节,围绕情节编写合理大纲,给出多个关键角色和多个剧情支线。
要求:编写4个步骤的情节脉络,包含故事背景铺垫、剧情发展和人物成长、高潮剧情和人物碰撞、故事结尾和影响。
故事背景设定唯一。
拉长时间线,使主角的成长更为丰满。
主线剧情较长、丰富、充满惊险与波折。
多条支线剧情。
穿插于多条支线的关键人物。
剧情支线不断碰撞,产生关键剧情节点。
人物前后人设不能撕裂,关键剧情不能互相冲突,不能战力崩坏,不能出现无效脉络。
输出严格满足格式:{format_instructions}
你可以使用的工具:{tools}。
"""
task=r"""
故事:{content}。
"""

class Planner:
    def __init__(self,llm):
        self.llm=llm

        self.prompt = ChatPromptTemplate.from_messages([
            # 使用 SystemMessagePromptTemplate 包装字符串，并声明 'format_instructions' 变量
            SystemMessagePromptTemplate.from_template(job_introduction),
            # 使用 HumanMessagePromptTemplate 包装字符串，并声明 'content' 变量
            HumanMessagePromptTemplate.from_template(task)
        ])

        self.jsonOutputParser = JsonOutputParser(pydantic_object=Storier)

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
    planner=Planner(llm=llm)
    result=planner({"content":"玄幻龙傲天打脸爽文"})
    print(result)