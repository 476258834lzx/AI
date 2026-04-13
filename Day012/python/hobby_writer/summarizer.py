from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

class Complete_written_text(BaseModel):
    complete_written_text:str=Field(description="按情节节点详细描写的剧情")

job_introduction=r"""
您是一位条理清晰的总结写手。
职责是按照故事背景、等级系统、战力系统、故事总纲、主线剧情、剧情分支、主线剧情时间线、全剧剧情列表、详细剧情列表、好结局、坏结局,总结剧情进行润色写出最后成文。
要求:按每章节1500字切分,加上章节序号。
在多个剧情节点添加转场修辞手法,减少因剧情突变带来的撕裂感。
同时写出两个大结局。
输出:{format_instructions}
你可以使用的工具:{tools}。
"""

task=r"""
故事背景:{system_background}。
等级系统:{level_system}。
战力系统:{combat_power_system}。
故事总纲:{storier}。
主线剧情:{main_storyline}。
剧情分支:{branchs}。
主线剧情时间线:{protagonist_timeline}。
全剧剧情列表:{node_list}。
详细剧情列表:{sub_storier_list}。
好结局:{happy_end}。
坏结局:{bad_end}。
"""

class Summarizer:
    def __init__(self,llm):
        self.llm=llm

        self.prompt = ChatPromptTemplate.from_messages([
            # 使用 SystemMessagePromptTemplate 包装字符串，并声明 'format_instructions' 变量
            SystemMessagePromptTemplate.from_template(job_introduction),
            # 使用 HumanMessagePromptTemplate 包装字符串，并声明 'content' 变量
            HumanMessagePromptTemplate.from_template(task)
        ])

        self.jsonOutputParser = JsonOutputParser(pydantic_object=Complete_written_text)

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
    summarizer=Summarizer(llm=llm)
    result=summarizer({})
    print(result.complete_written_text)