from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
from typing import List

class Plan(BaseModel):
    """制定的计划"""
    task_list:List[str]=Field(description="子任务列表")

planner_system_template="""
您是一位优秀的战略规划家。
您具备卓越的任务规划能力,职责是将精心构思一套详尽的解题策略,确保在应对用户查询时，规划出一套解题步骤,最终呈现出一个逻辑严密。
输出:{output_format}
"""


planner_user_template="""
查询目标:{query}。
"""

class Planner:
    def __init__(self,llm):
        prompt=ChatPromptTemplate.from_messages([
            ("system",planner_system_template),
            ("human",planner_user_template)
        ])

        parser=JsonOutputParser(pydantic_object=Plan)
        prompt=prompt.partial(output_format=parser.get_format_instructions())
        
        self.chain=prompt|llm|parser
        
    def __call__(self,state):
        return self.chain.invoke(state)
    
if __name__=="__main__":
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        # api_key="",
        openai_api_key="ollama",
        model="qwen3.5:latest",
        base_url="http://127.0.0.1:11434/v1"  # ollamaserve的地址/v1 协议版本
    )
    planner=Planner(llm=llm)
    result=planner({"query":"2024年法国巴黎奥运会女子10米跳水冠军的父亲是谁"})
    print(result)