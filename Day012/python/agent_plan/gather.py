from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import List

gatherer_system_template="""
您是一个优秀的助手，根据参考信息，准确的完成用户查询目标的任务。
只回答与提问相关的答案，不要引入其它信息。
"""


gatherer_user_template="""
参考信息：
{infos}
查询目标：
{query}
"""

class Gatherer:
    def __init__(self,llm):
        prompt=ChatPromptTemplate.from_messages([
            ("system",gatherer_system_template),
            ("human",gatherer_user_template)
        ])

        parser=StrOutputParser()
        
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
    gatherer=Gatherer(llm=llm)
    result=gatherer({"infos":["全红禅是2024年奥运会跳水冠军冠军","她住在四川"],
        "query":"2024年法国巴黎奥运会女子10米跳水冠军的家乡在哪里"})
    print(result)