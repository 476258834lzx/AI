from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent#未来版本create_react_agent更新为

@tool
def get_weather(city:Annotated[str,"被查询的城市,所有城市名均为汉字字符"]):
    """"
    用于查询输入城市今日的天气状况
    """
    if city=="上海":
        return "台风12级"
    else:
        return "特强沙尘暴"
    
llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)
strOutputParser = StrOutputParser()

agent=create_react_agent(llm,[get_weather,])
result=agent.invoke({"messages":[SystemMessage("你是一个家庭管家,请根据房间主人的日程提供建议和帮助"),HumanMessage("呼和浩特今日的天气如何")]})
print(result)
print(result["messages"][-1].content)