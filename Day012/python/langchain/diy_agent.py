from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)
strOutputParser = StrOutputParser()
messages=[
    SystemMessage("你是一个家庭管家,请根据房间主人的日程提供建议和帮助"),
    HumanMessage("呼和浩特今日的天气如何")
    ]

@tool
def get_weather(city:Annotated[str,"被查询的城市,所有城市名均为汉字字符"]):
    """"
    用于查询输入城市今日的天气状况
    """
    if city=="上海":
        return "台风12级"
    else:
        return "特强沙尘暴"

tools=[get_weather,]
agent=llm.bind_tools(tools)
result=agent.invoke(messages)
print(result)
print(result.tool_calls)

tool_res=""#如何指定为函数的返回类型
if result.tool_calls is not None or len(result.tool_calls)>0:
    messages.append(result)
    for tool in result.tool_calls:
        func=eval(tool["name"])
        tool_res=func.invoke(tool["args"])
        messages.append(tool_res)
print(tool_res)

result=llm.invoke(messages)
print(result)