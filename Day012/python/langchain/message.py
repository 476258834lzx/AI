from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)
strOutputParser = StrOutputParser()

messages=[
    ("system","你是一个翻译官,你需要将用户的语言翻译成英文"),
    # ("user","我爱北京天安门")
    ("human","我爱北京天安门")
]

#消息体
messages=[
    SystemMessage("你是一个翻译官,你需要将用户的语言翻译成英文"),
    HumanMessage("我爱北京天安门")
]

result=llm.invoke(messages)
print(result)

#输出解析器
result=strOutputParser.invoke(result)
print(result)   

#管道输出
chain=llm | strOutputParser
result=chain.invoke(messages)
print(result)

#流式输出(同步流,异步流)
for chunk in chain.stream(messages):
    print(chunk,end="")
