from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)
strOutputParser = StrOutputParser()

messages=ChatPromptTemplate([
    ("system","翻译以下的内容为{language}。"),
    ("human","{content}")
])

messagewithtmp=messages.invoke({"language":"法语","content":"我爱北京天安门"})
result=llm.invoke(messagewithtmp)
print(result)

#管道输出
chain=messages |llm | strOutputParser
result=chain.invoke({"language":"法语","content":"我爱北京天安门"})
print(result)