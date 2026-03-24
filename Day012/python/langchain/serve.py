from langchain_openai import ChatOpenAI

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import  ChatOpenAI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

_llm = ChatOpenAI(
    base_url="http://192.168.10.13:60001/v1",#ollamaserve的地址/v1 协议版本
    model="qwen2.5:7b",
    api_key="ollama",
    temperature=0.7
)

_template = ChatPromptTemplate([
    ("system","翻译以下的内容为{language}。"),
    ("human","{content}")
])

_strOutputParser = StrOutputParser()
_chain = _template|_llm|_strOutputParser

app = FastAPI(
    title="翻译",
    version="1.0",
    description="大模型智能翻译",
)

add_routes(
    app,
    _chain,
    path="/translate",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=60002)
