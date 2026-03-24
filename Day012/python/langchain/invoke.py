from langchain_openai import ChatOpenAI

llm=ChatOpenAI(
    # api_key="",
    apikey="ollama",
    model="qwen2.5:7b",
    base_url=""#ollamaserve的地址/v1 协议版本
)

llm.invoke("你是谁")