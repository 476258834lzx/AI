from langchain_openai import ChatOpenAI

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen3.5:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)

result=llm.invoke("你是谁")
print(result)