from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

llm=ChatOpenAI(
    # api_key="",
    openai_api_key="ollama",
    model="qwen2:latest",
    base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
)

messages=ChatPromptTemplate([
    ("system","你是一个家庭管家,请根据房间主人的日程提供建议和帮助"),
    MessagesPlaceholder(variable_name="history_contents"),
    ("human","{content}")
])

strOutputParser = StrOutputParser()
chain=messages|llm | strOutputParser
store={}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id]=ChatMessageHistory()  
    return store[session_id]

history_chain=RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="content",
    history_messages_key="history_contents"        
)

result1=history_chain.invoke({"content":"我今天要以'老舍'的笔名参加诗词大会，请叫我'老舍'"},config={"configurable":{"session_id":1}})
print(result1)

result2=history_chain.invoke({"content":"你还记得今天我的'角色'是什么吗"},config={"configurable":{"session_id":1}})
print(result2)