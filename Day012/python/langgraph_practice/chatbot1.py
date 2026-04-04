from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
from langchain_core.tools import tool
from tools import WebSearch,RAG

from langgraph.prebuilt import ToolNode,create_react_agent

@tool
def chatbot_search(query:Annotated[str,"需要查询的实时信息"]):
    """需要获取互联网最新信息时可以使用chatbot_search联网搜索最新互联网信息"""
    web_search=WebSearch()
    return web_search(query)

@tool
def chatbot_rag(query:Annotated[str,"需要对比向量库的信息"]):
    """如果用户查询的信息是关于人物的信息，先使用chatbot_rag工具"""
    rag=RAG()
    return rag(query)

class ChatBot:
    def __init__(self):
        self.tools=[chatbot_search,chatbot_rag,]
        self.llm=ChatOpenAI(
            # api_key="",
            openai_api_key="ollama",
            model="qwen2:latest",
            base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
        )
        self.graph=self.init_graph()

    def init_graph(self):
        builder = StateGraph(MessagesState)
        agent=create_react_agent(self.llm,[chatbot_search,])
        
        builder.add_node("chat_node", agent)
        
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)#原始不加查询工具边
        
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    
    def __call__(self):
        thread_id=uuid.uuid4
        while True:
            human=input("我:")
            if human=="bye":
                print("AI:速度滚")
                break
            config={"thread_id":thread_id}
            reply=self.graph.invoke({"messages":[HumanMessage(human)]},config=config)
            print(f"AI:{reply['messages'][-1].content}")

if __name__ == '__main__':
    chatbot=ChatBot()
    chatbot()