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

from langgraph.prebuilt import ToolNode

@tool
def chatbot_search(query:Annotated[str,"需要查询的实时信息"]):
    """可以使用chatbot_search联网搜索最新互联网信息"""
    web_search=WebSearch()
    return web_search(query)

@tool
def chatbot_rag(query:Annotated[str,"需要对比向量库的信息"]):
    """如果用户查询的信息是关于人物的信息，先使用chatbot_rag工具"""
    rag=RAG()
    return rag(query)

class ChatBot:
    def __init__(self):
        self.tools=[chatbot_search,chatbot_rag]
        self.llm=ChatOpenAI(
            # api_key="",
            openai_api_key="ollama",
            model="qwen2:latest",
            base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
        ).bind_tools(self.tools)
        self.graph=self.init_graph()

    def init_graph(self):
        builder = StateGraph(MessagesState)
        builder.add_node("chat_node", self.chat_agent)
        builder.add_node("tool_node",ToolNode(tools=self.tools))
        
        builder.add_edge(START, "chat_node")
        # builder.add_edge("chat_node", END)#原始不加查询工具边
        builder.add_conditional_edges("chat_node",self.chose_tool_or_end_edge)
        builder.add_edge("tool_node","chat_node")
        
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def chat_agent(self,state):
        # last_message=state.get("messages",[])[-1]
        # result=self.llm.invoke([last_message])
        result=self.llm.invoke(state["messages"])
        return {"messages":[result]}

    def chose_tool_or_end_edge(self,state):
        last_message=state["messages"][-1]
        print(last_message)
        if last_message.tool_calls:
            return "tool_node"
        else:
            return END
    
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