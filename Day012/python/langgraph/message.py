from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import MessagesState#也继承TypedDict,不使用MessageGraphE(无法传递配置，无法新增自定义消息)
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage

class MyMessages(MessagesState):
    foo:int

def agent1(state) :
    return {"messages":[HumanMessage("world")],"foo":2}

builder=StateGraph(MyMessages)

builder.add_node("node_01",agent1)
builder.add_edge(START,"node_01")
builder.add_edge("node_01",END)
graph=builder.compile()

result=graph.invoke({"messages":[HumanMessage("hello")],"foo":0})
print(result)