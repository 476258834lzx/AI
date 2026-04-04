from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import MessagesState#也继承TypedDict,不使用MessageGraphE(无法传递配置，无法新增自定义消息)
from operator import add
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.types import Send

class State(TypedDict):
    foo: Annotated[list[str],add]
    f:str

def agent_01(state):
    return {"foo":[state["f"]]}

def dispath(state):
    return [Send("node_01",{"f":str(x)})for x in range(10)]

builder=StateGraph(State)
builder.add_node("node_01",agent_01)
builder.add_conditional_edges(START,dispath)
builder.add_edge("node_01",END)

graph=builder.compile()

result=graph.invoke({"foo":[],"f":""})
print(result)