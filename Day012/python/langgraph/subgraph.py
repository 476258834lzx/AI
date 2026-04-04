from typing import Annotated
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import display, Image
from operator import add

class State(TypedDict):
    foo: Annotated[list[str],add]


def agent_m1(State):
    return {"foo":["m1"]}

def agent_m2(State):
    return {"foo":["m2"]}

def agent_l1(State):
    return {"foo":["l1"]}

def agent_l2(State):
    return {"foo":["l2"]}

def agent_r1(State):
    return {"foo":["r1"]}

sub_graph=StateGraph(State)
sub_graph.add_node("node_l1",agent_l1)
sub_graph.add_node("node_l2",agent_l2)
sub_graph.add_edge(START,"node_l1")
sub_graph.add_edge("node_l1","node_l2")
sub_graph.add_edge("node_l2",END)

builder = StateGraph(State)

builder.add_node("node_m1", agent_m1)
builder.add_node("node_m2", agent_m2)
builder.add_node("node_r1", agent_r1)
builder.add_node("sub_graph",sub_graph.compile())

builder.add_edge(START, "node_m1")
builder.add_edge("node_m1", "sub_graph")
builder.add_edge("node_m1", "node_r1")
builder.add_edge("sub_graph", "node_m2")
builder.add_edge("node_r1", "node_m2")
builder.add_edge("node_m2", END)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

result = graph.invoke({"foo": ["graph"]})
print(result)