from operator import add
from typing import Annotated
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from IPython.display  import display,Image

class State(TypedDict):
    # foo:float
    # foo:Annotated[list[float],add]
    foo:Annotated[list[float],lambda x,y:x+y]
    
def agent1(state) :
    new_state={"foo":state["foo"]}#可以返回一部分，只更新部分的状态
    # new_state={"foo":[state["foo"][0]+1]}
    return new_state

builder=StateGraph(State)

builder.add_node("node_01",agent1)
builder.add_edge(START,"node_01")
builder.add_edge("node_01",END)

graph=builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

result=graph.invoke({"foo":[20.0]})
print(result)