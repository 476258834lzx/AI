from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from IPython.display  import display,Image

class State(TypedDict):
    foo:float
    
def agent1(state) :
    new_state={"foo":state["foo"]+1}#可以返回一部分，只更新部分的状态
    return new_state

builder=StateGraph(State)

builder.add_node("node_01",agent1)
builder.add_edge(START,"node_01")
builder.add_edge("node_01",END)

graph=builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

result=graph.invoke({"foo":20.0})
print(result)