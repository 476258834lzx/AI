from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from IPython.display import display, Image
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    foo: float


def agent1(state,config):
    print(config.get("configurable",{}).get("a"))
    new_state = {"foo": state["foo"] + 1}  # 可以返回一部分，只更新部分的状态
    return new_state


builder = StateGraph(State)

builder.add_node("node_01", agent1)
builder.add_edge(START, "node_01")
builder.add_edge("node_01", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)#按照thread_id存储历史

display(Image(graph.get_graph().draw_mermaid_png()))

configuration={"thread_id":"1","recursion_limit":2,"configurable":{"a":1}}#langgraph中不需要使用session_id，常使用thread_id

result = graph.invoke({"foo": 20.0},config=configuration)
print(result)