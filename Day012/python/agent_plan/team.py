from planner import Planner
from executor import Executor
from gather import Gatherer
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from typing_extensions import TypedDict
from operator import add
from typing import List,Annotated

class PlanState(TypedDict):
    query:str
    task_list:List[str]
    infos:Annotated[List[str],add]
    result:str

_llm = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen3.5:latest",
    api_key="ollama")

planner=Planner(_llm)
executor=Executor(_llm)
gatherer=Gatherer(_llm)

def planner_node(state):
    return planner(state)

def executor_node(state):
    task_list=state["task_list"]
    infos=state.get("infos",[])
    task_index=len(infos)
    task=task_list[task_index]
    rt=executor({
        "infos": infos,
        "task": task
    })
    return {"infos":[rt]}
    
    

def gatherer_node(state):
    return {"result":gatherer(state)}

def router(state):
    task_len=len(state["task_list"])
    infos_len=len(state["infos"])
    if task_len==infos_len:
        return "gatherer_node"
    else:
        return "executor_node"

builder=StateGraph(PlanState)

builder.add_node("planner_node",planner_node)
builder.add_node("executor_node",executor_node)
builder.add_node("gatherer_node",gatherer_node)

builder.add_edge(START,"planner_node")
builder.add_edge("planner_node","executor_node")
builder.add_conditional_edges("executor_node",router,["gatherer_node","executor_node"])
# builder.add_conditional_edges("executor_node",router,{"gatherer_node":"gatherer_node","executor_node":"executor_node"})
builder.add_edge("gatherer_node",END)

graph=builder.compile()
png_bytes = graph.get_graph().draw_mermaid_png()
with open("my_graph.png", "wb") as f:
    f.write(png_bytes)

for events in graph.stream({"query":"2024年法国巴黎奥运会女子10米跳水冠军的父亲是谁"}):
    print(events)

# result=graph.invoke({"query":"2024年法国巴黎奥运会女子10米跳水冠军的父亲是谁"})
# print(result)