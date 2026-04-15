from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END,MessagesState

_llm = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen3.5:latest",
    api_key="ollama")

from researcher import Researcher
from painter import Painter

researcher=Researcher(_llm)
painter=Painter(_llm)

def researcher_node(state):
    return {"messages":[researcher(state)]}

def painter_node(state):
    return {"messages":[painter(state)]}

def router(state):
    last_message=state["messages"][-1]
    if "FINAL ANSWER" in last_message.content.upper():
        return END
    
    return "continue"

builder=StateGraph(MessagesState)

builder.add_node("researcher_node",researcher_node)
builder.add_node("painter_node",painter_node)

builder.add_edge(START,"researcher_node")
builder.add_conditional_edges("researcher_node",router,{"continue":"painter_node",END:END})
builder.add_conditional_edges("painter_node",router,{"continue":"researcher_node",END:END})

graph=builder.compile()

for events in graph.stream({"messages":[("human","去互联网上查询英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，保存为本地文件GDP.jpg，就完成。")]}):
    print(events)
    # for message in events["messages"]:
    #     print(message)