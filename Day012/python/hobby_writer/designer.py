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
from langgraph.types import Send
from operator import add
# from tools import WebSearch,RAG

from langgraph.prebuilt import ToolNode,create_react_agent
from planner import Planner
from screenwriter import Screenwriter
from writer import Writer

class branch(TypedDict):
    branch_storier:str
    key_role:list[str]

class timeline_node(TypedDict):
    current_time: str
    background: str
    state: str
    end: str

class node(TypedDict):
    time_period: str
    role: str
    personality: str
    background: str
    state: str
    collision: str

class State(TypedDict):
    content:str
    system_background:str
    level_system:str
    combat_power_system:str
    unit_time:str
    protagonist:str
    storier:str
    roles:list[str]
    main_storyline:str
    branchs:list[branch]
    protagonist_timeline:list[timeline_node]
    happy_end:str
    bad_end:str
    node_list: list[node]
    sub_storier_list:Annotated[list[str],add]
    article:str

class Designer:
    def __init__(self):
        llm=ChatOpenAI(
            # api_key="",
            openai_api_key="ollama",
            model="qwen3.5:latest",
            base_url="http://127.0.0.1:11434/v1"#ollamaserve的地址/v1 协议版本
        )
        
        self.planner=Planner(llm=llm)
        self.screenwriter=Screenwriter(llm=llm)
        self.writer=Writer(llm=llm)
        
        self.graph=self.init_graph()
        
    def init_graph(self):
        builder = StateGraph(State)
        builder.add_node("planner_node",self.planner_node)
        builder.add_node("writer_node",self.writer_node)
        builder.add_node("screenwriter_node",self.screenwriter_node)
        builder.add_node("summarizer_node",self.summarizer_node)
        
        builder.add_edge(START,"planner_node")
        builder.add_edge("planner_node","screenwriter_node")
        builder.add_conditional_edges("screenwriter_node",self.dispatch_edge)
        builder.add_edge("writer_node","summarizer_node")
        builder.add_edge("summarizer_node",END)
        
        return builder.compile()
        
    def planner_node(self,state):
        while True:
            try:
                return self.planner(state)
            except Exception as e:
                print("planner error:",e)
    
    def writer_node(self,state):
        while True:
            try:
                rt=self.writer(state)
                return {"sub_storier_list":[rt['sub_storier']]}
            except Exception as e:
                print("writer error:",e)
    
    def screenwriter_node(self,state):
        while True:
            try:
                return self.screenwriter(state)
            except Exception as e:
                print("screenwriter error:",e)
    
    def summarizer_node(self,state):
        article=""
        for string in state.get("sub_storier_list",[]):
            article+=string
        return {"article":article}
    
    def dispatch_edge(self,state):
        system_background=state.get("system_background",[])
        level_system=state.get("level_system",[])
        combat_power_system=state.get("combat_power_system",[])
        main_storyline=state.get("main_storyline",[])
        
        ret=[]
        node_list=state.get('node_list',[])
        if len(node_list)==0:
            raise ValueError("剧情节点数不能为零!")
        for node in node_list:
            ret.append(Send("writer_node",{"system_background":system_background,"level_system":level_system,"combat_power_system":combat_power_system,"main_storyline":main_storyline,"node":node}))
        return ret
    
    def __call__(self,storier):
        return self.graph.invoke({"content":storier})
        
if __name__ == '__main__':
    designer=Designer()
    result=designer("高武玄幻龙傲天打脸爽文")
    print(result)
    with open("article.txt","w+") as f:
        f.write(result["article"])
        