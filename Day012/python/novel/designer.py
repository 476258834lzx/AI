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
    suggestion:str
    node_list: list[node]
    current_node:node
    previous_end:str
    sub_storier_list:Annotated[list[str],add]
    article:str

class Designer:
    def __init__(self):
        self.planner      = Planner(llm=self.create_llm(temperature=0.7))
        self.screenwriter = Screenwriter(llm=self.create_llm(temperature=0.1))
        self.writer       = Writer(llm=self.create_llm(temperature=0.7))
        
        self.graph=self.init_graph()
        png_bytes = self.graph.get_graph().draw_mermaid_png()
        with open("storier_graph.png", "wb") as f:
            f.write(png_bytes)
        
    def create_llm(self,temperature):
        return ChatOpenAI(
            openai_api_key="ollama",
            model="qwen3.5:latest",
            base_url="http://127.0.0.1:11434/v1" ,
            temperature=temperature,
            timeout=300,
            max_retries=2,
        )
    
    def init_graph(self):
        builder = StateGraph(State)
        builder.add_node("planner_node",self.planner_node)
        builder.add_node("writer_node",self.writer_node)
        builder.add_node("screenwriter_node",self.screenwriter_node)
        builder.add_node("summarizer_node",self.summarizer_node)
        
        builder.add_edge(START,"planner_node")
        builder.add_edge("planner_node","screenwriter_node")
        builder.add_conditional_edges("screenwriter_node",self.coperation_router,{"continue":"planner_node","break":"writer_node"})
        builder.add_conditional_edges("writer_node",self.plan_router,["summarizer_node","writer_node"])
        builder.add_edge("summarizer_node",END)
        
        return builder.compile()
        
    def planner_node(self,state):
        while True:
            try:
                return self.planner(state)
            except Exception as e:
                print("planner error:",e)
    
    def writer_node(self,state):
        node_list=state["node_list"]
        system_background=state["system_background"]
        roles=state["roles"]
        level_system=state["level_system"]
        combat_power_system=state["combat_power_system"]
        storier=state["storier"]
        sub_storier_list=state.get("sub_storier_list",[])
        node_index=len(sub_storier_list)
        current_node=node_list[node_index]
        previous_end=sub_storier_list[-1][-100:] if len(sub_storier_list)>0 else ""
        
        while True:
            try:
                rt=self.writer({"system_background":system_background,"roles":roles,"level_system":level_system,"combat_power_system":combat_power_system,"storier":storier,"current_node":current_node,"previous_end":previous_end})
                return {"sub_storier_list":[rt]}
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
    
    def coperation_router(self,state):
        is_final=state["is_final"]
        if is_final:
            return "break"

        return "continue"
    
    def plan_router(self,state):
        node_len=len(state["node_list"])
        sub_storier_len=len(state["sub_storier_list"])
        if node_len==sub_storier_len:
            return "summarizer_node"
        else:
            return "writer_node"

    def get_initial_state(self) -> State:
        return {
            "content": "",
            "system_background": "",
            "level_system": "",
            "combat_power_system": "",
            "unit_time": "",
            "protagonist": "",
            "storier": "",
            "roles": [],
            "main_storyline": "",
            "branchs": [],
            "protagonist_timeline": [],
            "suggestion": "",
            "node_list": [],
            "previous_end": "",
            "sub_storier_list": [], # 带有 add reducer 的字段，初始给空列表即可
            "article": ""
        }
    
    def __call__(self,story):
        initial_state = self.get_initial_state()
        initial_state["content"]=story
        yield from self.graph.stream(initial_state)
        
if __name__ == '__main__':
    designer=Designer()
    for result in designer("高武玄幻龙傲天打脸爽文"):
        print(result)  # 实时打印每一步的输出
        if "summarizer_node" in result:
            # 提取嵌套在里面的 article 内容
            article_content = result["summarizer_node"]["article"]
        
            # 保存到文件
            with open("article.txt", "w+", encoding="utf-8") as f:
                f.write(article_content)
                print(">>> 文章已保存至article.txt")