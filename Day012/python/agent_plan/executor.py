from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import Annotated
from utils import websearch
from langgraph.prebuilt import create_react_agent


@tool
def web_search(query: Annotated[str, "互联网查询标题"]):
    """通过web_search工具查询互联网上的信息"""

    _rt = websearch(query)
    return _rt


_executor_system_template = """
您是一个优秀的子任务执行者，您需要根据子任务的名称和参考信息，完成子任务的信息查询。
您可以使用以下工具来协助您更好的完成该任务：
{tools_name}
"""

_executor_human_template = """
参考信息：
{infos}
子任务的名称：
{task}
"""


class Executor:

    def __init__(self, llm):

        _tools = [web_search,]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _executor_system_template),
            ("human", _executor_human_template)
        ])
        _prompt = _prompt.partial(tools_name = ",".join([_tool.name for _tool in _tools]))

        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)

        self._chain = _prompt | _llm_with_tools_agent
        self._parser = StrOutputParser()

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        return self._parser.invoke(_messages[-1])


if __name__ == '__main__':

    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:latest",
        api_key="ollama"
    )

    _executor = Executor(_llm)
    _rt = _executor({
        "infos": [],
        "task": "确定2024年法国巴黎奥运会女子10米跳水比赛的获胜者"
    })

    print(_rt)
