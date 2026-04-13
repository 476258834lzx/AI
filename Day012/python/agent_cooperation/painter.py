from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from utils import repl
from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage


@tool
def python_repl(code: Annotated[str, "生成图表所需执行的python代码。"]):
    "使用它来执行python代码。如果你想看到一个值的输出，你应该用`print（…）`打印出来。这对用户是可见的。"
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


_painter_system_template = """
你是一个乐于助人的人工智能助手，与其他助手合作。
使用提供的工具来回答问题。
如果你不能完全回答，没关系，另一个助手用不同的工具。
这将有助于你取得进展。尽你所能取得进展。
如果你或其他助理有最终答案或可交付成果，在你的回答前加上FINAL ANSWER，这样团队就知道该停下来了。
您可以访问以下工具：
{tools_name}
"""


class Painter:

    def __init__(self, llm):
        _tools = [python_repl]

        _prompt = ChatPromptTemplate.from_messages([
            ("system", _painter_system_template),
            MessagesPlaceholder(variable_name="messages")
        ])

        _prompt = _prompt.partial(tools_name=",".join([_tool.name for _tool in _tools]))

        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)
        self._chain = _prompt | _llm_with_tools_agent

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        # if isinstance(_messages[-2],ToolMessage):
        #     return _messages[-2:]
        # else:
        return _messages[-1]


if __name__ == '__main__':
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:latest",
        api_key="ollama"
    )

    # _researcher = Researcher(_llm)
    # _rt = _researcher({"query":"获取英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，就完成。"})

    # print(_rt)