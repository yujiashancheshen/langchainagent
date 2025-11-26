"""
使用 LangChain 框架和阿里 Qwen3-max 模型创建的 Agent
可以调用 Excel 和问财工具来获取信息并写入 Excel
"""

import os
import warnings
from dotenv import load_dotenv  # type: ignore

# 加载 .env 文件中的环境变量
load_dotenv()

# 必须在导入 LangChain 之前设置，以抑制 Pydantic V1 与 Python 3.14+ 的兼容性警告
# LangChain 内部仍使用 Pydantic V1，但功能正常
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14",
    category=UserWarning,
)

# 以下导入需要在警告过滤器之后执行
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI  # type: ignore
    except ImportError:
        raise ImportError(
            "请安装 langchain-openai 或 langchain: "
            "pip install langchain-openai 或 pip install langchain"
        )

try:
    # 尝试使用 langgraph (LangChain 1.0+ 的新方式)
    from langgraph.prebuilt import create_react_agent  # type: ignore
    USE_LANGGRAPH = True
    USE_OLD_VERSION = False
    create_react_agent_func = create_react_agent
    initialize_agent = None  # 新版本不需要
    AgentType = None
except ImportError:
    try:
        # 尝试导入旧版本的 LangChain
        from langchain.agents import (  # type: ignore
            initialize_agent, AgentType
        )
        USE_LANGGRAPH = False
        USE_OLD_VERSION = True
        create_react_agent_func = None
    except ImportError:
        raise ImportError(
            "无法导入 LangChain agents。"
            "请确保已安装 langchain 和 langgraph: "
            "pip install langchain langgraph"
        )

# 导入工具（必须在警告过滤器之后，以避免警告显示）
from excel_tools import read_excel, write_excel  # noqa: E402
from wencai_tool import get_iwencai_stock_data  # noqa: E402

# 注意：USE_LANGGRAPH 和 USE_OLD_VERSION 在导入时已设置，不需要再次初始化


def create_agent(
    api_key: str = None,
    model_name: str = "qwen3-max",
    temperature: float = 0.1,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
):
    """
    创建并配置 Agent

    Args:
        api_key (str): 阿里云 API Key，如果不提供则按以下顺序读取：
            1. 从 .env 文件中的 DASHSCOPE_API_KEY
            2. 从环境变量 DASHSCOPE_API_KEY
            3. 如果都未设置则抛出错误
        model_name (str): 模型名称，默认为 qwen3-max，也可以使用 qwen-plus, qwen-turbo 等
        temperature (float): 模型温度参数，默认 0.1
        base_url (str): API 基础 URL，默认使用阿里云兼容模式

    Returns:
        AgentExecutor: 配置好的 Agent 执行器
    """
    # 获取 API Key（优先从 .env 文件读取，然后从环境变量读取）
    if api_key is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "请设置 DASHSCOPE_API_KEY：\n"
                "  1. 在 .env 文件中添加：DASHSCOPE_API_KEY=your_api_key\n"
                "  2. 或设置环境变量：export DASHSCOPE_API_KEY='your_api_key'\n"
                "  3. 或在调用时传入 api_key 参数"
            )

    # 初始化 Qwen 模型
    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
    )

    # 定义工具列表
    # 这些工具已经用 @tool 装饰器装饰，可以直接使用
    tools = [
        get_iwencai_stock_data,
        read_excel,
        write_excel,
    ]

    # 创建 Agent
    if USE_LANGGRAPH and create_react_agent_func is not None:
        # 使用 langgraph (LangChain 1.0+ 的新方式)
        # create_react_agent 返回一个可以直接运行的 graph
        agent_executor = create_react_agent_func(llm, tools)
    elif not USE_LANGGRAPH and initialize_agent is not None:
        # 使用旧版本的 LangChain
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
        )
    else:
        raise ImportError(
            "无法创建 Agent：既无法使用 langgraph，也无法使用旧版本的 LangChain。"
            "请检查安装：pip install langchain langgraph"
        )

    return agent_executor


def run_agent_interactive(agent_executor):
    """
    以交互模式运行 Agent

    Args:
        agent_executor: Agent 执行器
    """
    print("=" * 60)
    print("Agent 已启动！")
    print("你可以询问关于股票数据的问题，Agent 会帮你获取并处理数据。")
    print("输入 '退出' 或 'exit' 来结束对话。")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("再见！")
                break

            print("\nAgent 正在处理...\n")
            # 兼容新旧版本的 LangChain
            # USE_LANGGRAPH 是模块级别的全局变量
            global USE_LANGGRAPH
            if USE_LANGGRAPH:
                # langgraph 使用 invoke 方法，返回字典
                response = agent_executor.invoke(
                    {"messages": [("user", user_input)]}
                )
                # 从响应中提取最后一条消息
                if isinstance(response, dict) and "messages" in response:
                    messages = response["messages"]
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            output = last_message.content
                        else:
                            output = str(last_message)
                    else:
                        output = str(response)
                else:
                    output = str(response)
            elif hasattr(agent_executor, 'invoke'):
                response = agent_executor.invoke({"input": user_input})
                output = response.get("output", str(response))
            else:
                # 旧版本使用 run 方法
                output = agent_executor.run(user_input)
            print(f"\nAgent: {output}\n")
            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}\n")
            print("-" * 60)
            print()


def run_agent_single(agent_executor, query: str) -> str:
    """
    运行 Agent 处理单个查询

    Args:
        agent_executor: Agent 执行器
        query (str): 用户查询

    Returns:
        str: Agent 的回复
    """
    try:
        # 兼容新旧版本的 LangChain
        # USE_LANGGRAPH 是模块级别的全局变量
        global USE_LANGGRAPH
        if USE_LANGGRAPH:
            # langgraph 使用 invoke 方法
            response = agent_executor.invoke({"messages": [("user", query)]})
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    else:
                        return str(last_message)
            return str(response)
        elif hasattr(agent_executor, 'invoke'):
            response = agent_executor.invoke({"input": query})
            return response.get("output", str(response))
        else:
            # 旧版本使用 run 方法
            return agent_executor.run(query)
    except Exception as e:
        return f"发生错误: {e}"


# --- 主程序 ---
if __name__ == "__main__":
    import sys

    # 创建 Agent
    try:
        agent = create_agent()
    except ValueError as e:
        print(f"错误: {e}")
        print("\n请设置环境变量：")
        print("export DASHSCOPE_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 如果提供了命令行参数，则执行单次查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"查询: {query}\n")
        result = run_agent_single(agent, query)
        print(f"\n结果:\n{result}")
    else:
        # 否则进入交互模式
        run_agent_interactive(agent)
