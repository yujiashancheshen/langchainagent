"""
使用 LangGraph 和阿里 Qwen3-max 模型创建的 Agent
可以调用 Excel 和问财工具来获取信息并写入 Excel
"""

import os
import sys
import time
import threading
import warnings
from typing import Dict, Any
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

# 导入 LangGraph (LangChain 1.0+ 的新方式)
try:
    from langgraph.prebuilt import create_react_agent  # type: ignore
except ImportError:
    raise ImportError(
        "无法导入 LangGraph。"
        "请确保已安装 langgraph: pip install langgraph"
    )

# 导入工具（必须在警告过滤器之后，以避免警告显示）
from excel_tools import read_excel, write_excel  # noqa: E402
from wencai_tool import get_iwencai_stock_data  # noqa: E402

# Token统计
_token_stats = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'call_count': 0
}


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """从响应中提取token使用情况"""
    usage = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

    if hasattr(response, 'response_metadata'):
        metadata = response.response_metadata
        if metadata and 'token_usage' in metadata:
            token_usage = metadata['token_usage']
            usage['prompt_tokens'] = token_usage.get('prompt_tokens', 0)
            usage['completion_tokens'] = token_usage.get(
                'completion_tokens', 0)
            usage['total_tokens'] = token_usage.get('total_tokens', 0)

    return usage


def _update_token_stats(usage: Dict[str, int]):
    """更新全局token统计"""
    _token_stats['total_prompt_tokens'] += usage['prompt_tokens']
    _token_stats['total_completion_tokens'] += usage['completion_tokens']
    _token_stats['total_tokens'] += usage['total_tokens']
    _token_stats['call_count'] += 1


def get_token_stats() -> Dict[str, Any]:
    """获取token统计信息"""
    return _token_stats.copy()


def reset_token_stats():
    """重置token统计"""
    global _token_stats
    _token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'call_count': 0
    }


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
        model_name (str): 模型名称，默认为 qwen3-max
        temperature (float): 模型温度参数，默认 0.1
        base_url (str): API 基础 URL，默认使用阿里云兼容模式

    Returns:
        AgentExecutor: 配置好的 Agent 执行器
    """
    # 获取 API Key
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
    tools = [
        get_iwencai_stock_data,
        read_excel,
        write_excel,
    ]

    # 创建 Agent (使用 LangGraph)
    agent_executor = create_react_agent(llm, tools)

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

            print("\n[处理中] 正在执行...\n")
            output = _run_langgraph_agent(
                agent_executor, user_input, verbose=True)

            print(f"\n[结果] {output}\n")
            print("-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n[agent] 错误 {e}\n")
            print("-" * 60)
            print()


def run_agent_single(
    agent_executor, query: str, verbose: bool = True
) -> str:
    """
    运行 Agent 处理单个查询

    Args:
        agent_executor: Agent 执行器
        query (str): 用户查询
        verbose (bool): 是否显示执行过程，默认为 True

    Returns:
        str: Agent 的回复
    """
    try:
        return _run_langgraph_agent(
            agent_executor, query, verbose=verbose)
    except Exception as e:
        return f"发生错误: {e}"


def _run_langgraph_agent(
    agent_executor, query: str, verbose: bool = True
) -> str:
    """运行langgraph agent的核心逻辑"""
    final_messages = None
    start_time = time.time()
    last_activity_time = time.time()
    step_count = 0
    last_tool_name = None
    # 使用字典共享状态，以便在进度指示器中使用
    status_info = {"tool": None, "step": 0}

    # 进度指示器
    progress_stop = threading.Event()

    def show_progress():
        """当长时间没有输出时显示进度指示"""
        dots = 0
        while not progress_stop.is_set():
            time.sleep(2)
            if time.time() - last_activity_time > 5:
                dots = (dots + 1) % 4
                # 根据当前状态显示更具体的信息
                if status_info["tool"]:
                    status_msg = f"[tool] 工具执行中: {status_info['tool']}"
                elif status_info["step"] > 0:
                    status_msg = "[agent] LLM处理中"
                else:
                    status_msg = "[agent] 正在执行"
                msg = (f"\r等待中{'.' * dots}{' ' * (3-dots)} "
                       f"{status_msg}，请稍候...")
                sys.stdout.write(msg)
                sys.stdout.flush()

    if verbose:
        print("=" * 60)
        print("执行过程:")
        print("=" * 60)
        progress_thread = threading.Thread(
            target=show_progress, daemon=True)
        progress_thread.start()

    try:
        stream_input = {"messages": [("user", query)]}
        for chunk in agent_executor.stream(stream_input):
            last_activity_time = time.time()
            if verbose:
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()

            for node_name, node_output in chunk.items():
                if node_name == "agent":
                    step_count += 1
                    status_info["step"] = step_count
                    status_info["tool"] = None  # 清除工具状态
                    if isinstance(node_output, dict):
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            final_messages = messages
                            if messages and verbose:
                                last_msg = messages[-1]
                                if hasattr(last_msg, 'content'):
                                    content = last_msg.content
                                    if content:
                                        # 提取token使用情况
                                        usage = _extract_token_usage(last_msg)
                                        if usage['total_tokens'] > 0:
                                            _update_token_stats(usage)
                                            print(
                                                f"\n[agent] 步骤{step_count} "
                                                f"LLM调用 - "
                                                f"Token: "
                                                f"{usage['total_tokens']} "
                                                f"(输入:"
                                                f"{usage['prompt_tokens']} "
                                                f"输出:"
                                                f"{usage['completion_tokens']}"
                                                f")"
                                            )
                                        else:
                                            print(
                                                f"\n[agent] 步骤{step_count} "
                                                "Agent思考"
                                            )
                elif node_name == "tools":
                    if isinstance(node_output, dict):
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            for msg in messages:
                                if hasattr(msg, 'name') and msg.name:
                                    tool_name = msg.name
                                    if tool_name != last_tool_name:
                                        status_info["tool"] = tool_name
                                        if verbose:
                                            print(
                                                f"[tool] 调用: {tool_name}"
                                            )
                                        last_tool_name = tool_name
                                if hasattr(msg, 'content') and verbose:
                                    content = str(msg.content)
                                    if content:
                                        # 只显示工具结果摘要
                                        result_preview = (
                                            content[:100]
                                            if len(content) > 100
                                            else content
                                        )
                                        suffix = (
                                            '...' if len(content) > 100
                                            else ''
                                        )
                                        print(
                                            f"[tool] 结果: "
                                            f"{result_preview}{suffix}"
                                        )

        if verbose:
            progress_stop.set()
            total_time = time.time() - start_time
            stats = get_token_stats()
            print(f"\n[agent] 完成 总耗时: {total_time:.1f}秒")
            if stats['call_count'] > 0:
                print(
                    f"[agent] 统计 Token使用 - "
                    f"总计:{stats['total_tokens']} "
                    f"(输入:{stats['total_prompt_tokens']} "
                    f"输出:{stats['total_completion_tokens']}) "
                    f"调用次数:{stats['call_count']}"
                )
            print("=" * 60)

    except Exception as stream_error:
        if verbose:
            progress_stop.set()
            print(f"\n[agent] 错误 {stream_error}")
        # 尝试使用 invoke 作为后备方案
        try:
            invoke_input = {"messages": [("user", query)]}
            response = agent_executor.invoke(invoke_input)
            if isinstance(response, dict):
                if "messages" in response:
                    final_messages = response["messages"]
        except Exception:
            pass

    # 从最后的消息中提取输出
    if final_messages:
        last_message = final_messages[-1]
        if hasattr(last_message, 'content'):
            return last_message.content
        else:
            return str(last_message)
    return "无法获取响应"


# --- 主程序 ---
if __name__ == "__main__":
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
