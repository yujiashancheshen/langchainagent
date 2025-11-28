"""
使用 LangGraph 和阿里 Qwen3-max 模型创建的 Agent
可以调用 Excel 和问财工具来获取信息并写入 Excel
"""

import os
import sys
import time
import warnings
from typing import Dict, Any
from dotenv import load_dotenv  # type: ignore
from logger_util import log_llm_call, log_tool_call  # noqa: E402

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
    from langchain_core.callbacks import (
        BaseCallbackHandler  # type: ignore
    )
except ImportError:
    try:
        from langchain.callbacks.base import (
            BaseCallbackHandler  # type: ignore
        )
    except ImportError:
        BaseCallbackHandler = None

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
from analyze_tool import merge_json_data

# Token统计
_token_stats = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'call_count': 0
}

# LLM调用日志记录
_llm_call_logs = []


class LLMLoggingCallback(
    BaseCallbackHandler if BaseCallbackHandler else object
):
    """自定义Callback用于记录LLM调用的入参和出参"""

    def __init__(self):
        self.current_call = None
        self.call_index = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM调用开始时记录入参"""
        self.call_index += 1
        self.current_call = {
            'call_index': self.call_index,
            'input': prompts if isinstance(prompts, list) else [prompts],
            'output': None,
            'token_usage': None
        }

    def on_llm_end(self, response, **kwargs):
        """LLM调用结束时记录出参"""
        if self.current_call:
            # 提取输出内容
            if hasattr(response, 'generations'):
                output_content = []
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, 'text'):
                            output_content.append(gen.text)
                        elif (hasattr(gen, 'message') and
                              hasattr(gen.message, 'content')):
                            output_content.append(gen.message.content)
                        else:
                            output_content.append(str(gen))
                self.current_call['output'] = "\n".join(output_content)
            elif hasattr(response, 'llm_output'):
                self.current_call['output'] = str(response.llm_output)
            else:
                self.current_call['output'] = str(response)

            # 提取token使用情况
            if hasattr(response, 'llm_output') and response.llm_output:
                token_usage = response.llm_output.get('token_usage', {})
                if token_usage:
                    self.current_call['token_usage'] = token_usage
                    _update_token_stats({
                        'prompt_tokens': token_usage.get(
                            'prompt_tokens', 0
                        ),
                        'completion_tokens': token_usage.get(
                            'completion_tokens', 0
                        ),
                        'total_tokens': token_usage.get('total_tokens', 0)
                    })

            # 准备输入消息用于日志
            input_messages = []
            for prompt in self.current_call['input']:
                if isinstance(prompt, str):
                    prompt_str = prompt
                elif hasattr(prompt, 'content'):
                    prompt_str = prompt.content
                elif isinstance(prompt, list):
                    # 处理消息列表
                    prompt_str = "\n".join([
                        (f"{msg.get('role', 'unknown')}: "
                         f"{msg.get('content', '')}")
                        if isinstance(msg, dict)
                        else str(msg)
                        for msg in prompt
                    ])
                else:
                    prompt_str = str(prompt)
                input_messages.append(prompt_str)

            # 将详细输入输出写入日志
            log_llm_call(
                call_index=self.current_call['call_index'],
                input_messages=input_messages,
                output=self.current_call['output'],
                token_usage=self.current_call.get('token_usage')
            )

            # 保存到日志
            _llm_call_logs.append(self.current_call.copy())
            self.current_call = None

    def on_llm_error(self, error, **kwargs):
        """LLM调用出错时记录"""
        if self.current_call:
            self.current_call['error'] = str(error)
            from logger_util import log_error
            call_idx = self.current_call['call_index']
            log_error(f"LLM调用 #{call_idx} 错误: {error}", "LLM")
            _llm_call_logs.append(self.current_call.copy())
            self.current_call = None


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
    global _token_stats, _llm_call_logs
    _token_stats = {
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'total_tokens': 0,
        'call_count': 0
    }
    _llm_call_logs = []


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
    # 创建callback用于记录LLM调用
    callbacks = []
    if BaseCallbackHandler:
        callbacks.append(LLMLoggingCallback())

    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=temperature,
        callbacks=callbacks if callbacks else None,
    )

    # 定义工具列表
    tools = [
        get_iwencai_stock_data,
        read_excel,
        write_excel,
        merge_json_data,
    ]

    # 创建 Agent (使用 LangGraph)
    # 打开debug参数以显示agent的执行过程
    agent_executor = create_react_agent(llm, tools, debug=False)

    return agent_executor


def run_agent_interactive(agent_executor):
    """
    以交互模式运行 Agent

    Args:
        agent_executor: Agent 执行器
    """
    print("[agent] Agent已启动，输入'退出'或'exit'结束对话")

    while True:
        try:
            # 每次交互前重置统计
            reset_token_stats()

            user_input = input("你: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("[agent] 再见！")
                break

            output = _run_langgraph_agent(
                agent_executor, user_input, verbose=True)

            print(f"[agent] 结果: {output}")

        except KeyboardInterrupt:
            print("\n[agent] 再见！")
            break
        except Exception as e:
            from logger_util import log_error
            log_error(f"交互模式错误: {e}", "Agent")
            print(f"[agent] 错误: {e}")


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
        # 重置统计
        reset_token_stats()
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
    llm_call_count = 0
    tool_call_count = 0
    tool_calls = []  # 记录工具调用信息

    try:
        stream_input = {"messages": [("user", query)]}
        for chunk in agent_executor.stream(stream_input):
            for node_name, node_output in chunk.items():
                if node_name == "agent":
                    if isinstance(node_output, dict):
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            final_messages = messages
                            if messages:
                                last_msg = messages[-1]
                                if hasattr(last_msg, 'content'):
                                    content = last_msg.content
                                    if content:
                                        # 提取token使用情况
                                        usage = _extract_token_usage(last_msg)
                                        if usage['total_tokens'] > 0:
                                            _update_token_stats(usage)
                                            llm_call_count += 1
                                            if verbose:
                                                msg = (
                                                    f"[agent] LLM调用 "
                                                    f"#{llm_call_count}"
                                                )
                                                print(msg)

                                            # 准备输入消息用于日志
                                            input_messages = []
                                            if len(messages) > 1:
                                                for msg in messages[:-1]:
                                                    msg_role = getattr(
                                                        msg, 'role', 'unknown'
                                                    )
                                                    if hasattr(msg, 'content'):
                                                        msg_content = str(
                                                            msg.content
                                                        )
                                                    elif isinstance(msg, dict):
                                                        msg_content = msg.get(
                                                            'content', str(msg)
                                                        )
                                                    else:
                                                        msg_content = str(msg)
                                                    input_messages.append({
                                                        'role': msg_role,
                                                        'content': msg_content
                                                    })

                                            # 将详细输入输出写入日志
                                            log_llm_call(
                                                call_index=llm_call_count,
                                                input_messages=input_messages,
                                                output=str(content),
                                                token_usage=usage
                                            )
                elif node_name == "tools":
                    if isinstance(node_output, dict):
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            tool_input = None
                            tool_output = None
                            tool_name = None

                            for msg in messages:
                                if hasattr(msg, 'name') and msg.name:
                                    tool_name = msg.name
                                    tool_names = [
                                        t['name'] for t in tool_calls
                                    ]
                                    if tool_name not in tool_names:
                                        tool_calls.append({
                                            'name': tool_name,
                                            'count': 0
                                        })
                                    # 更新工具调用计数
                                    for t in tool_calls:
                                        if t['name'] == tool_name:
                                            t['count'] += 1
                                            break
                                    tool_call_count += 1
                                    if verbose:
                                        msg_text = (
                                            f"[agent] 工具调用 "
                                            f"#{tool_call_count}: {tool_name}"
                                        )
                                        print(msg_text)

                                # 收集工具输入输出
                                if hasattr(msg, 'content'):
                                    content = str(msg.content)
                                    if content:
                                        # 判断是输入还是输出
                                        if hasattr(msg, 'role'):
                                            if msg.role == 'tool':
                                                tool_output = content
                                            else:
                                                tool_input = content
                                        else:
                                            # 默认认为是输出
                                            if tool_output is None:
                                                tool_output = content

                            # 将工具详细输入输出写入日志
                            if tool_name and (tool_input or tool_output):
                                log_tool_call(
                                    tool_name=tool_name,
                                    input_data=tool_input,
                                    output=tool_output or ""
                                )

        if verbose:
            total_time = time.time() - start_time
            stats = get_token_stats()
            # 汇总信息：一行显示
            tool_summary_parts = [
                f"{t['name']}({t['count']})" for t in tool_calls
            ]
            tool_summary = ", ".join(tool_summary_parts)
            summary = (
                f"[agent] 完成 | 耗时: {total_time:.1f}s | "
                f"LLM: {llm_call_count}次 | 工具: {tool_call_count}次"
            )
            if tool_summary:
                summary += f" ({tool_summary})"
            if stats['call_count'] > 0:
                summary += f" | Tokens: {stats['total_tokens']}"
            print(summary)

    except Exception as stream_error:
        if verbose:
            from logger_util import log_error
            log_error(f"Agent执行错误: {stream_error}", "Agent")
            print(f"[agent] 错误: {stream_error}")
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
        print(f"[agent] 错误: {e}")
        print("[agent] 请设置环境变量：")
        print("[agent] export DASHSCOPE_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 如果提供了命令行参数，则执行单次查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"[agent] 查询: {query}")
        result = run_agent_single(agent, query)
        print(f"[agent] 结果: {result}")
    else:
        # 否则进入交互模式
        run_agent_interactive(agent)
