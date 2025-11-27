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
        # 打印入参
        print(f"\n[agent] LLM调用 #{self.call_index} - 入参:")
        for i, prompt in enumerate(self.current_call['input']):
            if isinstance(prompt, str):
                prompt_str = prompt
            elif hasattr(prompt, 'content'):
                prompt_str = prompt.content
            elif isinstance(prompt, list):
                # 处理消息列表
                prompt_str = "\n".join([
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    if isinstance(msg, dict)
                    else str(msg)
                    for msg in prompt
                ])
            else:
                prompt_str = str(prompt)
            # 限制显示长度，避免输出过长
            if len(prompt_str) > 1000:
                truncated = prompt_str[:1000]
                total_len = len(prompt_str)
                print(
                    f"[agent]   提示 {i+1}: {truncated}... "
                    f"(已截断，总长度: {total_len})"
                )
            else:
                print(f"[agent]   提示 {i+1}: {prompt_str}")

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

            # 打印出参
            output_str = self.current_call['output']
            call_idx = self.current_call['call_index']
            if len(output_str) > 1000:
                truncated = output_str[:1000]
                total_len = len(output_str)
                print(
                    f"[agent] LLM调用 #{call_idx} - 出参: "
                    f"{truncated}... (已截断，总长度: {total_len})"
                )
            else:
                print(
                    f"[agent] LLM调用 #{call_idx} - 出参: {output_str}"
                )

            if self.current_call['token_usage']:
                usage = self.current_call['token_usage']
                total = usage.get('total_tokens', 0)
                prompt = usage.get('prompt_tokens', 0)
                completion = usage.get('completion_tokens', 0)
                print(
                    f"[agent] LLM调用 #{call_idx} - Token使用: "
                    f"总计={total}, 输入={prompt}, 输出={completion}"
                )

            # 保存到日志
            _llm_call_logs.append(self.current_call.copy())
            self.current_call = None

    def on_llm_error(self, error, **kwargs):
        """LLM调用出错时记录"""
        if self.current_call:
            self.current_call['error'] = str(error)
            call_idx = self.current_call['call_index']
            print(
                f"[agent] LLM调用 #{call_idx} - 错误: {error}"
            )
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
    print("[agent] " + "=" * 60)
    print("[agent] Agent 已启动！")
    print("[agent] 你可以询问关于股票数据的问题，Agent 会帮你获取并处理数据。")
    print("[agent] 输入 '退出' 或 'exit' 来结束对话。")
    print("[agent] " + "=" * 60)
    print()

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

            print("\n[agent] 处理中 正在执行...\n")
            output = _run_langgraph_agent(
                agent_executor, user_input, verbose=True)

            print(f"\n[agent] 结果 {output}\n")
            print("[agent] " + "-" * 60)
            print()

        except KeyboardInterrupt:
            print("\n\n[agent] 再见！")
            break
        except Exception as e:
            print(f"\n[agent] 错误 {e}\n")
            print("[agent] " + "-" * 60)
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
        print("[agent] " + "=" * 60)
        print("[agent] 执行过程:")
        print("[agent] " + "=" * 60)
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

                                            # 精简打印：只显示核心步骤
                                            print("[agent] 调用大模型")

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
                                                call_index=step_count,
                                                input_messages=input_messages,
                                                output=str(content),
                                                token_usage=usage
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
                            tool_input = None
                            tool_output = None
                            tool_name = None

                            for msg in messages:
                                if hasattr(msg, 'name') and msg.name:
                                    tool_name = msg.name
                                    if tool_name != last_tool_name:
                                        status_info["tool"] = tool_name
                                        if verbose:
                                            print(
                                                f"[agent] 调用工具: {tool_name}"
                                            )
                                        last_tool_name = tool_name

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
            progress_stop.set()
            total_time = time.time() - start_time
            stats = get_token_stats()
            print(f"[agent] 完成 (耗时: {total_time:.1f}秒)")
            if stats['call_count'] > 0:
                call_count = stats['call_count']
                total_tokens = stats['total_tokens']
                print(
                    f"[agent] 统计: LLM调用 {call_count} 次, "
                    f"Token消耗 {total_tokens}"
                )

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
        print(f"[agent] 错误: {e}")
        print("[agent] 请设置环境变量：")
        print("[agent] export DASHSCOPE_API_KEY='your_api_key_here'")
        sys.exit(1)

    # 如果提供了命令行参数，则执行单次查询
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"[agent] 查询: {query}\n")
        result = run_agent_single(agent, query)
        print(f"\n[agent] 结果:\n{result}")
    else:
        # 否则进入交互模式
        run_agent_interactive(agent)
