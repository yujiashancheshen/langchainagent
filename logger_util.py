"""
日志工具模块
用于记录详细的输入输出信息到日志文件
"""
import os
import logging
from datetime import datetime
from typing import Any, Optional, Dict


# 日志目录
LOG_DIR = os.path.expanduser("./agent_logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志文件名（按日期）
LOG_FILE = os.path.join(
    LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d')}.log"
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ]
)

logger = logging.getLogger('agent')


def log_llm_call(
    call_index: int,
    input_messages: Any,
    output: str,
    token_usage: Optional[Dict[str, int]] = None
):
    """
    记录LLM调用信息

    Args:
        call_index: 调用序号
        input_messages: 输入消息
        output: 输出内容
        token_usage: Token使用情况
    """
    logger.info(f"[LLM调用 #{call_index}]")
    logger.info(f"输入: {input_messages}")
    logger.info(f"输出: {output}")
    if token_usage:
        logger.info(f"Token使用: {token_usage}")


def log_tool_call(tool_name: str, input_data: Any, output: str):
    """
    记录工具调用信息

    Args:
        tool_name: 工具名称
        input_data: 输入数据
        output: 输出数据
    """
    logger.info(f"[工具调用: {tool_name}]")
    logger.info(f"输入: {input_data}")
    logger.info(f"输出: {output}")


def log_error(error_msg: str, error_type: str = "错误"):
    """
    记录错误信息

    Args:
        error_msg: 错误消息
        error_type: 错误类型
    """
    logger.error(f"[{error_type}] {error_msg}")

