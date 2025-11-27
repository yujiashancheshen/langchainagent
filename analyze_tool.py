"""
分析工具：对两个JSON列表取交集并合并列
用于在Agent中调用，自动检测匹配字段并合并数据
"""
import json
from typing import List, Dict, Any, Optional
from langchain.tools import tool  # type: ignore


def _detect_key_field(
    json1: List[Dict[str, Any]],
    json2: List[Dict[str, Any]]
) -> Optional[str]:
    """
    自动检测两个JSON列表共有的、适合作为匹配键的字段

    Args:
        json1: 第一个JSON列表
        json2: 第二个JSON列表

    Returns:
        检测到的键字段名，如果未找到则返回None
    """
    if not json1 or not json2:
        return None

    # 获取两个列表的所有字段
    fields1 = set(json1[0].keys()) if json1 else set()
    fields2 = set(json2[0].keys()) if json2 else set()

    # 找到共有的字段
    common_fields = fields1 & fields2

    # 移除空键
    common_fields = {f for f in common_fields if f and f.strip()}

    if not common_fields:
        return None

    # 优先选择看起来像ID的字段
    # 优先级：包含"代码"、"id"、"ID"、"编号"等关键词的字段
    priority_keywords = ["代码", "id", "ID", "编号", "code", "Code"]
    for keyword in priority_keywords:
        for field in common_fields:
            if keyword in field:
                # 验证该字段在两个列表中都有有效值
                if _is_valid_key_field(json1, json2, field):
                    return field

    # 如果没有找到优先级字段，选择第一个共有字段
    for field in common_fields:
        if _is_valid_key_field(json1, json2, field):
            return field

    return None


def _is_valid_key_field(
    json1: List[Dict[str, Any]],
    json2: List[Dict[str, Any]],
    field: str
) -> bool:
    """
    验证字段是否适合作为匹配键

    Args:
        json1: 第一个JSON列表
        json2: 第二个JSON列表
        field: 字段名

    Returns:
        如果字段适合作为匹配键返回True
    """
    # 检查字段在两个列表中都有值
    has_value1 = any(
        item.get(field) and str(item.get(field)).strip()
        for item in json1[:10]  # 只检查前10条
    )
    has_value2 = any(
        item.get(field) and str(item.get(field)).strip()
        for item in json2[:10]  # 只检查前10条
    )

    return has_value1 and has_value2


def _merge_json_lists(
    json1: List[Dict[str, Any]],
    json2: List[Dict[str, Any]],
    key_field: str
) -> List[Dict[str, Any]]:
    """
    对两个JSON列表取交集并合并列（内部函数）

    Args:
        json1: 第一个JSON列表（字典列表）
        json2: 第二个JSON列表（字典列表）
        key_field: 用于匹配的键字段名

    Returns:
        合并后的JSON列表，只包含两个列表中都存在的记录
    """
    # 将第二个列表转换为以key_field为键的字典，方便查找
    # 如果有重复的key，选择数据最完整的那条（非空字段最多的）
    json2_dict = {}
    for item in json2:
        if key_field in item and item[key_field]:
            key_value = str(item[key_field]).strip()
            if key_value:
                # 如果已存在，比较数据完整性（非空字段数量）
                if key_value in json2_dict:
                    existing_item = json2_dict[key_value]
                    existing_non_empty = sum(
                        1 for v in existing_item.values()
                        if v and (not isinstance(v, str) or v.strip())
                    )
                    current_non_empty = sum(
                        1 for v in item.values()
                        if v and (not isinstance(v, str) or v.strip())
                    )
                    # 如果当前记录更完整，则替换
                    if current_non_empty > existing_non_empty:
                        json2_dict[key_value] = item
                else:
                    json2_dict[key_value] = item

    # 遍历第一个列表，找到交集并合并
    result = []
    for item1 in json1:
        if key_field not in item1 or not item1[key_field]:
            continue

        key_value = str(item1[key_field]).strip()
        if not key_value:
            continue

        # 如果在第二个列表中找到匹配项
        if key_value in json2_dict:
            item2 = json2_dict[key_value]

            # 合并两个字典，去重列
            # 策略：优先保留第一个字典的值，如果第一个字典的值为空，
            # 则用第二个字典的值填充
            merged_item = {}

            # 先添加第一个字典的所有字段
            for k, v in item1.items():
                merged_item[k] = v

            # 再添加第二个字典的字段
            for k, v in item2.items():
                # 如果键不存在，直接添加
                if k not in merged_item:
                    merged_item[k] = v
                # 如果键已存在但值为空（或只有空白字符），则用新值替换
                elif not merged_item[k] or (
                    isinstance(merged_item[k], str) and
                    not merged_item[k].strip()
                ):
                    merged_item[k] = v
                # 如果键已存在且有值，保留原值（不覆盖）

            result.append(merged_item)

    return result


@tool
def merge_json_data(json1_str: str, json2_str: str) -> str:
    """
    对两个JSON列表取交集并合并列。自动检测两个JSON共有的字段作为匹配键。

    输入参数：
        json1_str (str): 第一个JSON字符串，必须是数组格式，例如：
            '[{"股票代码": "300308", "股票简称": "中际旭创", ...}, ...]'
        json2_str (str): 第二个JSON字符串，必须是数组格式，例如：
            '[{"股票代码": "300308", "个股热度": "12.31万", ...}, ...]'

    返回：
        str: 合并后的JSON字符串，包含两个列表的交集记录，所有列已去重合并。
            如果两个列表中有相同的键，会优先保留第一个列表的值。
            如果第一个列表中的值为空，会用第二个列表的值填充。

    示例：
        输入：
            json1_str: '[{"股票代码": "300308", "成交额": "236.41亿"}, ...]'
            json2_str: '[{"股票代码": "300308", "热度": "12.31万"}, ...]'
        输出：
            '[{"股票代码": "300308", "成交额": "236.41亿", "热度": "12.31万"}, ...]'

    注意：
        - 两个JSON必须是数组格式（列表）
        - 会自动检测共有的字段作为匹配键（优先选择包含"代码"、"id"等关键词的字段）
        - 只返回两个列表中都存在的记录（基于匹配键）
        - 如果匹配键字段为空或不存在，该记录会被跳过
    """
    try:
        # 解析JSON字符串
        json1 = json.loads(json1_str)
        json2 = json.loads(json2_str)

        # 验证是列表
        if not isinstance(json1, list) or not isinstance(json2, list):
            error_msg = "JSON必须是数组格式（列表）"
            return json.dumps({
                "error": error_msg,
                "success": False
            }, ensure_ascii=False)

        if not json1 or not json2:
            error_msg = "JSON列表不能为空"
            return json.dumps({
                "error": error_msg,
                "success": False,
                "result": []
            }, ensure_ascii=False)

        # 自动检测匹配键字段
        key_field = _detect_key_field(json1, json2)
        if not key_field:
            error_msg = "无法找到两个JSON共有的有效匹配字段"
            return json.dumps({
                "error": error_msg,
                "success": False
            }, ensure_ascii=False)

        # 执行合并
        result = _merge_json_lists(json1, json2, key_field)

        # 返回JSON字符串
        return json.dumps(result, ensure_ascii=False, indent=2)

    except json.JSONDecodeError as e:
        error_msg = f"JSON解析失败: {str(e)}"
        return json.dumps({
            "error": error_msg,
            "success": False
        }, ensure_ascii=False)
    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        return json.dumps({
            "error": error_msg,
            "success": False
        }, ensure_ascii=False)


# --- 测试代码 ---
if __name__ == "__main__":
    print("=" * 60)
    print("测试 analyze_tool.py")
    print("=" * 60)

    # 测试1: 从文件读取
    print("\n[测试1] 从 wencai.txt 文件读取并分析...")
    try:
        with open("wencai.txt", 'r', encoding='utf-8') as f:
            content = f.read()

        first_end = content.find(']')
        second_start = content.find('[', first_end + 1)
        second_end = content.rfind(']')

        json1_str = content[:first_end + 1].strip()
        json2_str = content[second_start:second_end + 1].strip()

        # 直接调用函数（不使用tool装饰器的调用方式）
        result_str = merge_json_data.invoke({
            "json1_str": json1_str,
            "json2_str": json2_str
        })
        result = json.loads(result_str)

        if isinstance(result, list):
            print(f"[analyze] 成功: 找到 {len(result)} 条交集记录")
            if result:
                print(f"[analyze] 合并后的列数: {len(result[0].keys())}")

                # 显示前3条记录
                print("\n前3条记录预览:")
                for i, item in enumerate(result[:3], 1):
                    print(f"\n记录 {i}:")
                    for key, value in item.items():
                        if key:  # 跳过空键
                            print(f"  {key}: {value}")

                # 保存结果到文件
                output_file = "analyze_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result_str)
                print(f"\n[analyze] 结果已保存到: {output_file}")

                # 统计信息
                print("\n[analyze] 统计信息:")
                print(f"  交集记录数: {len(result)}")
                all_keys = set()
                for item in result:
                    all_keys.update(item.keys())
                print(f"  合并后总列数: {len(all_keys)}")
                print(f"  列名: {', '.join(sorted([k for k in all_keys if k]))}")
        else:
            print(f"[analyze] 失败: {result.get('error', '未知错误')}")

    except Exception as e:
        print(f"[analyze] 测试失败: {e}")

    print("\n" + "=" * 60)
