"""
股票数据合并分析工具 - 用于合并两个股票查询结果

工具名称：merge_json_data
功能描述：将两个股票查询结果（JSON列表格式）进行合并，实现类似SQL INNER JOIN的操作。

主要用途：
1. 合并不同维度的股票数据（如成交额数据 + 热度数据）
2. 将多个股票查询结果进行关联分析
3. 对股票数据进行多维度整合
4. 合并成交额排名数据和个股热度数据
5. 合并不同时间段的股票查询结果
6. 整合多个数据源的股票信息

核心功能：
- 自动检测匹配字段（优先使用"股票代码"+"股票简称"组合）
- 只保留两个列表中都存在的股票记录（交集，类似INNER JOIN）
- 自动合并所有列，去重并填充空值
- 自动过滤序号列，只保留业务数据
- 智能去重：如果同一股票出现多次，自动选择数据最完整的记录

关键词：股票数据合并、股票查询结果合并、股票数据分析、INNER JOIN、数据关联、多维度分析
"""
import json
from typing import List, Dict, Any, Optional, Tuple
from langchain.tools import tool  # type: ignore


def _normalize_value(value: Any) -> str:
    """
    标准化值，用于匹配比较

    Args:
        value: 任意值

    Returns:
        标准化后的字符串
    """
    if value is None:
        return ""
    return str(value).strip()


def _detect_key_fields(
    json1: List[Dict[str, Any]],
    json2: List[Dict[str, Any]]
) -> Tuple[List[str], str]:
    """
    自动检测两个股票数据JSON列表共有的、适合作为匹配键的字段

    功能说明：
        自动识别两个股票数据列表中共同的字段，优先使用多个字段的组合
        （如股票代码+股票简称），提高匹配准确性。
        这对于股票数据特别重要，因为可能存在股票代码相同但股票不同的情况。

    Args:
        json1: 第一个股票数据JSON列表（字典列表）
        json2: 第二个股票数据JSON列表（字典列表）

    Returns:
        Tuple[List[str], str]: (匹配字段列表, 匹配模式)
        - 匹配字段列表：用于匹配的字段名列表，如["股票代码", "股票简称"]
        - 匹配模式："multi" 表示多字段组合, "single" 表示单字段

    匹配优先级（从高到低）：
        1. "股票代码" + "股票简称" 组合（最优，最准确）
        2. 其他"代码"字段 + "简称/名称"字段组合
        3. 第一个有效的共有字段（作为备选）

    注意：
        如果找不到有效的匹配字段，返回空列表和"single"模式。
    """
    if not json1 or not json2:
        return [], "single"

    # 获取两个列表的所有字段
    fields1 = set(json1[0].keys()) if json1 else set()
    fields2 = set(json2[0].keys()) if json2 else set()

    # 找到共有的字段
    common_fields = fields1 & fields2

    # 移除空键
    common_fields = {f for f in common_fields if f and f.strip()}

    if not common_fields:
        return [], "single"

    # 优先尝试股票代码+股票简称的组合
    preferred_fields = ["股票代码", "股票简称"]
    if all(field in common_fields for field in preferred_fields):
        # 验证这些字段都有有效值
        if all(_is_valid_key_field(json1, json2, field)
               for field in preferred_fields):
            return preferred_fields, "multi"

    # 尝试其他常见的多字段组合
    # 例如：代码+简称的组合
    code_fields = [
        f for f in common_fields
        if "代码" in f or "code" in f.lower()
    ]
    name_fields = [
        f for f in common_fields
        if "简称" in f or "名称" in f or "name" in f.lower()
    ]

    if code_fields and name_fields:
        combo = [code_fields[0], name_fields[0]]
        if all(_is_valid_key_field(json1, json2, field)
               for field in combo):
            return combo, "multi"

    # 如果多字段组合不可用，选择第一个有效的共有字段
    for field in common_fields:
        if _is_valid_key_field(json1, json2, field):
            return [field], "single"

    return [], "single"


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
        item.get(field) and _normalize_value(item.get(field))
        for item in json1[:10]  # 只检查前10条
    )
    has_value2 = any(
        item.get(field) and _normalize_value(item.get(field))
        for item in json2[:10]  # 只检查前10条
    )

    return has_value1 and has_value2


def _get_match_key(
    item: Dict[str, Any], key_fields: List[str]
) -> Optional[str]:
    """
    从字典中提取匹配键

    Args:
        item: 字典项
        key_fields: 匹配字段列表

    Returns:
        匹配键字符串，如果任何字段缺失或为空则返回None
    """
    key_parts = []
    for field in key_fields:
        value = item.get(field)
        normalized = _normalize_value(value)
        if not normalized:
            return None
        key_parts.append(normalized)

    # 使用特殊分隔符组合多个字段
    return "|||".join(key_parts)


def _count_non_empty_fields(item: Dict[str, Any]) -> int:
    """
    计算字典中非空字段的数量

    Args:
        item: 字典项

    Returns:
        非空字段数量
    """
    return sum(
        1 for v in item.values()
        if v and (not isinstance(v, str) or _normalize_value(v))
    )


def _merge_json_lists(
    json1: List[Dict[str, Any]],
    json2: List[Dict[str, Any]],
    key_fields: List[str]
) -> List[Dict[str, Any]]:
    """
    对两个股票数据JSON列表取交集并合并列（内部函数）

    功能说明：
        实现类似SQL INNER JOIN的操作，只保留两个列表中都存在的股票记录。
        对于重复的匹配键，自动选择数据最完整的记录进行合并。
        这是合并工具的核心逻辑函数。

    Args:
        json1: 第一个股票数据JSON列表（字典列表）
        json2: 第二个股票数据JSON列表（字典列表）
        key_fields: 用于匹配的键字段名列表，如["股票代码", "股票简称"]

    Returns:
        List[Dict[str, Any]]: 合并后的JSON列表，只包含两个列表中都存在的股票记录
            （基于匹配键）。结果会自动过滤掉"序号"列，只保留业务数据。

    合并逻辑（详细步骤）：
        1. 对两个列表分别去重：
           - 如果同一股票（相同匹配键）出现多次，选择数据最完整的记录
           - 数据完整性通过非空字段数量来判断
        2. 基于匹配键找到交集：
           - 只保留在两个列表中都存在的股票记录
           - 如果某个股票只在一个列表中存在，则不会出现在结果中
        3. 合并所有列：
           - 优先保留第一个列表的值
           - 如果第一个列表的值为空，用第二个列表的值填充
           - 如果列名不同，则全部合并到结果中
        4. 自动过滤：
           - 过滤掉"序号"列，只保留业务数据
           - 过滤掉空键（键名为空字符串的字段）
    """
    # 将第二个列表转换为以匹配键为键的字典，方便查找
    # 如果有重复的key，选择数据最完整的那条（非空字段最多的）
    json2_dict = {}
    for item in json2:
        match_key = _get_match_key(item, key_fields)
        if match_key:
            # 如果已存在，比较数据完整性（非空字段数量）
            if match_key in json2_dict:
                existing_item = json2_dict[match_key]
                if _count_non_empty_fields(item) > _count_non_empty_fields(
                    existing_item
                ):
                    json2_dict[match_key] = item
            else:
                json2_dict[match_key] = item

    # 将第一个列表也转换为字典，如果有重复的key，选择数据最完整的那条
    json1_dict = {}
    for item in json1:
        match_key = _get_match_key(item, key_fields)
        if match_key:
            if match_key in json1_dict:
                existing_item = json1_dict[match_key]
                if _count_non_empty_fields(item) > _count_non_empty_fields(
                    existing_item
                ):
                    json1_dict[match_key] = item
            else:
                json1_dict[match_key] = item

    # 遍历第一个列表的去重结果，找到交集并合并
    result = []
    for match_key, item1 in json1_dict.items():
        # 如果在第二个列表中找到匹配项
        if match_key in json2_dict:
            item2 = json2_dict[match_key]

            # 合并两个字典，去重列
            # 策略：优先保留第一个字典的值，如果第一个字典的值为空，
            # 则用第二个字典的值填充
            merged_item = {}

            # 先添加第一个字典的所有字段（跳过序号和空键）
            for k, v in item1.items():
                # 跳过空键和序号列
                if k and k.strip() and k != "序号":
                    merged_item[k] = v

            # 再添加第二个字典的字段
            for k, v in item2.items():
                # 跳过空键和序号列
                if not k or not k.strip() or k == "序号":
                    continue

                # 如果键不存在，直接添加
                if k not in merged_item:
                    merged_item[k] = v
                # 如果键已存在但值为空（或只有空白字符），则用新值替换
                elif not _normalize_value(merged_item[k]):
                    merged_item[k] = v
                # 如果键已存在且有值，保留原值（不覆盖）

            result.append(merged_item)

    return result


@tool
def merge_json_data(json1_str: str, json2_str: str) -> str:
    """
    股票数据合并工具：将两个股票查询结果（JSON列表）进行合并分析

    功能说明：
        本工具用于合并两个股票数据的JSON列表，自动识别匹配字段（如股票代码、股票简称），
        只保留两个列表中都存在的股票记录（类似SQL INNER JOIN），并将所有列合并在一起。
        这是股票数据分析中常用的工具，用于整合不同维度的股票信息。

    适用场景（何时使用此工具）：
        - 需要合并两个股票查询结果时
        - 需要将成交额数据和热度数据进行关联分析时
        - 需要合并不同维度的股票查询结果时
        - 需要整合多个数据源的股票信息时
        - 需要对股票数据进行多维度关联分析时
        - 需要将不同时间段的股票查询结果进行合并时

    输入参数：
        json1_str (str): 第一个股票数据JSON字符串，必须是数组格式（列表）。
            必须包含股票标识字段（如"股票代码"、"股票简称"）。
            示例：
            '[{"股票代码": "300308", "股票简称": "中际旭创", '
            '"成交额(元) 2025.11.27": "236.41亿", '
            '"成交额排名 2025.11.27": "1/5454"}, ...]'
        json2_str (str): 第二个股票数据JSON字符串，必须是数组格式（列表）。
            必须包含与第一个列表相同的股票标识字段（如"股票代码"、"股票简称"）。
            示例：
            '[{"股票代码": "300308", "股票简称": "中际旭创", '
            '"个股热度排名 2025.11.27": "47", '
            '"个股热度 2025.11.27": "12.31万"}, ...]'

    返回结果：
        str: 合并后的JSON字符串（数组格式），包含两个列表的交集记录
            （只包含同时出现在两个列表中的股票），
            所有列已去重合并。结果会自动过滤掉"序号"列，只保留业务数据。

    合并策略和规则：
        1. 自动检测匹配字段：
           - 优先使用"股票代码"+"股票简称"的组合（提高匹配准确性）
           - 如果不存在，则自动检测其他共有字段作为匹配键
        2. 交集合并（INNER JOIN）：
           - 只返回两个列表中都存在的股票记录
           - 如果某个股票只在一个列表中存在，则不会出现在结果中
        3. 列合并规则：
           - 如果两个列表有相同列名，优先保留第一个列表的值
           - 如果第一个列表的值为空，则用第二个列表的值填充
           - 如果列名不同，则全部合并到结果中
        4. 数据去重：
           - 如果同一股票（相同匹配键）出现多次，自动选择数据最完整的记录
           - 数据完整性通过非空字段数量来判断

    完整使用示例：
        场景：合并成交额数据和热度数据

        输入 json1_str:
            '[{"股票代码": "300308", "股票简称": "中际旭创", '
            '"成交额(元) 2025.11.27": "236.41亿", '
            '"成交额排名 2025.11.27": "1/5454", '
            '"成交量(股) 2025.11.27": "4,373.28万"}]'

        输入 json2_str:
            '[{"股票代码": "300308", "股票简称": "中际旭创", '
            '"个股热度排名 2025.11.27": "47", '
            '"个股热度 2025.11.27": "12.31万"}]'

        输出结果:
            '[{"股票代码": "300308", "股票简称": "中际旭创", '
            '"成交额(元) 2025.11.27": "236.41亿", '
            '"成交额排名 2025.11.27": "1/5454", '
            '"成交量(股) 2025.11.27": "4,373.28万", '
            '"个股热度排名 2025.11.27": "47", '
            '"个股热度 2025.11.27": "12.31万"}]'

    重要注意事项：
        - 两个JSON必须是数组格式（列表），不能是单个对象
        - 会自动检测共有的字段作为匹配键（优先使用"股票代码"+"股票简称"的组合）
        - 只返回两个列表中都存在的记录（基于匹配键，类似SQL INNER JOIN）
        - 如果匹配键字段为空或不存在，该记录会被跳过
        - 结果中会自动过滤掉"序号"列，只保留业务数据
        - 支持多字段组合匹配，提高匹配准确性，避免仅用股票代码可能出现的误匹配
        - 如果输入数据格式错误，会返回包含error字段的JSON对象

    关键词（帮助AI识别使用场景）：
        股票数据合并、股票查询结果合并、股票数据分析、INNER JOIN、数据关联、
        多维度分析、成交额数据、热度数据、股票信息整合、数据去重、交集合并
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

        # 自动检测匹配键字段（支持多字段组合）
        key_fields, match_mode = _detect_key_fields(json1, json2)
        if not key_fields:
            error_msg = "无法找到两个JSON共有的有效匹配字段"
            return json.dumps({
                "error": error_msg,
                "success": False
            }, ensure_ascii=False)

        # 执行合并
        result = _merge_json_lists(json1, json2, key_fields)

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
                output_file = "wencai_result.txt"
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
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
