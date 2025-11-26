import json
import os
from typing import List, Dict, Any, Union
from langchain.tools import tool  # type: ignore
import pandas as pd  # type: ignore


def _read_excel_impl(filename: str) -> str:
    """
    读取Excel文件并返回数据数组。

    Args:
        filename (str): Excel文件的路径（支持 .xlsx 和 .xls 格式）
            - 如果是绝对路径，则从指定路径读取
            - 如果是相对路径，则从桌面路径 ~/Desktop 读取

    Returns:
        str: JSON格式的字符串，包含读取的数据数组。
            如果文件不存在或读取失败，返回错误信息的JSON字符串。

    示例:
        read_excel("data.xlsx") -> '[{"列1": "值1", "列2": "值2"}, ...]'
    """
    try:
        print(f"[tool] 输入 文件: {filename}")
        # 处理文件路径：如果是相对路径，从桌面读取
        if not os.path.isabs(filename):
            desktop_path = "~/Desktop"
            filename = os.path.join(desktop_path, filename)

        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"[tool] 错误 文件不存在: {filename}")
            return json.dumps({
                "error": f"文件不存在: {filename}",
                "data": []
            }, ensure_ascii=False)

        # 读取Excel文件
        print("[tool] 步骤 读取Excel文件")
        df = pd.read_excel(filename)

        # 将DataFrame转换为字典列表
        data = df.to_dict('records')

        # 处理NaN值，将其转换为None（JSON中的null）
        for record in data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        result = {
            "success": True,
            "filename": filename,
            "row_count": len(data),
            "column_count": len(df.columns) if len(data) > 0 else 0,
            "columns": list(df.columns.tolist()) if len(data) > 0 else [],
            "data": data
        }

        print(
            f"[tool] 输出 成功读取: {len(data)} 行, "
            f"{len(df.columns)} 列"
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        print(f"[tool] 错误 {e}")
        error_result = {
            "error": str(e),
            "success": False,
            "filename": filename,
            "data": []
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def read_excel(filename: str) -> str:
    """
    读取Excel文件并返回数据数组。

    Args:
        filename (str): Excel文件的路径（支持 .xlsx 和 .xls 格式）
            - 如果是绝对路径，则从指定路径读取
            - 如果是相对路径，则从桌面路径 ~/Desktop 读取

    Returns:
        str: JSON格式的字符串，包含读取的数据数组。
            如果文件不存在或读取失败，返回错误信息的JSON字符串。

    示例:
        read_excel("data.xlsx") -> '[{"列1": "值1", "列2": "值2"}, ...]'
    """
    return _read_excel_impl(filename)


def _write_excel_impl(
    filename: str,
    data: Union[str, List[Dict[str, Any]]]
) -> str:
    """
    将数据写入Excel文件。

    Args:
        filename (str): 要写入的Excel文件路径（会自动创建 .xlsx 格式）
            - 如果是绝对路径，则写入到指定路径
            - 如果是相对路径，则写入到桌面路径 ~/Desktop
        data (str | List[Dict]): 要写入的数据。
            可以是JSON字符串或字典列表。
            如果是JSON字符串，会自动解析为字典列表。

    Returns:
        str: JSON格式的字符串，包含写入结果信息。

    示例:
        write_excel("output.xlsx", [{"列1": "值1", "列2": "值2"}])
        write_excel("/tmp/output.xlsx", [{"列1": "值1", "列2": "值2"}])
    """
    try:
        print(f"[tool] 输入 文件: {filename}")
        data_count = (
            len(data) if isinstance(data, list) else 'N/A'
        )
        print(f"[tool] 输入 数据行数: {data_count}")
        # 解析数据
        if isinstance(data, str):
            try:
                data = json.loads(data)
                print(f"[tool] 步骤 解析JSON数据: {len(data)} 行")
            except json.JSONDecodeError:
                print("[tool] 错误 无法解析JSON字符串")
                return json.dumps({
                    "error": "数据格式错误：无法解析JSON字符串",
                    "success": False,
                    "filename": filename
                }, ensure_ascii=False)

        if not isinstance(data, list):
            return json.dumps({
                "error": "数据格式错误：数据必须是列表或JSON字符串",
                "success": False,
                "filename": filename
            }, ensure_ascii=False)

        if len(data) == 0:
            return json.dumps({
                "error": "数据为空：无法写入空数据",
                "success": False,
                "filename": filename
            }, ensure_ascii=False)

        # 确保所有元素都是字典
        if not all(isinstance(item, dict) for item in data):
            return json.dumps({
                "error": "数据格式错误：列表中的每个元素必须是字典",
                "success": False,
                "filename": filename
            }, ensure_ascii=False)

        # 转换为DataFrame
        df = pd.DataFrame(data)

        # 处理文件路径：如果是相对路径，写入到桌面
        if not os.path.isabs(filename):
            desktop_path = "~/Desktop"
            filename = os.path.join(desktop_path, filename)

        # 确保文件扩展名是 .xlsx
        if not filename.endswith('.xlsx'):
            if filename.endswith('.xls'):
                filename = filename[:-4] + '.xlsx'
            else:
                filename = filename + '.xlsx'

        # 创建目录（如果不存在）
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 写入Excel文件
        print("[tool] 步骤 写入Excel文件")
        df.to_excel(filename, index=False, engine='openpyxl')

        result = {
            "success": True,
            "filename": filename,
            "row_count": len(data),
            "column_count": len(df.columns),
            "columns": list(df.columns.tolist()),
            "message": f"成功写入 {len(data)} 行数据到 {filename}"
        }

        print(
            f"[tool] 输出 成功写入: {len(data)} 行, "
            f"{len(df.columns)} 列到 {filename}"
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    except Exception as e:
        print(f"[tool] 错误 {e}")
        error_result = {
            "error": str(e),
            "success": False,
            "filename": filename
        }
        return json.dumps(error_result, ensure_ascii=False)


@tool
def write_excel(
    filename: str,
    data: Union[str, List[Dict[str, Any]]]
) -> str:
    """
    将数据写入Excel文件。

    Args:
        filename (str): 要写入的Excel文件路径（会自动创建 .xlsx 格式）
            - 如果是绝对路径，则写入到指定路径
            - 如果是相对路径，则写入到桌面路径 ~/Desktop
        data (str | List[Dict]): 要写入的数据。
            可以是JSON字符串或字典列表。
            如果是JSON字符串，会自动解析为字典列表。

    Returns:
        str: JSON格式的字符串，包含写入结果信息。

    示例:
        write_excel("output.xlsx", [{"列1": "值1", "列2": "值2"}])
        write_excel("/tmp/output.xlsx", [{"列1": "值1", "列2": "值2"}])
    """
    return _write_excel_impl(filename, data)


# --- 测试代码 ---
if __name__ == "__main__":
    print("=" * 50)
    print("测试 Excel 工具")
    print("=" * 50)

    # 创建测试文件（使用当前目录）
    test_file = "test_excel_tools.xlsx"

    # 测试数据
    test_data = [
        {"姓名": "张三", "年龄": 25, "城市": "北京"},
        {"姓名": "李四", "年龄": 30, "城市": "上海"},
        {"姓名": "王五", "年龄": 28, "城市": "广州"}
    ]

    print("\n1. 测试写入Excel文件...")
    print(f"   写入文件: {test_file}")
    write_result = _write_excel_impl(test_file, test_data)
    write_result_dict = json.loads(write_result)
    print(f"   写入结果: {write_result_dict.get('success', False)}")
    if write_result_dict.get('success'):
        print(f"   写入行数: {write_result_dict.get('row_count', 0)}")
        print(f"   列数: {write_result_dict.get('column_count', 0)}")
        print(f"   列名: {write_result_dict.get('columns', [])}")
    else:
        print(f"   错误: {write_result_dict.get('error', '未知错误')}")

    print("\n2. 测试读取Excel文件...")
    print(f"   读取文件: {test_file}")
    read_result = _read_excel_impl(test_file)
    read_result_dict = json.loads(read_result)
    print(f"   读取结果: {read_result_dict.get('success', False)}")
    if read_result_dict.get('success'):
        print(f"   读取行数: {read_result_dict.get('row_count', 0)}")
        print(f"   列数: {read_result_dict.get('column_count', 0)}")
        print(f"   列名: {read_result_dict.get('columns', [])}")
        print("   前3行数据:")
        data = read_result_dict.get('data', [])
        for i, row in enumerate(data[:3], 1):
            print(f"     行{i}: {row}")
    else:
        print(f"   错误: {read_result_dict.get('error', '未知错误')}")

    print("\n3. 测试JSON字符串输入...")
    json_data = json.dumps(test_data, ensure_ascii=False)
    test_file2 = "test_excel_tools2.xlsx"
    write_result2 = _write_excel_impl(test_file2, json_data)
    write_result_dict2 = json.loads(write_result2)
    print(f"   JSON字符串写入结果: {write_result_dict2.get('success', False)}")

    print("\n4. 测试错误处理...")
    # 测试读取不存在的文件
    read_error = _read_excel_impl("不存在的文件.xlsx")
    read_error_dict = json.loads(read_error)
    print(f"   读取不存在文件: {read_error_dict.get('error', '无错误')}")

    # 测试写入空数据
    write_error = _write_excel_impl("empty.xlsx", [])
    write_error_dict = json.loads(write_error)
    print(f"   写入空数据: {write_error_dict.get('error', '无错误')}")

    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
