import asyncio
from playwright.async_api import async_playwright  # type: ignore
import json
import time
from langchain.tools import tool  # type: ignore


async def fetch_iwencai_data(query: str, max_rows: int = 100) -> str:
    """
    异步获取同花顺问财页面表格数据。

    Args:
        query (str): 查询字符串，例如 "成交额前100的股票"。
        max_rows (int): 最大返回行数，默认为100。

    Returns:
        str: 包含表格数据的JSON字符串。
    """
    url = (
        f"https://www.iwencai.com/unifiedwap/result?"
        f"w={query}&querytype=stock"
    )

    all_data = []  # 存储所有提取的数据
    headers = None  # 存储表头，只在第一页获取一次

    async with async_playwright() as p:
        # 启动浏览器，设置更真实的浏览器标识
        browser = await p.chromium.launch(
            headless=True,  # 使用headless模式
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage'
            ]
        )
        context = await browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1920, 'height': 1080},
        )
        page = await context.new_page()

        # 移除webdriver标识
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        try:
            print(f"[tool] 输入 查询: {query}")
            print(f"[tool] 输入 URL: {url}")
            await page.goto(url, wait_until="networkidle", timeout=5000)

            # 等待表格数据出现 - 使用新的类选择器
            rows_found = False
            # 使用新的表格内容类选择器
            table_body_selector = (
                '.iwc-table-body.scroll-style2.big-mark tr, '
                '[class*="iwc-table-body"] tr'
            )

            for i in range(10):  # 最多尝试10次，每次0.5秒
                await page.wait_for_timeout(500)
                rows = await page.query_selector_all(table_body_selector)
                if len(rows) > 0:
                    rows_found = True
                    print(f"[tool] 步骤 找到 {len(rows)} 行数据")
                    break

            if not rows_found:
                # 再等待一下，可能数据还在加载
                await page.wait_for_timeout(2000)
                rows = await page.query_selector_all(table_body_selector)
                if len(rows) > 0:
                    rows_found = True
                    print(f"[tool] 步骤 找到 {len(rows)} 行数据")

            # --- 先提取表头 ---
            if headers is None:
                # 策略：先找 class="iwc-table-header" 的列，
                # 再找 class="iwc-table-header table-right-thead
                # scroll-style2" 的列
                headers = []

                # 第一部分：提取所有 class="iwc-table-header" 的列（强匹配）
                # 使用 query_selector_all 获取所有匹配的元素
                header_selector1 = '.iwc-table-header'
                header_elements1 = await page.query_selector_all(
                    header_selector1
                )

                for header_element1 in header_elements1:
                    # 检查是否完全匹配 class="iwc-table-header"（不包含其他类）
                    class_attr = await header_element1.get_attribute('class')
                    if class_attr:
                        # 将 class 属性按空格分割，检查是否只有 "iwc-table-header"
                        class_list = class_attr.strip().split()
                        # 强匹配：只包含 "iwc-table-header"，不包含其他类
                        if (len(class_list) == 1 and
                                class_list[0] == 'iwc-table-header'):
                            # 先尝试从 ul, li 中提取
                            header_ul = (
                                await header_element1.query_selector('ul')
                            )
                            if header_ul:
                                header_lis = (
                                    await header_ul.query_selector_all('li')
                                )
                                for li in header_lis:
                                    text = await li.inner_text()
                                    clean_text = ' '.join(text.split()).strip()
                                    if clean_text:
                                        headers.append(clean_text)
                                    else:
                                        headers.append('')
                            else:
                                # 如果没有ul/li，尝试从th/td中提取
                                header_cells1 = (
                                    await header_element1.query_selector_all(
                                        'th, td'
                                    )
                                )
                                if len(header_cells1) == 0:
                                    # 如果没有th/td，尝试从tr中提取
                                    header_rows1 = (
                                        await header_element1
                                        .query_selector_all('tr')
                                    )
                                    if header_rows1:
                                        header_cells1 = (
                                            header_rows1[0]
                                            .query_selector_all('th, td')
                                        )

                                for cell in header_cells1:
                                    text = await cell.inner_text()
                                    clean_text = ' '.join(text.split()).strip()
                                    if clean_text:
                                        headers.append(clean_text)
                                    else:
                                        headers.append('')

                # 第二部分：提取所有 class="iwc-table-header
                # table-right-thead scroll-style2" 的列（强匹配）
                header_selector2 = (
                    '.iwc-table-header.table-right-thead.scroll-style2'
                )
                header_elements2 = await page.query_selector_all(
                    header_selector2
                )

                for header_element2 in header_elements2:
                    # 检查是否完全匹配
                    # class="iwc-table-header table-right-thead scroll-style2"
                    class_attr2 = await header_element2.get_attribute('class')
                    if class_attr2:
                        class_list2 = class_attr2.strip().split()
                        # 强匹配：必须包含这三个类，且只有这三个类
                        expected_classes = {
                            'iwc-table-header',
                            'table-right-thead',
                            'scroll-style2'
                        }
                        if (len(class_list2) == 3 and
                                set(class_list2) == expected_classes):
                            # 先尝试从 ul, li 中提取
                            header_ul2 = (
                                await header_element2.query_selector('ul')
                            )
                            if header_ul2:
                                header_lis2 = (
                                    await header_ul2.query_selector_all('li')
                                )
                                for li in header_lis2:
                                    text = await li.inner_text()
                                    clean_text = ' '.join(text.split()).strip()
                                    if clean_text:
                                        headers.append(clean_text)
                                    else:
                                        headers.append('')
                            else:
                                # 如果没有ul/li，尝试从th/td中提取
                                header_cells2 = (
                                    await header_element2
                                    .query_selector_all('th, td')
                                )
                                if len(header_cells2) == 0:
                                    # 如果没有th/td，尝试从tr中提取
                                    header_rows2 = (
                                        await header_element2
                                        .query_selector_all('tr')
                                    )
                                    if header_rows2:
                                        header_cells2 = (
                                            header_rows2[0].query_selector_all(
                                                'th, td'
                                            )
                                        )

                                for cell in header_cells2:
                                    text = await cell.inner_text()
                                    clean_text = ' '.join(text.split()).strip()
                                    if clean_text:
                                        headers.append(clean_text)
                                    else:
                                        headers.append('')

                if headers:
                    print(f"[tool] 步骤 提取表头: {len(headers)} 列")
                else:
                    # 如果表头提取失败，从第一行数据获取列数，使用通用列名
                    print("[tool] 步骤 使用通用列名")
                    table_body_selector = (
                        '.iwc-table-body.scroll-style2.big-mark tr, '
                        '[class*="iwc-table-body"] tr'
                    )
                    first_row = await page.query_selector(
                        table_body_selector
                    )
                    if first_row:
                        first_cells = (
                            await first_row.query_selector_all('td')
                        )
                        if first_cells:
                            # 使用通用列名
                            headers = [
                                f"列{i+1}"
                                for i in range(len(first_cells))
                            ]
                            print(f"[tool] 步骤 通用列名: {len(headers)} 列")
                    else:
                        print("[tool] 错误 无法获取表头或数据行")
                        return json.dumps(
                            all_data, ensure_ascii=False, indent=2
                        )

            # --- 再提取内容 ---
            while len(all_data) < max_rows:
                # 等待当前页数据稳定加载 - 使用轮询方式
                table_body_selector = (
                    '.iwc-table-body.scroll-style2.big-mark tr, '
                    '[class*="iwc-table-body"] tr'
                )
                for i in range(5):  # 最多尝试5次，每次0.5秒
                    await page.wait_for_timeout(500)
                    rows_check = await page.query_selector_all(
                        table_body_selector
                    )
                    if len(rows_check) > 0:
                        first_row = rows_check[0]
                        first_cell = await first_row.query_selector('td')
                        if first_cell:
                            first_text = await first_cell.inner_text()
                            if first_text.strip():
                                break

                # --- 提取当前页所有数据行 ---
                # 使用新的表格内容类选择器
                table_body_selector = (
                    '.iwc-table-body.scroll-style2.big-mark tr, '
                    '[class*="iwc-table-body"] tr'
                )
                rows = await page.query_selector_all(table_body_selector)

                if not rows:
                    print("[tool] 步骤 未找到数据行，退出")
                    break

                # --- 提取当前页数据行 ---
                current_page_data = []
                for row in rows:
                    cells = await row.query_selector_all('td')
                    cell_data = []
                    for cell in cells:
                        text_content = await cell.inner_text()
                        clean_text = ' '.join(text_content.split())
                        cell_data.append(clean_text)

                    # 如果列数匹配表头，则认为是有效数据行
                    if headers and len(cell_data) == len(headers):
                        row_dict = dict(zip(headers, cell_data))
                        # 检查是否已存在（避免重复）
                        if row_dict not in current_page_data:
                            current_page_data.append(row_dict)
                    elif headers and len(cell_data) > 0:
                        # 如果列数不匹配但大于0，尝试截断或填充
                        if len(cell_data) > len(headers):
                            cell_data = cell_data[:len(headers)]
                        elif len(cell_data) < len(headers):
                            cell_data.extend(
                                [''] * (len(headers) - len(cell_data))
                            )
                        row_dict = dict(zip(headers, cell_data))
                        if row_dict not in current_page_data:
                            current_page_data.append(row_dict)

                # 添加当前页数据到总数据中
                rows_needed = max_rows - len(all_data)
                rows_to_add = min(rows_needed, len(current_page_data))
                added_before = len(all_data)
                all_data.extend(current_page_data[:rows_to_add])
                added_count = len(all_data) - added_before
                print(
                    f"[tool] 步骤 添加 {added_count} 行，"
                    f"累计 {len(all_data)} 行"
                )

                # --- 检查并处理分页 ---
                if len(all_data) >= max_rows:
                    print("[tool] 步骤 达到最大行数，停止翻页")
                    break

                # --- 查找并点击下一页 ---
                next_button = page.locator(
                    'button:has-text("下页"), '
                    'button:has-text("下一页"), '
                    '[role="button"]:has-text("下页"), '
                    '[role="button"]:has-text("下一页")'
                )

                # 检查按钮是否存在且可点击
                button_count = await next_button.count()
                if button_count > 0:
                    next_button = next_button.first()
                    is_visible = await next_button.is_visible()
                    is_enabled = await next_button.is_enabled()

                    if is_visible and is_enabled:
                        print("[tool] 步骤 翻页")

                        # 记录翻页前的状态
                        old_row_count = len(rows)
                        old_first_row_text = ""
                        if rows:
                            first_cell = await rows[0].query_selector('td')
                            if first_cell:
                                old_first_row_text = (
                                    await first_cell.inner_text()
                                )

                        # 点击按钮
                        await next_button.click()

                        # --- 等待新页面数据加载 ---
                        try:
                            await page.wait_for_function("""
                                ([oldCount, oldFirstText]) => {
                                    const tableBody = document.querySelector(
                                        '.iwc-table-body.scroll-style2.big-mark'
                                    ) || document.querySelector(
                                        '[class*="iwc-table-body"]'
                                    );
                                    if (!tableBody) return false;
                                    const newRows = (
                                        tableBody.querySelectorAll('tr')
                                    );
                                    const newRowCount = newRows.length;

                                    // 检查第一行内容是否变化
                                    if (newRowCount > 0) {
                                        const firstRow = newRows[0];
                                        const firstCell = firstRow ?
                                            firstRow.querySelector('td') :
                                            null;
                                        const newFirstText = firstCell ?
                                            firstCell.innerText.trim() : '';

                                        // 如果第一行内容变了，说明新数据已加载
                                        if (newFirstText !== oldFirstText &&
                                            newFirstText !== '') {
                                            return true;
                                        }
                                    }

                                    return false; // 继续等待
                                }
                            """, [old_row_count, old_first_row_text],
                                timeout=5000)

                            # 再短暂等待一下，确保完全加载
                            await page.wait_for_timeout(500)

                        except asyncio.TimeoutError:
                            print("[tool] 步骤 翻页超时，可能已到最后一页")
                            break  # 超时则退出循环
                    else:
                        print("[tool] 步骤 已到最后一页")
                    break
                else:
                    print("[tool] 步骤 已到最后一页")
                    break

        except Exception as e:
            print(f"[tool] 错误 {e}")
            import traceback
            traceback.print_exc()
        finally:
            await context.close()
            await browser.close()

    # 返回JSON格式的字符串
    result = json.dumps(all_data, ensure_ascii=False, indent=2)
    print(f"[tool] 输出 返回 {len(all_data)} 行数据")
    return result


# --- LangChain 工具封装 ---
@tool
def get_iwencai_stock_data(query: str) -> str:
    """
    调用同花顺问财网站，根据自然语言查询获取股票表格数据。
    最多返回100条记录。

    Args:
        query (str): 自然语言查询，例如 "成交额前100的股票" 或
            "热度前100的股票"。

    Returns:
        str: 包含查询结果的JSON字符串。
    """
    return asyncio.run(fetch_iwencai_data(query))


# --- 测试案例 ---
if __name__ == "__main__":
    async def main():
        print("--- 测试案例 1: 成交额前100的股票 ---")
        start_time = time.time()
        result1 = await fetch_iwencai_data("成交额前100的股票", max_rows=100)
        end_time = time.time()
        print(f"耗时: {end_time - start_time:.2f} 秒")
        print("结果预览 (前3行):")
        data1 = json.loads(result1)
        print(json.dumps(data1[:3], indent=2, ensure_ascii=False))
        print(f"总共获取到 {len(data1)} 行数据。")
        print("-" * 30)

        print("\n--- 测试案例 2: 同花顺热度前100的股票 ---")
        start_time = time.time()
        result2 = await fetch_iwencai_data(
            "热度前100的股票", max_rows=100
        )
        end_time = time.time()
        print(f"耗时: {end_time - start_time:.2f} 秒")
        print("结果预览 (前3行):")
        data2 = json.loads(result2)
        print(json.dumps(data2[:3], indent=2, ensure_ascii=False))
        print(f"总共获取到 {len(data2)} 行数据。")
        print("-" * 30)

    # 运行测试
    asyncio.run(main())
