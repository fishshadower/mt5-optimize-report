# parser.py
import xml.etree.ElementTree as ET
import pandas as pd

# Excel XML 命名空间
NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}


def try_number(value: str):
    """尝试把字符串转成数字，否则原样返回"""
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        f = float(value)
        if f.is_integer():
            return int(f)
        return f
    except Exception:
        return value


def parse_xml(path: str):
    """
    解析 MT5 优化导出的 XML（Excel 2003 XML 格式）
    返回: df, param_cols, metric_cols
    """
    tree = ET.parse(path)
    root = tree.getroot()

    ws = root.find(".//ss:Worksheet", NS)
    if ws is None:
        raise RuntimeError("XML 中未找到 Worksheet 节点")

    table = ws.find("ss:Table", NS)
    if table is None:
        raise RuntimeError("Worksheet 中未找到 Table 节点")

    rows = table.findall("ss:Row", NS)
    if len(rows) < 2:
        raise RuntimeError("Table 行数不足，至少需要表头 + 一行数据")

    # 表头
    headers = []
    header_cells = rows[0].findall("ss:Cell", NS)
    for c in header_cells:
        data = c.find("ss:Data", NS)
        headers.append(data.text if data is not None else None)

    # 数据行
    records = []
    for row in rows[1:]:
        cells = row.findall("ss:Cell", NS)
        if not cells:
            continue

        values = []
        for cell in cells:
            data = cell.find("ss:Data", NS)
            text = data.text if data is not None else None
            values.append(try_number(text))

        if len(values) < len(headers):
            values += [None] * (len(headers) - len(values))

        records.append(dict(zip(headers, values)))

    df = pd.DataFrame(records)

    # 参数列：以 inp 开头
    param_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("inp")]
    # 指标列：其他（排除 Custom）
    metric_cols = [c for c in df.columns if c not in param_cols and c != "Custom"]

    return df, param_cols, metric_cols
