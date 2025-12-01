# analyze.py
import sys
from pathlib import Path
from parser import parse_xml
from report import generate_report


def process_one(xml_path: Path, reports_dir: Path):
    """处理单个 XML，生成对应 HTML 报告"""
    if not xml_path.exists():
        print(f"❌ 文件不存在: {xml_path}")
        return

    out_html = reports_dir / f"{xml_path.stem}.html"
    if out_html.exists():
        print(f"⏩ 已存在报告，跳过: {xml_path.name}")
        return

    print(f"⏳ 正在处理: {xml_path.name}")
    df, param_cols, metric_cols = parse_xml(str(xml_path))
    generate_report(df, param_cols, metric_cols, str(out_html), file_name=xml_path.name)


def main():
    base_dir = Path(__file__).parent

    # 输入 XML 文件夹：把 MT5 导出的 XML 丢这里
    input_dir = base_dir / "mt5_xml"

    # 输出 HTML 报告文件夹
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 如果命令行带了参数：兼容单文件模式
    if len(sys.argv) >= 2:
        xml_path = Path(sys.argv[1])
        process_one(xml_path, reports_dir)
        return

    # 不带参数：批量模式
    if not input_dir.exists():
        print(f"⚠ 未找到输入目录: {input_dir}")
        print("请先在项目目录下创建 mt5_xml 文件夹，并把 XML 文件放进去。")
        return

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        print(f"⚠ {input_dir} 中没有找到任何 .xml 文件。")
        return

    print(f"📂 批量模式：扫描 {input_dir}，共发现 {len(xml_files)} 个 XML。")
    for xml_path in xml_files:
        process_one(xml_path, reports_dir)

    print("✅ 全部处理完成。报告已生成在:", reports_dir)


if __name__ == "__main__":
    main()
