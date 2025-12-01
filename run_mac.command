#!/bin/bash
# 放在项目根目录，双击运行即可批量处理 mt5_xml 下的 XML

cd "$(dirname "$0")"

echo "📂 运行 MT5 优化报告生成器"
echo

# 不带参数：走批量模式
python3 analyze.py

echo
echo "✅ 处理结束。如果没有报错，HTML 报告在 ./reports 目录下。"
read -p "按回车关闭窗口..." _
