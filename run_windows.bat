@echo off
REM 放在项目根目录，双击运行 = 批量处理 mt5_xml 下所有 XML
REM 也可以把单个 XML 文件拖到这个 bat 上 = 只处理这个文件

cd /d "%~dp0"

echo 📂 运行 MT5 优化报告生成器
echo.

REM 如果有参数（拖拽 XML 到 bat 图标）
IF NOT "%~1"=="" (
    echo 🎯 单文件模式: %~1
    python analyze.py "%~1"
) ELSE (
    echo 📦 批量模式：扫描 .\mt5_xml 目录下所有 .xml
    python analyze.py
)

echo.
echo ✅ 处理结束，如果没有报错，HTML 报告在 .\reports 目录下。
pause
