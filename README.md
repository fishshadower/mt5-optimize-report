# MT5 优化报告分析工具

把 MT5 遗传优化导出的 XML，自动生成一个交互式 HTML 报告：
- 权重可调的综合评分
- 参数敏感性折线图
- 帕累托前沿解高亮
- 参数范围 & 步长表
- 参数组合排行榜（Top 30，可切换全部）

## 目录结构

```text
project_root/
  parser.py       # 解析 MT5 导出的 XML
  report.py       # 生成 HTML 报告
  analyze.py      # 分析入口（批量/单文件）
  run.command     # macOS 一键运行入口
  run_windows.bat # Windows 一键运行入口
  mt5_xml/        # 你放 MT5 导出的 XML 文件
  reports/        # 自动生成的 HTML 报告
  README.md
```

## 使用方法

- 确保安装python3
- 将MT5生成的xml文件放到mt5_xml文件夹下
- 运行命令需要给予权限
  - Mac：
  - Windows：
- 按自己电脑系统点击一键运行命令

## 注意

默认为批量处理，使用以下命令单独处理某个文件
```
python3 analyze.py /path/to/ReportOptimizer-xxxx.xml
```

