# report.py
import pandas as pd
import numpy as np
import json
from jinja2 import Template
from datetime import datetime

# 英文列名 -> 中文显示名
DISPLAY_NAME_MAP = {
    "Pass": "通过",
    "Result": "余额",
    "Profit": "利润",
    "Expected Payoff": "预期收益",
    "Profit Factor": "盈利因子",
    "Recovery Factor": "采收率",
    "Sharpe Ratio": "夏普比率",
    "Equity DD %": "回撤",
    "Trades": "交易次数",
}

# 默认权重（你那一套）
DEFAULT_WEIGHTS = {
    "profit": 0.30,           # 净利润
    "drawdown": -0.25,        # 最大回撤（负权重）
    "sharpe_ratio": 0.20,     # 夏普比率
    "profit_factor": 0.10,    # 盈利因子
    "recovery_factor": 0.10,  # 采收率
    "expected_payoff": 0.05,  # 预期收益
}

# 权重 key -> 中文名
WEIGHT_LABELS = {
    "profit": "利润",
    "drawdown": "最大回撤",
    "sharpe_ratio": "夏普比率",
    "profit_factor": "盈利因子",
    "recovery_factor": "采收率",
    "expected_payoff": "预期收益",
}

# 指标 key -> (原始列名, z列名, 中文名称)
METRIC_DEF = {
    "profit": ("Profit", "z_profit", "利润"),
    "drawdown": ("Equity DD %", "z_drawdown", "回撤"),
    "sharpe_ratio": ("Sharpe Ratio", "z_sharpe_ratio", "夏普比率"),
    "profit_factor": ("Profit Factor", "z_profit_factor", "盈利因子"),
    "recovery_factor": ("Recovery Factor", "z_recovery_factor", "采收率"),
    "expected_payoff": ("Expected Payoff", "z_expected_payoff", "预期收益"),
}


def zscore(series: pd.Series):
    """Z 分数，标准差为 0 时返回 0"""
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

def format_step(value):
    """智能格式化步长，消除浮点误差，如 0.020000000000000018 → 0.02"""
    if value is None:
        return None

    # 尝试保留合适的小数位（最多 8 位）
    for digits in range(0, 9):
        v = round(value, digits)
        if abs(v - value) < 1e-10:
            return v

    # 如果全部失败，就保留 6 位
    return round(value, 6)

def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """为存在的指标列添加 z_xxx 列"""
    df = df.copy()
    for key, (col, zcol, _) in METRIC_DEF.items():
        if col in df.columns:
            df[zcol] = zscore(df[col])
    return df


def compute_default_score(df: pd.DataFrame) -> pd.DataFrame:
    """用默认权重算一遍初始 Score_Weighted（用于智能建议）"""
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    for key, weight in DEFAULT_WEIGHTS.items():
        col, zcol, _ = METRIC_DEF[key]
        if zcol in df.columns:
            score += weight * df[zcol]

    df["Score_Weighted"] = score
    return df


def compute_pareto(df: pd.DataFrame):
    """
    计算帕累托前沿（基于 Profit ↑, Sharpe ↑, Drawdown ↓）
    只标记，不画图
    """
    n = len(df)
    if n == 0:
        return pd.Series(False, index=df.index)

    profit = pd.to_numeric(df.get("Profit", 0), errors="coerce").fillna(0)
    sharpe = pd.to_numeric(df.get("Sharpe Ratio", 0), errors="coerce").fillna(0)
    dd = pd.to_numeric(df.get("Equity DD %", 0), errors="coerce").fillna(0)

    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_pareto[i]:
            continue
        p_i, s_i, d_i = profit.iloc[i], sharpe.iloc[i], dd.iloc[i]
        for j in range(n):
            if i == j:
                continue
            p_j, s_j, d_j = profit.iloc[j], sharpe.iloc[j], dd.iloc[j]
            # j 支配 i
            if (
                (p_j >= p_i)
                and (s_j >= s_i)
                and (d_j <= d_i)
                and ((p_j > p_i) or (s_j > s_i) or (d_j < d_i))
            ):
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=df.index)


def pretty_param_str(row, param_cols):
    """把参数列变成 'Period=30, Threshold=0.54' 这种形式"""
    items = []
    for col in param_cols:
        key = col[3:] if col.startswith("inp") else col
        val = row[col]
        items.append(f"{key}={val}")
    return ", ".join(items)


def build_suggestion_cards(df: pd.DataFrame, param_cols):
    """构建 4 张建议卡片内容（title + body_html）"""
    cards = []

    if len(df) == 0:
        cards.append({
            "title": "暂无数据",
            "body": "<p>数据记录为空，无法生成建议。</p>"
        })
        return cards

    profit = pd.to_numeric(df.get("Profit", 0), errors="coerce").fillna(0)
    sharpe = pd.to_numeric(df.get("Sharpe Ratio", 0), errors="coerce").fillna(0)
    dd = pd.to_numeric(df.get("Equity DD %", 0), errors="coerce").fillna(0)
    trades = pd.to_numeric(df.get("Trades", 0), errors="coerce").fillna(0)

    # 1. 激进型：利润最大（简单过滤）
    mask_aggr = (trades >= 10)
    df_aggr = df[mask_aggr] if mask_aggr.any() else df
    row_aggr = df_aggr.loc[profit[df_aggr.index].idxmax()]
    cards.append({
        "title": "1. 激进型策略（追求高利润）",
        "body": f"""
        <p>✓ 推荐参数：<code>{pretty_param_str(row_aggr, param_cols)}</code><br>
           📈 预期：利润 {row_aggr.get('Profit', 'N/A')}, 夏普 {row_aggr.get('Sharpe Ratio', 'N/A')}, 回撤 {row_aggr.get('Equity DD %', 'N/A')}%<br>
           ⚠️ 风险：以收益为先，可能伴随较大的回撤，适合风险承受力较强的用户。</p>
        """
    })

    # 2. 平衡型：综合评分最高（默认权重）
    row_bal = df.sort_values("Score_Weighted", ascending=False).iloc[0]
    cards.append({
        "title": "2. 平衡型策略（风险收益平衡）",
        "body": f"""
        <p>✓ 推荐参数：<code>{pretty_param_str(row_bal, param_cols)}</code><br>
           📈 预期：利润 {row_bal.get('Profit', 'N/A')}, 夏普 {row_bal.get('Sharpe Ratio', 'N/A')}, 回撤 {row_bal.get('Equity DD %', 'N/A')}%<br>
           👍 适合：希望兼顾收益与回撤的大多数交易者。</p>
        """
    })

    # 3. 保守型：利润>0 & 夏普>0 中回撤最小
    mask_cons = (profit > 0) & (sharpe > 0)
    df_cons = df[mask_cons] if mask_cons.any() else df
    dd_cons = dd[df_cons.index]
    row_cons = df_cons.loc[dd_cons.idxmin()]
    cards.append({
        "title": "3. 保守型策略（低回撤优先）",
        "body": f"""
        <p>✓ 推荐参数：<code>{pretty_param_str(row_cons, param_cols)}</code><br>
           📈 预期：利润 {row_cons.get('Profit', 'N/A')}, 夏普 {row_cons.get('Sharpe Ratio', 'N/A')}, 回撤 {row_cons.get('Equity DD %', 'N/A')}%<br>
           🛡️ 特点：更偏向资金安全，适合稳健型交易者。</p>
        """
    })

    # 4. 稳健区间：从 Score_Weighted 前 20% 里算参数分位数
    top_n = max(10, len(df) // 5)
    top_df = df.sort_values("Score_Weighted", ascending=False).head(top_n)

    range_lines = []
    for p in param_cols:
        vals = pd.to_numeric(top_df[p], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        name = p[3:] if p.startswith("inp") else p
        range_lines.append(f"📊 {name}: {q1} - {q3}（表现稳定区间）")

    if range_lines:
        body = "<p>" + "<br>".join(range_lines) + "<br>💡 提示：在上述参数区间内微调，一般可以获得更稳定的收益表现。</p>"
    else:
        body = "<p>暂未识别出明显的稳健参数区间，可考虑扩大优化范围或增加样本数量。</p>"

    cards.append({
        "title": "4. 稳健参数范围建议",
        "body": body
    })

    return cards


def generate_report(df: pd.DataFrame, param_cols, metric_cols, output_path: str, file_name: str):
    """
    生成报告：
    - 顶部信息卡片
    - 权重设置（前端可调）
    - 智能分析建议（卡片 + 默认权重说明）
    - 参数 vs 综合评分（平均）的折线图（前端绘制，可重算）
    - 排行表（前端按当前权重重排，默认前 30 条；帕累托解标绿色）
    """
    df = df.copy()

    # 过滤掉交易次数为 0 的
    if "Trades" in df.columns:
        trades = pd.to_numeric(df["Trades"], errors="coerce").fillna(0)
        df = df[trades > 0]

    # 添加 z 分数列
    df = add_z_scores(df)

    # 用默认权重算一遍初始评分（用于智能建议）
    df = compute_default_score(df)

    # 帕累托（标记）
    pareto_flag = compute_pareto(df)
    df["Is_Pareto"] = pareto_flag

    # 顶部信息
    param_count = len(param_cols)
    total_runs = len(df)
    profit = pd.to_numeric(df.get("Profit", 0), errors="coerce").fillna(0)
    valid_count = int((profit > 0).sum())
    pareto_count = int(pareto_flag.sum())
    analyze_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ==== 参数范围 & 步长 ====
    param_ranges = []
    for p in param_cols:
        col = pd.to_numeric(df[p], errors="coerce")
        vals = sorted(col.dropna().unique())
        if len(vals) >= 2:
            raw_step = vals[1] - vals[0]
            step = format_step(raw_step)
        else:
            step = 0

        param_ranges.append({
            "name": p[3:] if p.startswith("inp") else p,
            "min": vals[0] if len(vals) > 0 else None,
            "max": vals[-1] if len(vals) > 0 else None,
            "step": step
        })


    # 建议卡片
    suggestion_cards = build_suggestion_cards(df, param_cols)

    # 默认权重说明文本
    default_weight_items = []
    for key, val in DEFAULT_WEIGHTS.items():
        cn = WEIGHT_LABELS.get(key, key)
        default_weight_items.append(f"{cn}: {val}")
    default_weights_text = "，".join(default_weight_items)

    # ======== 前端需要的数据 ========

    # rawData：每行包含参数、原始指标、z_xxx、初始 Score_Weighted、Is_Pareto
    raw_records = df.to_dict(orient="records")
    raw_json = json.dumps(raw_records, ensure_ascii=False)

    # paramCols
    param_cols_json = json.dumps(param_cols, ensure_ascii=False)

    # metricsConfig：只保留确实存在的指标
    metrics_config = {}
    for key, (col, zcol, label) in METRIC_DEF.items():
        if col in df.columns and zcol in df.columns:
            metrics_config[key] = {
                "col": col,
                "zcol": zcol,
                "label": label,
            }
    metrics_config_json = json.dumps(metrics_config, ensure_ascii=False)

    # 默认权重
    default_weights_json = json.dumps(DEFAULT_WEIGHTS, ensure_ascii=False)

    # 展示用列名映射（排行榜用）
    display_name_map_json = json.dumps(DISPLAY_NAME_MAP, ensure_ascii=False)

    # 排行表列：参数 + 常见指标 + 综合评分
    table_cols = []
    table_cols.extend(param_cols)
    for c in ["Profit", "Equity DD %", "Sharpe Ratio", "Profit Factor", "Recovery Factor", "Expected Payoff", "Trades"]:
        if c in df.columns and c not in table_cols:
            table_cols.append(c)
    table_cols.append("Score_Weighted")
    table_cols_json = json.dumps(table_cols, ensure_ascii=False)

    # 排行默认显示前 N 条
    rank_top_n = 30

    # ===== HTML 模板（Bootstrap + 前端 Plotly）=====
    html_template = """
<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <title>策略优化报告 - {{ file_name }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet"
          integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH"
          crossorigin="anonymous">
    <!-- Plotly JS -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  </head>
  <body class="bg-light">
    <div class="container my-4">
      <div class="d-flex justify-content-between align-items-end mb-3">
        <h1 class="mb-0">策略优化报告 - {{ file_name }}</h1>
        <div class="text-muted small">分析时间：{{ analyze_time }}</div>
      </div>

      <!-- 顶部信息卡片 -->
      <div class="row mb-4">
        <div class="col-md-3 mb-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title">测试参数数</h6>
              <p class="card-text fs-5 mb-0">{{ param_count }}</p>
            </div>
          </div>
        </div>
        <div class="col-md-3 mb-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title">总回测次数</h6>
              <p class="card-text fs-5 mb-0">{{ total_runs }}</p>
            </div>
          </div>
        </div>
        <div class="col-md-3 mb-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title">有效结果（利润>0）</h6>
              <p class="card-text fs-5 mb-0">{{ valid_count }}</p>
            </div>
          </div>
        </div>
        <div class="col-md-3 mb-3">
          <div class="card shadow-sm">
            <div class="card-body">
              <h6 class="card-title">帕累托前沿解</h6>
              <p class="card-text fs-5 mb-0">{{ pareto_count }}</p>
            </div>
          </div>
        </div>
      </div>
      
            <!-- 测试参数范围 -->
      <div class="card mb-4 shadow-sm">
        <div class="card-header">
          本次测试的参数范围与步长
        </div>
        <div class="card-body p-0">
          <table class="table mb-0 table-bordered table-sm">
            <thead class="table-light">
              <tr>
                <th>参数</th>
                <th>最小值</th>
                <th>最大值</th>
                <th>步长</th>
              </tr>
            </thead>
            <tbody>
              {% for item in param_ranges %}
              <tr>
                <td>{{ item.name }}</td>
                <td>{{ item.min }}</td>
                <td>{{ item.max }}</td>
                <td>{{ item.step }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
      
      <!-- 智能分析建议 -->
      <div class="mb-4">
        <h4 class="mb-3">智能分析建议（基于默认权重）</h4>
        <div class="row">
          {% for card in suggestion_cards %}
          <div class="col-md-6 mb-3">
            <div class="card shadow-sm h-100">
              <div class="card-body">
                <h5 class="card-title">{{ card.title }}</h5>
                <div class="card-text">
                  {{ card.body | safe }}
                </div>
              </div>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <!-- 权重设置 -->
      <div class="card mb-4 shadow-sm">
        <div class="card-header">
          权重设置（归一化综合评分）
        </div>
        <div class="card-body">
          <div class="row g-3">
            <div class="col-md-4">
              <label class="form-label">利润权重 (profit)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_profit">
                <span class="input-group-text" id="w_profit_val"></span>
              </div>
            </div>
            <div class="col-md-4">
              <label class="form-label">最大回撤权重 (drawdown)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_drawdown">
                <span class="input-group-text" id="w_drawdown_val"></span>
              </div>
            </div>
            <div class="col-md-4">
              <label class="form-label">夏普比率权重 (sharpe_ratio)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_sharpe_ratio">
                <span class="input-group-text" id="w_sharpe_ratio_val"></span>
              </div>
            </div>
            <div class="col-md-4">
              <label class="form-label">盈利因子权重 (profit_factor)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_profit_factor">
                <span class="input-group-text" id="w_profit_factor_val"></span>
              </div>
            </div>
            <div class="col-md-4">
              <label class="form-label">采收率权重 (recovery_factor)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_recovery_factor">
                <span class="input-group-text" id="w_recovery_factor_val"></span>
              </div>
            </div>
            <div class="col-md-4">
              <label class="form-label">预期收益权重 (expected_payoff)</label>
              <div class="input-group">
                <input type="number" step="0.01" class="form-control" id="w_expected_payoff">
                <span class="input-group-text" id="w_expected_payoff_val"></span>
              </div>
            </div>
          </div>
          <div class="mt-3">
            <button class="btn btn-primary" id="btn-recompute">更新</button>
            <button class="btn btn-outline-secondary ms-2" id="btn-reset">重置</button>
            <span class="text-muted small ms-2">（权重不强制总和为 1，可自由调整）</span>
          </div>
        </div>
        <div class="text-muted small mt-2 ps-3 pb-3">
            默认权重：{{ default_weights_text }}
          </div>
      </div>

      <!-- 评分与参数敏感性 -->
      <div class="mb-4">
        <h4 class="mb-3">评分与参数敏感性（归一化综合评分）</h4>
        <div id="param-charts">
          {% for p in param_cols %}
          <div class="card mb-3 shadow-sm">
            <div class="card-body">
              <div id="param-chart-{{ loop.index0 }}" style="height: 360px;"></div>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <!-- 排行明细 -->
      <div class="mb-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
          <h4 class="mb-0">参数组合排行榜（按当前综合评分）</h4>
          <div class="btn-group btn-group-sm" role="group">
            <button type="button" class="btn btn-outline-secondary" id="btn-top30">显示前 30 条</button>
            <button type="button" class="btn btn-outline-secondary" id="btn-all">显示全部</button>
          </div>
        </div>
        <div class="card shadow-sm">
          <div class="card-body table-responsive" id="rank-table-container">
            <!-- JS 动态填充 -->
          </div>
        </div>
      </div>

      <footer class="text-muted my-3">
        <small>本报告由 Python + Plotly(前端) + Bootstrap 自动生成。可在上方调整权重，实时更新综合评分与图表。</small>
      </footer>
    </div>

    <script>
      const rawData = {{ raw_json | safe }};
      const paramCols = {{ param_cols_json | safe }};
      const metricsConfig = {{ metrics_config_json | safe }};
      let weights = {{ default_weights_json | safe }};
      const displayNameMap = {{ display_name_map_json | safe }};
      const tableColumns = {{ table_cols_json | safe }};
      let rankTopN = {{ rank_top_n }};   // 默认显示前 N 条
      const defaultWeights = JSON.parse(JSON.stringify(weights)); // 备份默认值

      // 初始化权重输入框
      function initWeightInputs() {
        for (const [key, val] of Object.entries(weights)) {
          const input = document.getElementById('w_' + key);
          const span = document.getElementById('w_' + key + '_val');
          if (!input || !span) continue;
          if (!(key in metricsConfig)) {
            input.value = 0;
            input.disabled = true;
            span.textContent = "不可用";
            continue;
          }
          input.value = val;
          span.textContent = val;
          input.addEventListener('input', () => {
            span.textContent = input.value;
          });
        }
      }

      // 根据当前 weights 重新计算每行 score
      function recomputeScores() {
        for (const key of Object.keys(weights)) {
          const input = document.getElementById('w_' + key);
          if (!input) continue;
          const v = parseFloat(input.value);
          weights[key] = isNaN(v) ? 0 : v;
        }

        rawData.forEach(row => {
          let s = 0;
          for (const [key, cfg] of Object.entries(metricsConfig)) {
            const w = weights[key] || 0;
            const zcol = cfg.zcol;
            const z = Number(row[zcol]);
            if (!isNaN(z)) {
              s += w * z;
            }
          }
          row.score = s;
        });
      }
      
      function resetWeights() {
        // 恢复默认值
        for (const key of Object.keys(defaultWeights)) {
          weights[key] = defaultWeights[key];
          const input = document.getElementById('w_' + key);
          const span = document.getElementById('w_' + key + '_val');
          if (input && span) {
            input.value = defaultWeights[key];
            span.textContent = defaultWeights[key];
          }
        }
    
        // 重算分数 + 更新图表 + 排行榜
        recomputeScores();
        buildParamCharts();
        buildRankingTable();
    }


      // 构建参数敏感性图：每个参数 vs score 平均值
      function buildParamCharts() {
        paramCols.forEach((param, idx) => {
          const grouped = {};
          rawData.forEach(row => {
            const v = row[param];
            if (v === undefined || v === null) return;
            const key = String(v);
            if (!grouped[key]) grouped[key] = [];
            grouped[key].push(row.score);
          });

          const xs = [];
          const ys = [];
          const keys = Object.keys(grouped).sort((a, b) => parseFloat(a) - parseFloat(b));
          keys.forEach(k => {
            const arr = grouped[k];
            const avg = arr.reduce((sum, val) => sum + val, 0) / arr.length;
            xs.push(parseFloat(k));
            ys.push(avg);
          });

          const divId = 'param-chart-' + idx;
          const titleName = param.startsWith('inp') ? param.slice(3) : param;

          const trace = {
            x: xs,
            y: ys,
            mode: 'lines+markers',
            name: titleName
          };
          const layout = {
            title: titleName + ' vs 综合评分（平均值）',
            xaxis: { title: titleName },
            yaxis: { title: '综合评分（平均）' },
            margin: { t: 40, r: 20, b: 40, l: 50 }
          };

          Plotly.react(divId, [trace], layout);
        });
      }

      // 排行榜：按 score 排序，默认前 N 条，帕累托解标绿色
      function buildRankingTable() {
        const container = document.getElementById('rank-table-container');
        if (!container) return;

        const rows = rawData.slice().sort((a, b) => (b.score || 0) - (a.score || 0));
        const total = rows.length;
        const rowsToShow = (rankTopN && rankTopN > 0) ? rows.slice(0, rankTopN) : rows;

        let html = '<table class="table table-striped table-sm"><thead><tr>';
        tableColumns.forEach(col => {
          let label;
          if (col === 'Score_Weighted') {
            label = '综合评分';
          } else if (col.startsWith('inp')) {
            label = col.slice(3);
          } else {
            label = displayNameMap[col] || col;
          }
          html += '<th>' + label + '</th>';
        });
        html += '</tr></thead><tbody>';

        rowsToShow.forEach(row => {
          const isPareto = row["Is_Pareto"] === true || row["Is_Pareto"] === 1 || row["Is_Pareto"] === "True";
          const trClass = isPareto ? ' class="table-success"' : '';
          html += '<tr' + trClass + '>';
          tableColumns.forEach(col => {
            let val;
            if (col === 'Score_Weighted') {
              val = row.score != null ? row.score.toFixed(3) : '';
            } else {
              val = row[col];
            }
            if (val === undefined || val === null) val = '';
            html += '<td>' + val + '</td>';
          });
          html += '</tr>';
        });

        html += '</tbody></table>';
        html += '<div class="text-muted small mt-2">提示：绿色行表示帕累托前沿解；当前显示 '
              + rowsToShow.length + ' 条，共 ' + total + ' 条。</div>';
        container.innerHTML = html;
      }

      document.addEventListener('DOMContentLoaded', () => {
        initWeightInputs();
        recomputeScores();
        buildParamCharts();
        buildRankingTable();

        const btn = document.getElementById('btn-recompute');
        if (btn) {
          btn.addEventListener('click', () => {
            recomputeScores();
            buildParamCharts();
            buildRankingTable();
          });
        }
        
        const btnReset = document.getElementById('btn-reset');
        if (btnReset) {
          btnReset.addEventListener('click', () => {
            resetWeights();
          });
        }


        const btnTop30 = document.getElementById('btn-top30');
        const btnAll = document.getElementById('btn-all');

        if (btnTop30) {
          btnTop30.addEventListener('click', () => {
            rankTopN = 30;
            buildRankingTable();
          });
        }
        if (btnAll) {
          btnAll.addEventListener('click', () => {
            rankTopN = 0;  // 0 表示全部
            buildRankingTable();
          });
        }
      });
    </script>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
            integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz"
            crossorigin="anonymous"></script>
  </body>
</html>
    """

    tpl = Template(html_template)
    final_html = tpl.render(
        file_name=file_name,
        analyze_time=analyze_time,
        param_count=param_count,
        total_runs=total_runs,
        valid_count=valid_count,
        pareto_count=pareto_count,
        suggestion_cards=suggestion_cards,
        default_weights_text=default_weights_text,
        raw_json=raw_json,
        param_cols_json=param_cols_json,
        metrics_config_json=metrics_config_json,
        default_weights_json=default_weights_json,
        display_name_map_json=display_name_map_json,
        table_cols_json=table_cols_json,
        param_cols=param_cols,
        rank_top_n=rank_top_n,
        param_ranges=param_ranges,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    print("🎉 HTML 报告生成成功：", output_path)
