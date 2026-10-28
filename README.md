# A-Share Strategy Data

> A 股多因子策略的**数据底座**：从公开数据源拉取行情与财务数据，对齐成标准面板，再计算基本面因子与高频因子。

![python](https://img.shields.io/badge/python-3.10%2B-3776ab) ![data](https://img.shields.io/badge/source-akshare%20%2F%20baostock-green)

---

## 它解决什么问题

做 A 股多因子研究，最耗时的从来不是模型，而是**数据口径**：

- 不同数据源的字段名不一样（`retainedprofit` / `retainedearnings` / `retainedprofit`…）；
- 财务数据是季度披露的，但因子要按日频对齐，中间涉及**财报滞后**与**前视偏差**；
- 单季值、TTM、同比、环比、均值、波动率——同一个指标要派生出十几种形态。

这个仓库把这三件事标准化了：**统一字段名 → 统一面板结构（date × asset 的 MultiIndex）→ 统一派生算子**。

---

## 目录结构

```
a-share-strategy-data/
└── src/
    ├── fetch_a_share_data.py              # 数据拉取（akshare + baostock）
    ├── prepare_a_share_strategy_data.py   # 字段标准化与面板对齐
    ├── ai_factor_strategies.py            # ⭐ 因子计算 + Lasso 合成 + CLI
    └── main.py                            # 命令行入口
```

---

## 数据流

```
   akshare / baostock
          │
          ▼
  fetch_a_share_data.py          原始行情 + 财务报表
          │
          ▼
  prepare_a_share_strategy_data.py
          │   字段映射：retainedprofit → UNASSIGN_RPOFIT
          │             operatingrevenue → OPERATE_INCOME
          │             netprofit → NETPROFIT   ...
          ▼
   MultiIndex 面板 (date, asset)
          │
          ├──▶ 基本面因子库（ROE / 毛利率 / 应计 / 现金流质量 …）
          └──▶ 高频因子库（上下影线空间 / 日内 VWAP 偏离 / 动量加速 …）
          │
          ▼
  rolling_lasso_composite()       滚动 Lasso 合成复合因子
```

---

## 快速开始

```bash
pip install pandas numpy scikit-learn akshare baostock
```

```bash
cd src

# 1. 拉数据（示例：沪深 300 成分股，日频）
python fetch_a_share_data.py --symbols 600519 000001 300750 --start 2020-01-01

# 2. 字段标准化 + 面板对齐
python prepare_a_share_strategy_data.py --input raw/ --output panel.csv

# 3. 一键跑通（合成数据 demo，验证环境）
python main.py demo --output-dir outputs

# 4. 只算基本面因子
python main.py fundamental --panel panel.csv --output factors.csv

# 5. 计算完整特征库并对齐到日频价格面板
python main.py fundamental \
  --panel panel.csv \
  --daily-price-panel price_panel.csv \
  --output factors_daily.csv
```

---

## 核心设计

### 1. 字段别名表

不同数据源对同一财务科目的命名差异，用一个 `FIELD_ALIASES` 字典兜住：

```python
FIELD_ALIASES = {
    "retainedearnings": ("retainedearnings", "retainedprofit"),
    "totalpaidincapital": ("totalpaidincapital", "paidincapital"),
    "grossprofit": ("grossprofit",),
}
```

新增数据源时只需要在这里加一行，不用改因子代码。

### 2. 面板索引强校验

所有因子函数入口都调用 `ensure_panel_index()`，要求索引必须是 `[date, asset]` 两级 MultiIndex，且自动排序。**在错误的数据结构上算因子，比算错因子更可怕**——所以这里选择直接抛错而不是静默兼容。

### 3. 派生算子

| 算子 | 作用 |
|---|---|
| `refq(s, n)` | 按季度回溯 n 期（`groupby(asset).shift(n)`） |
| `ttm(s)` | 滚动 4 季求和，把单季值转成 TTM |
| `yoy(s)` | 同比 |
| `sud(s, w)` / `sud2(s, w)` | 标准化 unexpectedly deviation，衡量偏离历史均值的程度 |
| `svr(s, w)` | 滚动波动率 |
| `average_with_lag(s, n)` | 滞后平均，用于规避财报披露延迟 |

### 4. 高频因子

输入是分钟级宽表（`date` × `asset`），派生出：

- `get_up_space` / `get_mid_space` / `get_down_space`：日内上下影线空间；
- `intraday_vwap`：日内成交量加权均价；
- `momentum_acceleration_corr`：动量加速度与相关性；
- `speculative_frenzy_instability`：投机情绪不稳定性；
- `extreme_gamma_burst_ratio`：极端波动爆发比。

### 5. 滚动 Lasso 合成

`rolling_lasso_composite()` 用滚动窗口训练 `LassoCV`（带 `StandardScaler` + `SimpleImputer` 的 Pipeline），把多个因子压缩成一个复合打分，避免样本外使用未来信息。

---

## 注意事项

- **财务数据必须滞后**：`prepare_a_share_strategy_data.py` 只做字段对齐，**不做**披露日对齐；正式回测请按实际公告日 `shift`，否则会引入前视偏差。
- **停牌与涨跌停**：拉取阶段未做剔除，因子计算前需自行处理，否则极值会污染截面标准化。
- **数据源限制**：akshare / baostock 为免费源，历史深度与字段完整度有限；生产级研究建议替换为 Wind / iFinD，接口层已做抽象，替换成本较低。
- 仓库内**不含任何行情数据**，需自行拉取。
