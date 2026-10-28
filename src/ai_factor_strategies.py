from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PanelFrame = pd.DataFrame
WideFrame = pd.DataFrame


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "retainedearnings": ("retainedearnings", "retainedprofit"),
    "totalpaidincapital": ("totalpaidincapital", "paidincapital"),
    "grossprofit": ("grossprofit",),
}


def safe_divide(numerator: pd.Series | pd.DataFrame, denominator: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def ensure_panel_index(frame: PanelFrame) -> PanelFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("Expected a MultiIndex panel with levels [date, asset].")
    names = list(frame.index.names)
    if names[0] is None:
        names[0] = "date"
    if names[1] is None:
        names[1] = "asset"
    frame = frame.copy()
    frame.index = frame.index.set_names(names)
    return frame.sort_index()


def panel_dates(frame: PanelFrame | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(frame.index.get_level_values(0))


def get_field(frame: PanelFrame, name: str) -> pd.Series:
    frame = ensure_panel_index(frame)
    candidates = FIELD_ALIASES.get(name, (name,))
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate].astype(float)

    if name == "grossprofit":
        revenue = None
        for revenue_name in ("operatingrevenue", "totaloperatingrevenue"):
            if revenue_name in frame.columns:
                revenue = frame[revenue_name].astype(float)
                break
        cost = None
        for cost_name in ("operatingcost", "totaloperatingcost"):
            if cost_name in frame.columns:
                cost = frame[cost_name].astype(float)
                break
        if revenue is not None and cost is not None:
            return revenue - cost

    raise KeyError(f"Missing field '{name}'. Available columns: {list(frame.columns)}")


def refq(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.groupby(level=1).shift(periods)


def quarterly_flow(series: pd.Series) -> pd.Series:
    dates = panel_dates(series)
    prev_values = series.groupby(level=1).shift(1)
    prev_dates = pd.Series(dates, index=series.index).groupby(level=1).shift(1)
    same_year = pd.Series(dates.year, index=series.index).eq(prev_dates.dt.year)
    flow = (series - prev_values).where(same_year, series)
    return flow.astype(float)


def grouped_rolling(series: pd.Series, window: int, stat: str, min_periods: int | None = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    rolled = getattr(series.groupby(level=1).rolling(window, min_periods=min_periods), stat)()
    return rolled.droplevel(0).reindex(series.index)


def ttm(series: pd.Series) -> pd.Series:
    q = quarterly_flow(series)
    return grouped_rolling(q, window=4, stat="sum", min_periods=4)


def yoy(series: pd.Series) -> pd.Series:
    return series.groupby(level=1).pct_change(4)


def average_with_lag(series: pd.Series, periods: int = 1) -> pd.Series:
    return (series + refq(series, periods)) / 2.0


def sud(series: pd.Series, window: int = 8) -> pd.Series:
    history = series.groupby(level=1).shift(1)
    mean = grouped_rolling(history, window=window, stat="mean", min_periods=max(4, window // 2))
    std = grouped_rolling(history, window=window, stat="std", min_periods=max(4, window // 2))
    return safe_divide(series - mean, std)


def sud2(series: pd.Series, window: int = 4) -> pd.Series:
    history = series.groupby(level=1).shift(1)
    mean = grouped_rolling(history, window=window, stat="mean", min_periods=max(2, window // 2))
    std = grouped_rolling(history, window=window, stat="std", min_periods=max(2, window // 2))
    return safe_divide(series - mean, std)


def svr(series: pd.Series, window: int = 8) -> pd.Series:
    history = series.groupby(level=1).shift(1)
    std = grouped_rolling(history, window=window, stat="std", min_periods=max(4, window // 2))
    return safe_divide(series, std)


def series_dict_to_frame(series_map: Mapping[str, pd.Series]) -> PanelFrame:
    frame = pd.concat(series_map, axis=1)
    frame.columns = list(series_map.keys())
    frame.index = frame.index.set_names(["date", "asset"])
    return frame.sort_index()


def compute_representative_fundamental_factors(panel: PanelFrame) -> PanelFrame:
    panel = ensure_panel_index(panel)

    retained = get_field(panel, "retainedearnings")
    mv = get_field(panel, "mv")
    goods_cash = get_field(panel, "goodssaleservicerendercash")
    goods_paid = get_field(panel, "goodsservicescashpaid")
    gross_profit = get_field(panel, "grossprofit")
    selling_expense = get_field(panel, "operatingexpense")
    admin_expense = get_field(panel, "totaladminexpense")
    shares = get_field(panel, "paidincapital")
    revenue = get_field(panel, "totaloperatingrevenue")
    account_receivable = get_field(panel, "accountreceivable")
    net_profit = get_field(panel, "npparentcompanyowners")
    dividend_ttm = get_field(panel, "dividend_ttm")
    equity = get_field(panel, "sewithoutmi")

    factors = {
        "REP_LF": safe_divide(retained, mv),
        "CGP_TTM": safe_divide(ttm(goods_cash) - ttm(goods_paid), mv),
        "COPPS_QR": safe_divide(
            quarterly_flow(gross_profit) - quarterly_flow(selling_expense) - quarterly_flow(admin_expense),
            shares,
        ),
        "ART_QR": safe_divide(quarterly_flow(revenue), average_with_lag(account_receivable)),
        "SGR_LF": safe_divide(ttm(net_profit) - dividend_ttm, equity),
    }
    return series_dict_to_frame(factors)


def build_fundamental_feature_library(panel: PanelFrame) -> PanelFrame:
    base = compute_representative_fundamental_factors(panel)
    features: dict[str, pd.Series] = {}

    for column in base.columns:
        series = base[column]
        features[column] = series
        features[f"{column}_SUD8"] = sud(series, window=8)
        features[f"{column}_SUD4"] = sud2(series, window=4)
        features[f"{column}_SVR8"] = svr(series, window=8)

    return series_dict_to_frame(features)


def forward_fill_panel_to_dates(panel: PanelFrame, dates: pd.DatetimeIndex) -> PanelFrame:
    panel = ensure_panel_index(panel)
    dates = pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates))))
    pieces: list[pd.Series] = []

    for column in panel.columns:
        wide = panel[column].unstack("asset").sort_index()
        filled = wide.reindex(dates).ffill()
        try:
            stacked = filled.stack(dropna=False)
        except ValueError:
            stacked = filled.stack(future_stack=True)
        pieces.append(stacked.rename(column))

    result = pd.concat(pieces, axis=1).sort_index()
    result.index = result.index.set_names(["date", "asset"])
    return result


def compute_forward_vwap_return(price_panel: PanelFrame, price_col: str = "vwap", horizon: int = 5) -> pd.Series:
    price_panel = ensure_panel_index(price_panel)
    price = get_field(price_panel, price_col)
    future = price.groupby(level=1).shift(-horizon)
    return safe_divide(future, price) - 1.0


def rolling_lasso_composite(
    features: PanelFrame,
    target: pd.Series,
    train_years: int = 5,
    min_train_rows: int = 250,
    random_state: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    features = ensure_panel_index(features)
    target = target.sort_index()
    joined = features.join(target.rename("target"), how="inner")
    years = panel_dates(joined).year
    predictions = pd.Series(index=joined.index, dtype=float, name="lasso_composite")
    coeff_rows: list[pd.Series] = []

    for year in sorted(pd.unique(years)):
        train_mask = (years >= year - train_years) & (years < year)
        test_mask = years == year
        if train_mask.sum() < min_train_rows or test_mask.sum() == 0:
            continue

        train = joined.loc[train_mask].dropna(subset=["target"])
        test = joined.loc[test_mask]
        if len(train) < min_train_rows:
            continue

        feature_cols = [
            column
            for column in features.columns
            if train[column].notna().sum() >= max(30, train_years * 12)
        ]
        if not feature_cols:
            continue

        x_train = train[feature_cols]
        y_train = train["target"]
        x_test = test[feature_cols]
        if x_test.empty:
            continue

        cv_folds = min(5, max(2, len(train) // 120))
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lasso", LassoCV(cv=cv_folds, random_state=random_state, max_iter=50000)),
            ]
        )
        model.fit(x_train, y_train)
        predictions.loc[test.index] = model.predict(x_test)

        coeff = pd.Series(model.named_steps["lasso"].coef_, index=feature_cols, name=year)
        coeff_rows.append(coeff)

    coeff_frame = pd.DataFrame(coeff_rows)
    coeff_frame.index.name = "prediction_year"
    return predictions, coeff_frame


def demean_cross_section(frame: WideFrame) -> WideFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


def intraday_vwap(inputs: Mapping[str, WideFrame]) -> WideFrame:
    if "vwap" in inputs:
        return inputs["vwap"].astype(float)
    if "amount" in inputs and "volume" in inputs:
        return safe_divide(inputs["amount"].astype(float), inputs["volume"].astype(float))
    return inputs["close"].astype(float)


def validate_minute_bundle(inputs: Mapping[str, WideFrame]) -> dict[str, WideFrame]:
    required = {"close", "high", "low", "volume"}
    missing = required - set(inputs)
    if missing:
        raise ValueError(f"Missing minute inputs: {sorted(missing)}")

    close = inputs["close"]
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Minute data must use a DatetimeIndex.")

    normalized: dict[str, WideFrame] = {}
    for name, frame in inputs.items():
        casted = frame.astype(float).sort_index()
        if not casted.index.equals(close.index) or not casted.columns.equals(close.columns):
            raise ValueError(f"Minute input '{name}' must share the same index and columns as 'close'.")
        normalized[name] = casted

    if "open" not in normalized:
        normalized["open"] = normalized["close"].shift(1)
    return normalized


def same_time_rolling_stat(frame: WideFrame, window: int = 20, stat: str = "mean") -> WideFrame:
    frame = frame.sort_index()
    times = np.array(frame.index.time)
    pieces: list[WideFrame] = []
    min_periods = max(5, window // 2)

    for current_time in pd.unique(times):
        mask = times == current_time
        sub = frame.loc[mask]
        rolled = getattr(sub.shift(1).rolling(window, min_periods=min_periods), stat)()
        pieces.append(rolled)

    return pd.concat(pieces).sort_index().reindex(frame.index)


def get_up_space(frame: WideFrame) -> WideFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    return frame.gt(mean + std, axis=0)


def get_mid_space(frame: WideFrame) -> WideFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    lower = frame.ge(mean - std, axis=0)
    upper = frame.le(mean + std, axis=0)
    return lower & upper


def get_down_space(frame: WideFrame) -> WideFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    return frame.lt(mean - std, axis=0)


def get_up_space_rolling(frame: WideFrame, window: int = 20) -> WideFrame:
    mean = same_time_rolling_stat(frame, window=window, stat="mean")
    std = same_time_rolling_stat(frame, window=window, stat="std")
    return frame > (mean + std)


def get_mid_space_rolling(frame: WideFrame, window: int = 20) -> WideFrame:
    mean = same_time_rolling_stat(frame, window=window, stat="mean")
    std = same_time_rolling_stat(frame, window=window, stat="std")
    return (frame >= (mean - std)) & (frame <= (mean + std))


def get_down_space_rolling(frame: WideFrame, window: int = 20) -> WideFrame:
    mean = same_time_rolling_stat(frame, window=window, stat="mean")
    std = same_time_rolling_stat(frame, window=window, stat="std")
    return frame < (mean - std)


def daily_corr(left: WideFrame, right: WideFrame) -> WideFrame:
    grouped_dates = pd.Index(left.index.normalize(), name="date")
    rows: list[pd.Series] = []

    for date in pd.unique(grouped_dates):
        mask = grouped_dates == date
        corr = left.loc[mask].corrwith(right.loc[mask], axis=0)
        corr.name = pd.Timestamp(date)
        rows.append(corr)

    return pd.DataFrame(rows).sort_index()


def smooth_daily_factor(frame: WideFrame, window: int = 5) -> WideFrame:
    return frame.sort_index().rolling(window, min_periods=1).mean()


def speculative_frenzy_instability(inputs: Mapping[str, WideFrame], rolling_window: int = 20) -> WideFrame:
    data = validate_minute_bundle(inputs)
    high = data["high"]
    low = data["low"]
    close = data["close"]
    volume = data["volume"]
    vwap = intraday_vwap(data)

    amp_norm = safe_divide(high - low, vwap)
    price_dev = safe_divide(close, vwap) - 1.0
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    idio_abs_ret = demean_cross_section(ret).abs()

    stress = (
        get_up_space_rolling(volume, window=rolling_window)
        & get_up_space_rolling(amp_norm.abs(), window=rolling_window)
        & get_up_space_rolling(price_dev.abs(), window=rolling_window)
    )
    daily = idio_abs_ret.where(stress).groupby(close.index.normalize()).std()
    return smooth_daily_factor(daily)


def extreme_gamma_burst_ratio(inputs: Mapping[str, WideFrame], rolling_window: int = 20) -> WideFrame:
    data = validate_minute_bundle(inputs)
    close = data["close"]
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    gamma = ret.diff().replace([np.inf, -np.inf], np.nan)
    burst = get_up_space_rolling(gamma.abs(), window=rolling_window)
    burst_var = gamma.where(burst).groupby(close.index.normalize()).var()
    full_var = gamma.groupby(close.index.normalize()).var()
    daily = safe_divide(burst_var, full_var)
    return smooth_daily_factor(daily)


def momentum_acceleration_corr(inputs: Mapping[str, WideFrame]) -> WideFrame:
    data = validate_minute_bundle(inputs)
    close = data["close"]
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    momentum = ret.shift(1)
    gamma = ret.diff()
    daily = daily_corr(momentum, gamma)
    return smooth_daily_factor(daily)


def dual_stress_rolling_idio_ret_dev_product(
    inputs: Mapping[str, WideFrame],
    rolling_window: int = 20,
) -> WideFrame:
    data = validate_minute_bundle(inputs)
    close = data["close"]
    volume = data["volume"]
    vwap = intraday_vwap(data)

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    price_dev = safe_divide(close, vwap) - 1.0
    idio_ret = demean_cross_section(ret)
    idio_price_dev = demean_cross_section(price_dev)

    dual_stress = (
        get_up_space_rolling(volume, window=rolling_window)
        & get_up_space_rolling(idio_ret.abs(), window=rolling_window)
    )
    product = idio_ret * idio_price_dev
    daily = product.where(dual_stress).groupby(close.index.normalize()).mean()
    return smooth_daily_factor(daily)


def compute_high_frequency_factor_suite(inputs: Mapping[str, WideFrame], rolling_window: int = 20) -> dict[str, WideFrame]:
    return {
        "speculative_frenzy_instability": speculative_frenzy_instability(inputs, rolling_window=rolling_window),
        "extreme_gamma_burst_ratio": extreme_gamma_burst_ratio(inputs, rolling_window=rolling_window),
        "momentum_acceleration_corr": momentum_acceleration_corr(inputs),
        "dual_stress_rolling_idio_ret_dev_product": dual_stress_rolling_idio_ret_dev_product(
            inputs,
            rolling_window=rolling_window,
        ),
    }


def load_panel_csv(path: str | Path, date_col: str = "date", asset_col: str = "asset") -> PanelFrame:
    frame = pd.read_csv(path)
    frame[date_col] = pd.to_datetime(frame[date_col])
    return ensure_panel_index(frame.set_index([date_col, asset_col]))


def load_wide_csv(path: str | Path) -> WideFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def save_panel_csv(frame: PanelFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reset_index().to_csv(path, index=False)


def save_wide_csv(frame: WideFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def load_minute_bundle(directory: str | Path) -> dict[str, WideFrame]:
    directory = Path(directory)
    bundle: dict[str, WideFrame] = {}
    for name in ("open", "high", "low", "close", "volume", "amount", "vwap"):
        candidate = directory / f"{name}.csv"
        if candidate.exists():
            bundle[name] = load_wide_csv(candidate)
    return validate_minute_bundle(bundle)


def synthetic_fundamental_panel(seed: int = 42) -> PanelFrame:
    rng = np.random.default_rng(seed)
    assets = [f"STK{i:03d}" for i in range(1, 9)]
    dates = pd.date_range("2014-03-31", "2025-12-31", freq="QE")
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for asset in assets:
        price_level = rng.uniform(40, 150)
        equity_base = rng.uniform(2e9, 8e9)
        share_count = rng.uniform(8e8, 2e9)
        cumulative = {
            "totaloperatingrevenue": 0.0,
            "goodssaleservicerendercash": 0.0,
            "goodsservicescashpaid": 0.0,
            "npparentcompanyowners": 0.0,
            "operatingexpense": 0.0,
            "totaladminexpense": 0.0,
            "dividend_ttm": 0.0,
        }

        for date in dates:
            if date.quarter == 1:
                for key in cumulative:
                    cumulative[key] = 0.0

            revenue_q = rng.uniform(2e8, 7e8)
            gross_profit_q = revenue_q * rng.uniform(0.18, 0.42)
            goods_cash_q = revenue_q * rng.uniform(0.95, 1.12)
            goods_paid_q = revenue_q * rng.uniform(0.52, 0.82)
            net_profit_q = gross_profit_q * rng.uniform(0.28, 0.55)
            operating_expense_q = revenue_q * rng.uniform(0.03, 0.12)
            admin_expense_q = revenue_q * rng.uniform(0.02, 0.09)
            dividend_q = max(net_profit_q * rng.uniform(0.05, 0.25), 0.0)

            cumulative["totaloperatingrevenue"] += revenue_q
            cumulative["goodssaleservicerendercash"] += goods_cash_q
            cumulative["goodsservicescashpaid"] += goods_paid_q
            cumulative["npparentcompanyowners"] += net_profit_q
            cumulative["operatingexpense"] += operating_expense_q
            cumulative["totaladminexpense"] += admin_expense_q
            cumulative["dividend_ttm"] += dividend_q

            retained_profit = equity_base * rng.uniform(0.10, 0.45)
            equity = equity_base + rng.uniform(-3e8, 5e8)
            mv = price_level * share_count * rng.uniform(0.9, 1.1)
            receivable = revenue_q * rng.uniform(0.12, 0.28)

            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "retainedprofit": retained_profit,
                    "mv": mv,
                    "goodssaleservicerendercash": cumulative["goodssaleservicerendercash"],
                    "goodsservicescashpaid": cumulative["goodsservicescashpaid"],
                    "grossprofit": gross_profit_q,
                    "operatingexpense": cumulative["operatingexpense"],
                    "totaladminexpense": cumulative["totaladminexpense"],
                    "paidincapital": share_count,
                    "totaloperatingrevenue": cumulative["totaloperatingrevenue"],
                    "accountreceivable": receivable,
                    "npparentcompanyowners": cumulative["npparentcompanyowners"],
                    "dividend_ttm": max(cumulative["dividend_ttm"], 1.0),
                    "sewithoutmi": equity,
                }
            )

    frame = pd.DataFrame(rows)
    return ensure_panel_index(frame.set_index(["date", "asset"]))


def synthetic_daily_price_panel(assets: list[str], seed: int = 7) -> PanelFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", "2025-12-31")
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for asset in assets:
        base = rng.uniform(20, 80)
        noise = rng.normal(0.0003, 0.018, len(dates))
        prices = base * np.exp(np.cumsum(noise))
        for date, price in zip(dates, prices, strict=True):
            rows.append({"date": date, "asset": asset, "vwap": price})

    frame = pd.DataFrame(rows)
    return ensure_panel_index(frame.set_index(["date", "asset"]))


def synthetic_minute_bundle(seed: int = 11) -> dict[str, WideFrame]:
    rng = np.random.default_rng(seed)
    assets = [f"STK{i:03d}" for i in range(1, 7)]
    days = pd.bdate_range("2025-01-02", periods=35)
    minutes = pd.date_range("09:30", "10:29", freq="1min").time
    index = pd.DatetimeIndex([pd.Timestamp.combine(day.date(), minute) for day in days for minute in minutes])

    base_returns = rng.normal(0, 0.0015, (len(index), len(assets)))
    close = pd.DataFrame(50 * np.exp(np.cumsum(base_returns, axis=0)), index=index, columns=assets)
    open_ = close.shift(1).fillna(close)
    high = close * (1 + np.abs(rng.normal(0.0008, 0.0006, close.shape)))
    low = close * (1 - np.abs(rng.normal(0.0008, 0.0006, close.shape)))
    volume = pd.DataFrame(rng.lognormal(mean=10.5, sigma=0.45, size=close.shape), index=index, columns=assets)
    amount = close * volume * (1 + rng.normal(0, 0.002, close.shape))
    vwap = safe_divide(amount, volume)

    return {
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": volume.astype(float),
        "amount": amount.astype(float),
        "vwap": vwap.astype(float),
    }


def build_demo_bundle() -> tuple[PanelFrame, PanelFrame, dict[str, WideFrame]]:
    fundamental = synthetic_fundamental_panel()
    assets = sorted(fundamental.index.get_level_values("asset").unique())
    prices = synthetic_daily_price_panel(assets)
    minute_bundle = synthetic_minute_bundle()
    return fundamental, prices, minute_bundle


def run_demo(output_dir: str | Path = "demo_outputs") -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fundamental, price_panel, minute_bundle = build_demo_bundle()
    fundamental_factors = compute_representative_fundamental_factors(fundamental)
    feature_library = build_fundamental_feature_library(fundamental)
    expanded_features = forward_fill_panel_to_dates(feature_library, panel_dates(price_panel))
    target = compute_forward_vwap_return(price_panel)
    composite, coeffs = rolling_lasso_composite(expanded_features, target)
    hf_factors = compute_high_frequency_factor_suite(minute_bundle)

    save_panel_csv(fundamental_factors, output_dir / "fundamental_factors.csv")
    save_panel_csv(feature_library, output_dir / "fundamental_feature_library.csv")
    save_panel_csv(composite.rename("lasso_composite").to_frame(), output_dir / "fundamental_lasso_composite.csv")
    coeffs.to_csv(output_dir / "fundamental_lasso_coefficients.csv")

    for name, frame in hf_factors.items():
        save_wide_csv(frame, output_dir / f"{name}.csv")

    return {
        "fundamental_factors": fundamental_factors,
        "feature_library": feature_library,
        "composite": composite,
        "coefficients": coeffs,
        "high_frequency_factors": hf_factors,
        "output_dir": output_dir,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Implement representative AI factor strategies from the report.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run a full synthetic-data smoke test.")
    demo_parser.add_argument("--output-dir", default="demo_outputs")

    fundamental_parser = subparsers.add_parser("fundamental", help="Compute fundamental factors and optional Lasso composite.")
    fundamental_parser.add_argument("--panel", required=True, help="Panel CSV with [date, asset] plus financial columns.")
    fundamental_parser.add_argument("--output", required=True, help="Output CSV for representative fundamental factors.")
    fundamental_parser.add_argument("--daily-price-panel", help="Optional daily [date, asset, vwap] panel for Lasso composite.")
    fundamental_parser.add_argument("--composite-output", help="Optional output CSV for the Lasso composite signal.")
    fundamental_parser.add_argument("--coeff-output", help="Optional output CSV for yearly Lasso coefficients.")

    highfreq_parser = subparsers.add_parser("highfreq", help="Compute minute-based high-frequency factors.")
    highfreq_parser.add_argument("--minute-dir", required=True, help="Directory containing close/high/low/volume CSV files.")
    highfreq_parser.add_argument("--output-dir", required=True, help="Directory where factor CSV files will be written.")
    highfreq_parser.add_argument("--rolling-window", type=int, default=20)

    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        result = run_demo(args.output_dir)
        print(f"Demo completed. Outputs written to: {result['output_dir']}")
        print("Representative factors:", ", ".join(result["fundamental_factors"].columns))
        print("High-frequency factors:", ", ".join(result["high_frequency_factors"].keys()))
        return 0

    if args.command == "fundamental":
        panel = load_panel_csv(args.panel)
        factors = compute_representative_fundamental_factors(panel)
        save_panel_csv(factors, args.output)
        print(f"Saved representative factors to {args.output}")

        if args.daily_price_panel:
            feature_library = build_fundamental_feature_library(panel)
            price_panel = load_panel_csv(args.daily_price_panel)
            expanded = forward_fill_panel_to_dates(feature_library, panel_dates(price_panel))
            target = compute_forward_vwap_return(price_panel)
            composite, coeffs = rolling_lasso_composite(expanded, target)

            if args.composite_output:
                save_panel_csv(composite.rename("lasso_composite").to_frame(), args.composite_output)
                print(f"Saved Lasso composite to {args.composite_output}")
            if args.coeff_output:
                coeffs.to_csv(args.coeff_output)
                print(f"Saved yearly Lasso coefficients to {args.coeff_output}")
        return 0

    if args.command == "highfreq":
        bundle = load_minute_bundle(args.minute_dir)
        outputs = compute_high_frequency_factor_suite(bundle, rolling_window=args.rolling_window)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            target = output_dir / f"{name}.csv"
            save_wide_csv(frame, target)
            print(f"Saved {name} to {target}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
