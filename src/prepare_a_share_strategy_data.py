from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


FUNDAMENTAL_COLUMN_MAP = {
    "mv": "TOTAL_MV",
    "sewithoutmi": "TOTAL_PARENT_EQUITY",
    "accountreceivable": "ACCOUNTS_RECE",
    "paidincapital": "SHARE_CAPITAL",
    "capitalreservefund": "CAPITAL_RESERVE",
    "retainedprofit": "UNASSIGN_RPOFIT",
    "totalassets": "TOTAL_ASSETS",
    "totalliability": "TOTAL_LIABILITIES",
    "operatingrevenue": "OPERATE_INCOME",
    "totaloperatingrevenue": "OPERATE_INCOME",
    "operatingcost": "OPERATE_EXPENSE",
    "operatingexpense": "SALE_EXPENSE",
    "totaladminexpense": "MANAGE_EXPENSE",
    "financialexpense": "FINANCE_EXPENSE",
    "operatingprofit": "OPERATE_PROFIT",
    "totalprofit": "TOTAL_PROFIT",
    "netprofit": "NETPROFIT",
    "npparentcompanyowners": "PARENT_NETPROFIT",
    "npdeductnonrecurringpl": "DEDUCT_PARENT_NETPROFIT",
    "basiceps": "BASIC_EPS",
    "goodssaleservicerendercash": "RECEIVE_OTHER_OPERATE",
    "goodsservicescashpaid": "BUY_FIN_LEASE",
    "staffbehalfpaid": "PAY_STAFF_CASH",
    "alltaxespaid": "PAY_ALL_TAX",
    "otheroperatecashpaid": "PAY_OTHER_OPERATE",
    "netoperatecashflow": "NETCASH_OPERATE",
    "fixintanotherassetacquicash": "CONSTRUCT_LONG_ASSET",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def list_symbol_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.csv"))


def load_minute_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["symbol"] = frame["symbol"].str.zfill(6)
    return frame.sort_values("datetime")


def load_daily_file(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].str.zfill(6)
    return frame.sort_values("date")


def convert_minute_directory_to_wide(minute_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    minute_dir = Path(minute_dir)
    output_dir = ensure_dir(output_dir)
    files = list_symbol_files(minute_dir)
    if not files:
        raise FileNotFoundError(f"No minute CSV files found in {minute_dir}")

    field_frames: dict[str, list[pd.DataFrame]] = {field: [] for field in ["open", "high", "low", "close", "volume", "amount", "vwap"]}
    for file in files:
        df = load_minute_file(file)
        symbol = df["symbol"].iloc[0]
        for field in field_frames:
            one = df[["datetime", field]].rename(columns={field: symbol}).set_index("datetime")
            field_frames[field].append(one)

    written: dict[str, Path] = {}
    for field, parts in field_frames.items():
        wide = pd.concat(parts, axis=1).sort_index()
        target = output_dir / f"{field}.csv"
        wide.to_csv(target, encoding="utf-8-sig")
        written[field] = target
    return written


def build_daily_panel(daily_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    files = list_symbol_files(daily_dir)
    if not files:
        raise FileNotFoundError(f"No daily CSV files found in {daily_dir}")

    frames = []
    for file in files:
        frame = load_daily_file(file)
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    save_csv(panel, output_path)
    return panel


def load_financial_table(symbol_dir: Path, table_name: str) -> pd.DataFrame:
    path = symbol_dir / f"{table_name}.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def merge_symbol_fundamentals(symbol_dir: str | Path, symbol: str) -> pd.DataFrame:
    symbol_dir = Path(symbol_dir)
    balance = load_financial_table(symbol_dir, "balance_sheet")
    profit = load_financial_table(symbol_dir, "profit_sheet")
    cash_flow = load_financial_table(symbol_dir, "cash_flow_sheet")
    indicator = load_financial_table(symbol_dir, "financial_indicator")

    if balance.empty and profit.empty and cash_flow.empty:
        raise FileNotFoundError(f"No usable financial tables found in {symbol_dir}")

    key_cols = ["SECUCODE", "SECURITY_CODE", "SECURITY_NAME_ABBR", "REPORT_DATE", "REPORT_TYPE", "NOTICE_DATE", "UPDATE_DATE"]
    merged = None
    for table in [balance, profit, cash_flow]:
        if table.empty:
            continue
        if merged is None:
            merged = table.copy()
        else:
            extra_cols = [c for c in table.columns if c not in merged.columns or c in key_cols]
            merged = merged.merge(table[extra_cols], on=[c for c in key_cols if c in table.columns and c in merged.columns], how="outer")

    if merged is None:
        merged = pd.DataFrame()

    if not indicator.empty and "日期" in indicator.columns:
        indicator = indicator.copy()
        indicator["REPORT_DATE"] = pd.to_datetime(indicator["日期"], errors="coerce")
        indicator["symbol"] = symbol
        for source, target in {"每股收益": "basiceps"}.items():
            if source in indicator.columns:
                merged = merged.merge(
                    indicator[["REPORT_DATE", source]].rename(columns={source: target}),
                    on="REPORT_DATE",
                    how="left",
                )

    merged = merged.copy()
    base = {
        "date": pd.to_datetime(merged["REPORT_DATE"], errors="coerce"),
        "asset": symbol.zfill(6),
    }
    mapped = {
        target_name: pd.to_numeric(merged[source_name], errors="coerce")
        for target_name, source_name in FUNDAMENTAL_COLUMN_MAP.items()
        if source_name in merged.columns
    }
    output = pd.DataFrame({**base, **mapped})

    # Extra aliases that are often useful when the preferred source is absent.
    if "sewithoutmi" not in output.columns and "PARENT_EQUITY_BALANCE" in merged.columns:
        output["sewithoutmi"] = pd.to_numeric(merged["PARENT_EQUITY_BALANCE"], errors="coerce")
    if "netoperatecashflow" not in output.columns and "NETCASH_OPERATE" in merged.columns:
        output["netoperatecashflow"] = pd.to_numeric(merged["NETCASH_OPERATE"], errors="coerce")
    if "dividend_ttm" not in output.columns:
        output["dividend_ttm"] = pd.NA

    output = output.sort_values("date").dropna(subset=["date"])
    output = (
        output.groupby(["date", "asset"], as_index=False)
        .agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NA)
        .sort_values("date")
    )
    return output


def build_fundamental_panel(fundamental_root: str | Path, output_path: str | Path) -> pd.DataFrame:
    root = Path(fundamental_root)
    symbol_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not symbol_dirs:
        raise FileNotFoundError(f"No symbol directories found in {root}")

    panels: list[pd.DataFrame] = []
    for symbol_dir in symbol_dirs:
        symbol = str(symbol_dir.name).zfill(6)
        try:
            panels.append(merge_symbol_fundamentals(symbol_dir, symbol))
        except FileNotFoundError:
            continue

    if not panels:
        raise RuntimeError("No usable fundamental panels were built.")

    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "asset"])
    save_csv(panel, output_path)
    return panel


def attach_market_value_and_returns(
    fundamental_panel_path: str | Path,
    daily_panel_path: str | Path,
    output_path: str | Path,
    price_column: str = "close",
) -> pd.DataFrame:
    fundamental = pd.read_csv(fundamental_panel_path, dtype={"asset": str})
    daily = pd.read_csv(daily_panel_path, dtype={"symbol": str})

    fundamental["date"] = pd.to_datetime(fundamental["date"])
    fundamental["asset"] = fundamental["asset"].str.zfill(6)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = daily["symbol"].str.zfill(6)

    if "symbol" not in daily.columns:
        raise ValueError("daily panel must include a symbol column")
    if price_column not in daily.columns:
        raise ValueError(f"daily panel missing price column: {price_column}")

    daily = daily.sort_values(["symbol", "date"])
    daily["ret_1d"] = daily.groupby("symbol")[price_column].pct_change()
    daily["mv"] = daily["close"] * pd.to_numeric(daily.get("volume"), errors="coerce")

    join_cols = ["date", "symbol", "close", "volume", "mv", "ret_1d"]
    merged_parts: list[pd.DataFrame] = []
    for asset, sub_fundamental in fundamental.sort_values("date").groupby("asset", sort=False):
        sub_daily = daily.loc[daily["symbol"] == asset, join_cols].sort_values("date")
        sub = pd.merge_asof(
            sub_fundamental.sort_values("date"),
            sub_daily,
            on="date",
            direction="backward",
        )
        merged_parts.append(sub)

    merged = pd.concat(merged_parts, ignore_index=True).sort_values(["date", "asset"])
    merged["vwap"] = merged["close"]
    merged["ret_5d_fwd"] = merged.groupby("asset")["close"].shift(-5) / merged["close"] - 1.0
    save_csv(merged, output_path)
    return merged


def build_daily_mv_panel_from_balance(fundamental_panel: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    panel = fundamental_panel.copy()
    if "mv" not in panel.columns:
        panel["mv"] = pd.NA
    daily = panel[["date", "asset", "mv"]].dropna(subset=["date"]).sort_values(["date", "asset"])
    save_csv(daily, output_path)
    return daily


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare fetched A-share raw data for factor strategy inputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    minute_parser = subparsers.add_parser("minute-wide", help="Convert per-symbol minute CSV files to wide matrices.")
    minute_parser.add_argument("--minute-dir", required=True)
    minute_parser.add_argument("--output-dir", required=True)

    fundamental_parser = subparsers.add_parser("fundamental-panel", help="Merge raw financial tables into one panel CSV.")
    fundamental_parser.add_argument("--fundamental-dir", required=True)
    fundamental_parser.add_argument("--output", required=True)

    daily_parser = subparsers.add_parser("daily-panel", help="Merge per-symbol daily CSV files into one panel CSV.")
    daily_parser.add_argument("--daily-dir", required=True)
    daily_parser.add_argument("--output", required=True)

    attach_parser = subparsers.add_parser("attach-market", help="Attach daily prices and labels to the fundamental panel.")
    attach_parser.add_argument("--fundamental-panel", required=True)
    attach_parser.add_argument("--daily-panel", required=True)
    attach_parser.add_argument("--output", required=True)

    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "minute-wide":
        written = convert_minute_directory_to_wide(args.minute_dir, args.output_dir)
        print("saved wide minute matrices:", ", ".join(f"{k}={v}" for k, v in written.items()))
        return 0

    if args.command == "fundamental-panel":
        panel = build_fundamental_panel(args.fundamental_dir, args.output)
        print(f"saved fundamental panel to {args.output}: {panel.shape}")
        return 0

    if args.command == "daily-panel":
        panel = build_daily_panel(args.daily_dir, args.output)
        print(f"saved daily panel to {args.output}: {panel.shape}")
        return 0

    if args.command == "attach-market":
        panel = attach_market_value_and_returns(args.fundamental_panel, args.daily_panel, args.output)
        print(f"saved attached panel to {args.output}: {panel.shape}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
