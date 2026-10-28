from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import akshare as ak
import baostock as bs
import pandas as pd


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if len(symbol) == 8 and symbol[:2].upper() in {"SH", "SZ"} and symbol[2:].isdigit():
        return symbol.upper()
    if "." in symbol:
        market, code = symbol.split(".", 1)
        market = market.upper()
        return f"{market}{code}"
    if symbol.startswith(("6", "9")):
        return f"SH{symbol}"
    return f"SZ{symbol}"


def to_baostock_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)
    return f"{normalized[:2].lower()}.{normalized[2:]}"


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def try_akshare_spot_list() -> pd.DataFrame:
    df = ak.stock_zh_a_spot_em()
    renamed = df.rename(
        columns={
            "代码": "symbol",
            "名称": "name",
            "总市值": "total_mv",
            "流通市值": "circ_mv",
        }
    )
    keep = [col for col in ["symbol", "name", "total_mv", "circ_mv"] if col in renamed.columns]
    return renamed[keep].copy()


def fetch_symbol_list(limit: int | None = None) -> pd.DataFrame:
    try:
        universe = try_akshare_spot_list()
    except Exception:
        # Fallback universe from BaoStock basics when EastMoney is unavailable in the current network.
        login_result = bs.login()
        if login_result.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")
        rs = bs.query_all_stock(day="2025-06-30")
        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        universe = pd.DataFrame(rows, columns=rs.fields).rename(columns={"code": "symbol", "code_name": "name"})
        universe["symbol"] = universe["symbol"].str.replace(".", "", regex=False).str.upper()
        universe = universe[["symbol", "name"]]

    universe = universe.drop_duplicates(subset=["symbol"]).sort_values("symbol")
    if limit is not None:
        universe = universe.head(limit)
    return universe.reset_index(drop=True)


def fetch_baostock_minute(
    symbol: str,
    start_date: str,
    end_date: str,
    frequency: str = "5",
    adjustflag: str = "2",
) -> pd.DataFrame:
    bs_symbol = to_baostock_symbol(symbol)
    rs = bs.query_history_k_data_plus(
        bs_symbol,
        "date,time,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock minute query failed for {symbol}: {rs.error_msg}")

    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    frame = pd.DataFrame(rows, columns=rs.fields)
    if frame.empty:
        return frame

    frame["datetime"] = pd.to_datetime(frame["time"], format="%Y%m%d%H%M%S%f")
    frame["symbol"] = normalize_symbol(symbol)[2:]
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["vwap"] = frame["amount"] / frame["volume"].replace(0, pd.NA)
    return frame[["datetime", "date", "symbol", "open", "high", "low", "close", "volume", "amount", "vwap"]]


def fetch_baostock_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjustflag: str = "2",
) -> pd.DataFrame:
    bs_symbol = to_baostock_symbol(symbol)
    rs = bs.query_history_k_data_plus(
        bs_symbol,
        "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjustflag,
    )
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock daily query failed for {symbol}: {rs.error_msg}")

    rows: list[list[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    frame = pd.DataFrame(rows, columns=rs.fields)
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = normalize_symbol(symbol)[2:]
    numeric_cols = [col for col in frame.columns if col not in {"date", "code", "symbol"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def fetch_baostock_financial_bundle(symbol: str, year: int, quarter: int) -> dict[str, pd.DataFrame]:
    bs_symbol = to_baostock_symbol(symbol)
    loaders = {
        "profit": bs.query_profit_data,
        "operation": bs.query_operation_data,
        "growth": bs.query_growth_data,
        "balance": bs.query_balance_data,
        "cash_flow": bs.query_cash_flow_data,
        "dupont": bs.query_dupont_data,
    }
    bundle: dict[str, pd.DataFrame] = {}
    for name, loader in loaders.items():
        rs = loader(code=bs_symbol, year=year, quarter=quarter)
        if rs.error_code != "0":
            raise RuntimeError(f"BaoStock {name} query failed for {symbol} {year}Q{quarter}: {rs.error_msg}")
        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
        bundle[name] = pd.DataFrame(rows, columns=rs.fields)
    return bundle


def fetch_akshare_financial_bundle(symbol: str) -> dict[str, pd.DataFrame]:
    normalized = normalize_symbol(symbol)
    return {
        "balance_sheet": ak.stock_balance_sheet_by_report_em(symbol=normalized),
        "profit_sheet": ak.stock_profit_sheet_by_report_em(symbol=normalized),
        "cash_flow_sheet": ak.stock_cash_flow_sheet_by_report_em(symbol=normalized),
        "financial_indicator": ak.stock_financial_analysis_indicator(symbol=normalized[2:]),
    }


def fetch_and_store_minute_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    frequency: str = "5",
    sleep_seconds: float = 0.2,
) -> None:
    output_dir = ensure_dir(output_dir)
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")

    try:
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            frame = fetch_baostock_minute(symbol, start_date=start_date, end_date=end_date, frequency=frequency)
            save_csv(frame, output_dir / f"{symbol[2:]}.csv")
            print(f"saved minute data for {symbol}: {len(frame)} rows")
            time.sleep(sleep_seconds)
    finally:
        bs.logout()


def fetch_and_store_daily_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    sleep_seconds: float = 0.2,
) -> None:
    output_dir = ensure_dir(output_dir)
    login_result = bs.login()
    if login_result.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")

    try:
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            frame = fetch_baostock_daily(symbol, start_date=start_date, end_date=end_date)
            save_csv(frame, output_dir / f"{symbol[2:]}.csv")
            print(f"saved daily data for {symbol}: {len(frame)} rows")
            time.sleep(sleep_seconds)
    finally:
        bs.logout()


def fetch_and_store_financial_data(
    symbols: Iterable[str],
    output_dir: str | Path,
    year: int | None = None,
    quarter: int | None = None,
    sleep_seconds: float = 0.2,
) -> None:
    output_dir = ensure_dir(output_dir)

    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        symbol_dir = ensure_dir(output_dir / symbol[2:])

        try:
            bundle = fetch_akshare_financial_bundle(symbol)
            source = "akshare"
        except Exception as exc:
            if year is None or quarter is None:
                now = pd.Timestamp.today()
                if quarter is None:
                    quarter = max(1, min(4, (now.month - 1) // 3 + 1))
                if year is None:
                    year = int(now.year if quarter < 4 else now.year)
            login_result = bs.login()
            if login_result.error_code != "0":
                raise RuntimeError(f"BaoStock login failed: {login_result.error_msg}")
            try:
                bundle = fetch_baostock_financial_bundle(symbol, year=year, quarter=quarter)
                source = "baostock"
            finally:
                bs.logout()

        for name, frame in bundle.items():
            save_csv(frame, symbol_dir / f"{name}.csv")
        print(f"saved financial bundle for {symbol} via {source}")
        time.sleep(sleep_seconds)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch A-share minute and financial data for factor research.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    universe_parser = subparsers.add_parser("universe", help="Fetch a stock universe list.")
    universe_parser.add_argument("--output", default="data_raw/universe/a_share_universe.csv")
    universe_parser.add_argument("--limit", type=int)

    minute_parser = subparsers.add_parser("minute", help="Fetch minute bars using BaoStock.")
    minute_parser.add_argument("--symbols", nargs="+", required=True, help="Symbols like 000001 or SH600519.")
    minute_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    minute_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    minute_parser.add_argument("--frequency", default="5", choices=["5", "15", "30", "60"])
    minute_parser.add_argument("--output-dir", default="data_raw/minute")

    daily_parser = subparsers.add_parser("daily", help="Fetch daily bars using BaoStock.")
    daily_parser.add_argument("--symbols", nargs="+", required=True, help="Symbols like 000001 or SH600519.")
    daily_parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    daily_parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    daily_parser.add_argument("--output-dir", default="data_raw/daily")

    financial_parser = subparsers.add_parser("financial", help="Fetch financial reports and indicators.")
    financial_parser.add_argument("--symbols", nargs="+", required=True, help="Symbols like 000001 or SH600519.")
    financial_parser.add_argument("--output-dir", default="data_raw/fundamental")
    financial_parser.add_argument("--fallback-year", type=int)
    financial_parser.add_argument("--fallback-quarter", type=int, choices=[1, 2, 3, 4])

    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "universe":
        universe = fetch_symbol_list(limit=args.limit)
        save_csv(universe, args.output)
        print(f"saved universe list to {args.output}: {len(universe)} rows")
        return 0

    if args.command == "minute":
        fetch_and_store_minute_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            frequency=args.frequency,
        )
        return 0

    if args.command == "daily":
        fetch_and_store_daily_data(
            symbols=args.symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
        )
        return 0

    if args.command == "financial":
        fetch_and_store_financial_data(
            symbols=args.symbols,
            output_dir=args.output_dir,
            year=args.fallback_year,
            quarter=args.fallback_quarter,
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli())
