import os
import ssl
import smtplib
from datetime import datetime
from typing import List, Dict
import time
import json
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf

# Default tickers: Micro equity futures
DEFAULT_TICKERS = "MNQ=F,MES=F,MYM=F,M2K=F"


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and (val is None or (isinstance(val, str) and val.strip() == "")):
        raise SystemExit(f"Missing required environment variable: {name}")
    if isinstance(val, str):
        return val.strip()
    return val


def send_email(subject: str, body: str) -> None:
    server = _get_env("SMTP_SERVER", "smtp.gmail.com")
    port = int(_get_env("SMTP_PORT", "465"))
    username = _get_env("SMTP_USERNAME", required=True)
    password = _get_env("SMTP_PASSWORD", required=True)
    sender = _get_env("EMAIL_FROM", username)
    to_csv = _get_env("EMAIL_TO", required=True)
    recipients = [x.strip() for x in to_csv.split(",") if x.strip()]

    msg = (
        f"From: {sender}\r\n"
        f"To: {', '.join(recipients)}\r\n"
        f"Subject: {subject}\r\n"
        "\r\n"
        f"{body}"
    )

    if port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(server, port, context=context) as smtp:
            smtp.login(username, password)
            smtp.sendmail(sender, recipients, msg.encode("utf-8"))
    else:
        with smtplib.SMTP(server, port) as smtp:
            smtp.ehlo()
            if port == 587:
                smtp.starttls(context=ssl.create_default_context())
                smtp.ehlo()
            if username and password:
                smtp.login(username, password)
            smtp.sendmail(sender, recipients, msg.encode("utf-8"))


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def fetch_daily(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")

    # Handle possible MultiIndex columns returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        cols0 = df.columns.get_level_values(0)
        cols1 = df.columns.get_level_values(1)
        if "Close" in cols0 or "close" in cols0.str.lower():
            df = df.droplevel(1, axis=1)
        elif "Close" in cols1 or "close" in cols1.str.lower():
            df = df.droplevel(0, axis=1)
        else:
            df.columns = [str(c[0]) for c in df.columns]

    # Find canonical column names regardless of exact case
    def _find_col(name: str) -> str:
        lname = name.lower()
        for c in df.columns:
            if str(c).strip().lower() == lname:
                return c
        raise RuntimeError(f"Missing column '{name}' in downloaded data")

    close_col = _find_col("Close")
    high_col = _find_col("High")
    low_col = _find_col("Low")

    df["SMA20"] = df[close_col].rolling(20).mean()
    df["SMA50"] = df[close_col].rolling(50).mean()
    df["RSI14"] = _rsi(df[close_col], 14)
    df["HH20"] = df[high_col].rolling(20).max()
    df["LL20"] = df[low_col].rolling(20).min()
    return df.dropna().copy()


def detect_signals(df: pd.DataFrame) -> List[str]:
    if len(df) < 3:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signals: List[str] = []

    if last.SMA20 > last.SMA50 and prev.SMA20 <= prev.SMA50:
        signals.append("Bullish 20/50 SMA cross")
    if last.SMA20 < last.SMA50 and prev.SMA20 >= prev.SMA50:
        signals.append("Bearish 20/50 SMA cross")

    if last.RSI14 > 70 and prev.RSI14 <= 70:
        signals.append("RSI > 70 (overbought)")
    if last.RSI14 < 30 and prev.RSI14 >= 30:
        signals.append("RSI < 30 (oversold)")

    if last.Close > prev.HH20 and prev.Close <= prev.HH20:
        signals.append("20-day breakout up")
    if last.Close < prev.LL20 and prev.Close >= prev.LL20:
        signals.append("20-day breakdown down")

    return signals


def build_report(tickers: List[str]) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for t in tickers:
        try:
            df = fetch_daily(t)
            sigs = detect_signals(df)
            if sigs:
                results[t] = sigs
        except Exception as e:
            # Include errors as notes so you notice issues
            results[t] = [f"Error: {e}"]
    return results


def main() -> int:
    tickers_csv = _get_env("YF_TICKERS", DEFAULT_TICKERS) or DEFAULT_TICKERS
    tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]

    findings = build_report(tickers)

    date_str = datetime.now().strftime("%Y-%m-%d")
    # Separate real signals from error notes so errors don't count as alerts
    signals_by_ticker: Dict[str, List[str]] = {}
    errors_by_ticker: Dict[str, List[str]] = {}
    for t, items in findings.items():
        if not items:
            continue
        sigs = [x for x in items if not isinstance(x, str) or not x.startswith("Error:")]
        errs = [x for x in items if isinstance(x, str) and x.startswith("Error:")]
        if sigs:
            signals_by_ticker[t] = sigs
        if errs:
            errors_by_ticker[t] = errs

    total_alerts = sum(len(v) for v in signals_by_ticker.values())
    subject = f"Signals: {len(tickers)} tickers, {total_alerts} alerts ({date_str})"

    lines: List[str] = []
    for t in tickers:
        if t in signals_by_ticker:
            for s in signals_by_ticker[t]:
                lines.append(f"- {t}: {s}")
    if not lines:
        lines.append("No signals today.")

    # Optionally include error notes at the end of the email body (informational)
    include_errors = (_get_env("INCLUDE_ERRORS_IN_BODY", "1") == "1")
    if include_errors and errors_by_ticker:
        lines.append("")
        lines.append("Notes (not counted as alerts):")
        for t in tickers:
            if t in errors_by_ticker:
                for e in errors_by_ticker[t]:
                    lines.append(f"- {t}: {e}")

    body = "\n".join(lines)

    # Only send when there are alerts by default; override with ALWAYS_SEND=1
    always_send = (_get_env("ALWAYS_SEND", "0") == "1")
    # Allow sending if there are only errors (no signals) when SEND_ON_ERROR=1
    send_on_error = (_get_env("SEND_ON_ERROR", "0") == "1")

    # Duplicate suppression: avoid sending identical content too frequently
    disable_dedup = (_get_env("DISABLE_DEDUP", "0") == "1")
    min_interval_s = int(_get_env("MIN_SEND_INTERVAL_SECONDS", "86400") or "86400")
    state_path = _get_env(
        "STATE_FILE",
        os.path.join(os.path.dirname(__file__), ".signal_emailer_state.json"),
    )

    def _load_state(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(path: str, state: dict) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        except Exception:
            pass

    content_hash = hashlib.sha256((subject + "\n" + body).encode("utf-8")).hexdigest()
    now_ts = int(time.time())
    state = _load_state(state_path)
    last_hash = state.get("last_hash")
    last_sent = int(state.get("last_sent", 0))

    has_errors_only = (total_alerts == 0 and len(errors_by_ticker) > 0)
    should_send_now = (total_alerts > 0 or always_send or (send_on_error and has_errors_only))

    # Optional debug logging of decision inputs
    debug = (_get_env("DEBUG", "0") == "1")
    if debug:
        print("[DEBUG] total_alerts=", total_alerts)
        print("[DEBUG] errors_only=", has_errors_only, "errors_count=", sum(len(v) for v in errors_by_ticker.values()))
        print("[DEBUG] always_send=", always_send, "send_on_error=", send_on_error)
        print("[DEBUG] disable_dedup=", disable_dedup, "min_interval_s=", min_interval_s)
    if should_send_now and not disable_dedup:
        if last_hash == content_hash and (now_ts - last_sent) < min_interval_s:
            should_send_now = False
            remaining = max(0, min_interval_s - (now_ts - last_sent))
            print(
                f"Skipping duplicate email (next allowed in ~{remaining} sec)."
            )

    if should_send_now:
        # Dry run mode for safe verification
        dry_run = (_get_env("DRY_RUN", "0") == "1")
        if dry_run:
            print("[DRY_RUN] Would send email with subject:", subject)
            print("[DRY_RUN] Body:\n" + body)
        else:
            send_email(subject, body)
            print(f"Email sent: {subject}")
        # update state
        state.update({"last_hash": content_hash, "last_sent": now_ts})
        _save_state(state_path, state)
    else:
        if total_alerts == 0 and not always_send and not (send_on_error and has_errors_only):
            print("No alerts; not sending email (set ALWAYS_SEND=1 or SEND_ON_ERROR=1 to force).")

    return 0


if __name__ == "__main__":
    # Looping is now opt-in: provide CHECK_INTERVAL_SECONDS>0 to enable.
    interval_raw = _get_env("CHECK_INTERVAL_SECONDS", None)
    interval_s = int(interval_raw) if interval_raw not in (None, "", "0") else 0
    try:
        if interval_s > 0:
            while True:
                main()
                # Avoid spamming if ALWAYS_SEND=1; dedup also protects identical content
                time.sleep(max(5, interval_s))
        else:
            # Single run (recommended to use OS scheduler for cadence)
            main()
    except KeyboardInterrupt:
        pass
