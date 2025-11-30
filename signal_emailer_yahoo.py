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

# Default strategy params
DEFAULT_BASE_INTERVAL = os.environ.get("BASE_INTERVAL", "1m").strip() or "1m"
DEFAULT_CONTEXT_INTERVAL = os.environ.get("CONTEXT_INTERVAL", "60m").strip() or "60m"
DONCHIAN_N = int(os.environ.get("DONCHIAN_N", "20") or "20")
EMA_N = int(os.environ.get("EMA_N", "50") or "50")
ATR_N = int(os.environ.get("ATR_N", "14") or "14")

# Sizing params (informational only; no trade placement)
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "100"))
MIN_QTY = int(os.environ.get("MIN_QTY", "1"))
MAX_CONTRACTS = int(os.environ.get("MAX_CONTRACTS", "2"))


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and (val is None or (isinstance(val, str) and val.strip() == "")):
        raise SystemExit(f"Missing required environment variable: {name}")
    if isinstance(val, str):
        return val.strip()
    return val

# Optional file logging helper. Set LOG_FILE to a path to enable.
LOG_FILE = os.environ.get("LOG_FILE")

def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    finally:
        lf = os.environ.get("LOG_FILE", LOG_FILE)
        if lf:
            try:
                with open(lf, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass


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


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # expects columns High, Low, Close (case-insensitive handled before)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _resolve_period_for_interval(interval: str, lookback_days: int = 120) -> str:
    # Allow global override
    global_override = _get_env("YF_PERIOD", None)
    # Allow per-interval override: YF_PERIOD_5M, YF_PERIOD_1H, etc.
    key = f"YF_PERIOD_{interval.upper().replace('/', '_')}"
    per_override = _get_env(key, None)
    if per_override and per_override.strip():
        return per_override
    if global_override and global_override.strip():
        return global_override
    # Sensible defaults
    if interval.endswith("m"):
        # Intraday minutes: Yahoo restricts 1m to ~7d; others are also limited
        return "7d"
    if interval.endswith("h") or interval in ("60m", "90m", "1h"):
        return "60d"
    # Daily or higher
    return f"{lookback_days}d"


def fetch_bars(ticker: str, interval: str, lookback_days: int = 120) -> pd.DataFrame:
    yf_interval = interval
    yf_period = _resolve_period_for_interval(yf_interval, lookback_days)

    df = yf.download(
        ticker,
        period=yf_period,
        interval=yf_interval,
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

    # Normalize canonical caps used later
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})
    if high_col != "High":
        df = df.rename(columns={high_col: "High"})
    if low_col != "Low":
        df = df.rename(columns={low_col: "Low"})

    return df.copy()


def _point_value_for_ticker(ticker: str) -> float:
    # Allow overrides per ticker
    key = f"POINT_VALUE_{ticker.replace('=','_').replace('-','_').upper()}"
    ov = os.environ.get(key)
    if ov:
        try:
            return float(ov)
        except Exception:
            pass
    gen = os.environ.get("POINT_VALUE")
    if gen:
        try:
            return float(gen)
        except Exception:
            pass
    mapping = {
        "MNQ=F": 2.0,
        "MES=F": 5.0,
        "MYM=F": 0.5,
        "M2K=F": 5.0,
    }
    return mapping.get(ticker.upper(), 1.0)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _sizing(atr_points: float, point_value: float) -> int:
    if atr_points <= 0 or point_value <= 0:
        return 0
    qty = int(np.floor(RISK_PER_TRADE / (2.0 * atr_points * point_value)))
    return _clamp(qty, MIN_QTY, MAX_CONTRACTS) if qty > 0 else 0


def _donchian(df: pd.DataFrame, n: int) -> tuple[pd.Series, pd.Series]:
    hh = df["High"].rolling(n).max()
    ll = df["Low"].rolling(n).min()
    return hh, ll


def _add_context_and_indicators(df_1m: pd.DataFrame, df_60m: pd.DataFrame) -> dict:
    # Compute hourly EMA and slope, hourly ATR as context
    ema50_h = _ema(df_60m["Close"], EMA_N)
    ema50_slope_h = ema50_h - ema50_h.shift(1)
    atr_h = _atr(df_60m, ATR_N)
    context = {
        "ema50_h": ema50_h,
        "ema50_slope_h": ema50_slope_h,
        "atr_h": atr_h,
    }
    # 1m indicators
    atr_1m = _atr(df_1m, ATR_N)
    hh20, ll20 = _donchian(df_1m, DONCHIAN_N)
    ind = {
        "atr_1m": atr_1m,
        "hh": hh20,
        "ll": ll20,
    }
    return {**context, **ind}


def _get_hourly_context_at(df_60m: pd.DataFrame, ema50_h: pd.Series, ema50_slope_h: pd.Series, atr_h: pd.Series, ts: pd.Timestamp) -> tuple[float, float, float]:
    # Align 1m ts to last available hourly bar at or before ts
    idx = df_60m.index.get_indexer([ts], method="pad")
    pos = idx[0] if len(idx) else -1
    if pos < 0:
        return np.nan, np.nan, np.nan
    return float(ema50_h.iloc[pos]), float(ema50_slope_h.iloc[pos]), float(atr_h.iloc[pos])


def _build_entry_signal(ticker: str, df_1m: pd.DataFrame, df_60m: pd.DataFrame) -> dict | None:
    if len(df_1m) < max(DONCHIAN_N + 1, ATR_N + 1) or len(df_60m) < max(EMA_N + 1, ATR_N + 1):
        return None
    inds = _add_context_and_indicators(df_1m, df_60m)
    last = df_1m.iloc[-1]
    prev = df_1m.iloc[-2]
    ts = df_1m.index[-1]

    ema50_h, ema50_slope, atr_h = _get_hourly_context_at(
        df_60m, inds["ema50_h"], inds["ema50_slope_h"], inds["atr_h"], ts
    )
    if np.isnan(ema50_h) or np.isnan(ema50_slope):
        return None

    hh_prev = inds["hh"].iloc[-2]
    ll_prev = inds["ll"].iloc[-2]
    atr_1m = float(inds["atr_1m"].iloc[-1])
    pv = _point_value_for_ticker(ticker)
    qty = _sizing(atr_1m, pv)
    if qty <= 0:
        return None

    long_ok = (last["Close"] > hh_prev) and (last["Close"] > ema50_h) and (ema50_slope > 0)
    short_ok = (last["Close"] < ll_prev) and (last["Close"] < ema50_h) and (ema50_slope < 0)

    if long_ok:
        return {
            "dir": "long",
            "entry": float(last["Close"]),
            "atr": atr_1m,
            "qty": qty,
            "pv": pv,
            "ts": ts,
            "ema50_h": float(ema50_h),
            "ema_slope": float(ema50_slope),
            "atr_h": float(atr_h) if not np.isnan(atr_h) else None,
        }
    if short_ok:
        return {
            "dir": "short",
            "entry": float(last["Close"]),
            "atr": atr_1m,
            "qty": qty,
            "pv": pv,
            "ts": ts,
            "ema50_h": float(ema50_h),
            "ema_slope": float(ema50_slope),
            "atr_h": float(atr_h) if not np.isnan(atr_h) else None,
        }
    return None


    


def _fmt_ts(ts: pd.Timestamp) -> str:
    try:
        if getattr(ts, 'tzinfo', None) is not None and ts.tzinfo is not None:
            return ts.strftime('%Y-%m-%d %H:%M %Z')
        return ts.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return str(ts)


def build_report(tickers: List[str]) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for t in tickers:
        bucket: List[str] = []
        try:
            df1 = fetch_bars(t, DEFAULT_BASE_INTERVAL)
            df60 = fetch_bars(t, DEFAULT_CONTEXT_INTERVAL, lookback_days=400)
            atr_1m = _atr(df1, ATR_N)
            hh, ll = _donchian(df1, DONCHIAN_N)
            df1 = df1.assign(ATR1m=atr_1m, HH=hh, LL=ll).dropna()
            if len(df1) == 0:
                continue
            sig = _build_entry_signal(t, df1, df60)
            if sig:
                stop = sig["entry"] - 2 * sig["atr"] if sig["dir"] == "long" else sig["entry"] + 2 * sig["atr"]
                bucket.append(
                    f"[{DEFAULT_BASE_INTERVAL}] entry {sig['dir']} @ {sig['entry']:.2f} (bar={_fmt_ts(sig['ts'])}) | ATR={sig['atr']:.2f} stop={stop:.2f} qty={sig['qty']} (EMA50h={sig['ema50_h']:.2f} slope={sig['ema_slope']:+.2f})"
                )
        except Exception as e:
            bucket.append(f"[{DEFAULT_BASE_INTERVAL}] Error: {e}")
        if bucket:
            results[t] = bucket
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

    # Heartbeat: always print a summary each check so you know it's alive
    hb = (_get_env("HEARTBEAT", "1") == "1")
    if hb:
        errors_count = sum(len(v) for v in errors_by_ticker.values())
        _log(
            f"[HEARTBEAT] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"tickers={len(tickers)} alerts={total_alerts} errors={errors_count} "
            f"base_interval={DEFAULT_BASE_INTERVAL} ctx_interval={DEFAULT_CONTEXT_INTERVAL}"
        )

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
        _log(f"[DEBUG] total_alerts= {total_alerts}")
        _log(f"[DEBUG] errors_only= {has_errors_only} errors_count= {sum(len(v) for v in errors_by_ticker.values())}")
        _log(f"[DEBUG] always_send= {always_send} send_on_error= {send_on_error}")
        _log(f"[DEBUG] disable_dedup= {disable_dedup} min_interval_s= {min_interval_s}")
    if should_send_now and not disable_dedup:
        if last_hash == content_hash and (now_ts - last_sent) < min_interval_s:
            should_send_now = False
            remaining = max(0, min_interval_s - (now_ts - last_sent))
            _log(f"Skipping duplicate email (next allowed in ~{remaining} sec).")

    if should_send_now:
        # Dry run mode for safe verification
        dry_run = (_get_env("DRY_RUN", "0") == "1")
        if dry_run:
            _log(f"[DRY_RUN] Would send email with subject: {subject}")
            _log("[DRY_RUN] Body:\n" + body)
        else:
            send_email(subject, body)
            _log(f"Email sent: {subject}")
        # update state
        state.update({"last_hash": content_hash, "last_sent": now_ts})
        _save_state(state_path, state)
    else:
        if total_alerts == 0 and not always_send and not (send_on_error and has_errors_only):
            _log("No alerts; not sending email (set ALWAYS_SEND=1 or SEND_ON_ERROR=1 to force).")

    return 0


if __name__ == "__main__":
    # Looping default: check every 60s unless disabled with CHECK_INTERVAL_SECONDS=0
    interval_s = int(_get_env("CHECK_INTERVAL_SECONDS", "60") or "60")
    try:
        if interval_s > 0:
            hb = (_get_env("HEARTBEAT", "1") == "1")
            if hb:
                _log(f"[HEARTBEAT] service start; interval={interval_s}s")
            while True:
                main()
                # Avoid spamming if ALWAYS_SEND=1; dedup also protects identical content
                hb = (_get_env("HEARTBEAT", "1") == "1")
                if hb:
                    _log(f"[HEARTBEAT] next check in {max(5, interval_s)} sec")
                time.sleep(max(5, interval_s))
        else:
            # Single run (use OS scheduler for cadence)
            main()
    except KeyboardInterrupt:
        pass
