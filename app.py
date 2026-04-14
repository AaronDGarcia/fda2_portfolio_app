import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import random


@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date, *, max_retries=3, per_ticker_retries=2):
    """Download adjusted close prices for tickers and the S&P 500 benchmark.
    Returns (prices_df, errors_list, warnings_list).
    - errors_list: tickers that completely failed to download
    - warnings_list: messages about truncated ranges or dropped tickers
    """
    warnings = []
    if not tickers:
        return None, ["No tickers provided"], warnings

    assets = list(tickers)
    benchmark = "^GSPC"
    all_tickers = assets + [benchmark]

    # Attempt a single batched download with retries/backoff
    raw = None
    attempt = 0
    while attempt < max_retries:
        try:
            raw = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            break
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raw = None
                break
            backoff = min(2 ** (attempt - 1), 8)
            time.sleep(backoff + random.uniform(0, 0.3))

    if raw is None:
        # final attempt: try per-ticker to identify failures
        errors = []
        frames = {}
        for t in all_tickers:
            ok = False
            for i in range(per_ticker_retries):
                try:
                    r = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    if isinstance(r, pd.Series):
                        s = r
                    elif isinstance(r, pd.DataFrame):
                        # try to extract Close/Adj Close if present
                        if ("Close" in r.columns):
                            s = r["Close"].copy()
                        else:
                            s = r.iloc[:, 0]
                    else:
                        s = pd.Series(dtype=float)
                    if s.dropna().shape[0] >= 2:
                        frames[t] = s
                        ok = True
                        break
                except Exception:
                    time.sleep(0.5 + random.uniform(0, 0.2))
            if not ok:
                errors.append(t)

        if len(errors) == len(all_tickers):
            return None, errors, ["Batched download failed and all per-ticker attempts failed."]

        # assemble DataFrame from individual series
        if frames:
            df = pd.concat(frames.values(), axis=1)
            df.columns = list(frames.keys())
        else:
            df = pd.DataFrame()

    else:
        # normalize raw to a simple prices DataFrame
        if isinstance(raw, pd.Series):
            df = raw.to_frame()
        elif isinstance(raw, pd.DataFrame):
            # yf may return multiindex (e.g., ('Close', 'AAPL')), handle common cases
            if getattr(raw.columns, 'nlevels', 1) > 1:
                if 'Close' in raw.columns.get_level_values(0):
                    df = raw['Close'].copy()
                elif 'Adj Close' in raw.columns.get_level_values(0):
                    df = raw['Adj Close'].copy()
                else:
                    # try to collapse by taking first subcolumn
                    df = raw.xs(raw.columns.levels[1][0], axis=1, level=1, drop_level=True)
            else:
                df = raw.copy()

        else:
            df = pd.DataFrame()

        # identify obviously missing tickers
        errors = []
        for t in all_tickers:
            if t not in df.columns or df[t].dropna().shape[0] < 2:
                errors.append(t)

        # If some tickers failed, attempt per-ticker retries to isolate failures
        if errors:
            still_errors = []
            for t in errors:
                ok = False
                for i in range(per_ticker_retries):
                    try:
                        r = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
                        if isinstance(r, pd.Series):
                            s = r
                        elif isinstance(r, pd.DataFrame):
                            if ('Close' in r.columns):
                                s = r['Close'].copy()
                            else:
                                s = r.iloc[:, 0]
                        else:
                            s = pd.Series(dtype=float)
                        if s.dropna().shape[0] >= 2:
                            df[t] = s
                            ok = True
                            break
                    except Exception:
                        time.sleep(0.5 + random.uniform(0, 0.2))
                if not ok:
                    still_errors.append(t)
            errors = still_errors

    # At this point df contains whatever we fetched; remove columns not in all_tickers
    df = df.loc[:, df.columns.intersection(all_tickers)]

    # Drop tickers (assets only) with >5% missing values (per project rules)
    drop_threshold = 0.05
    errors_out = []
    dropped = []
    for t in assets:
        if t in df.columns:
            missing_frac = df[t].isna().mean()
            if missing_frac > drop_threshold:
                df = df.drop(columns=[t])
                dropped.append(t)
    if dropped:
        warnings.append(f"Dropped tickers due to >5% missing data: {', '.join(dropped)}")

    # Compute overlapping valid date range across remaining asset columns
    asset_cols = [c for c in df.columns if c != benchmark]
    if asset_cols:
        first_valid = [df[c].first_valid_index() for c in asset_cols]
        last_valid = [df[c].last_valid_index() for c in asset_cols]
        # if any None (all NaN), treat as failure
        if any(x is None for x in first_valid + last_valid):
            # mark columns with no data as errors
            for c in asset_cols:
                if df[c].dropna().empty:
                    errors_out.append(c)
                    df = df.drop(columns=[c])
            if errors_out:
                warnings.append(f"Tickers removed due to no data: {', '.join(errors_out)}")
            asset_cols = [c for c in df.columns if c != benchmark]

        if asset_cols:
            start_common = max(df[c].first_valid_index() for c in asset_cols)
            end_common = min(df[c].last_valid_index() for c in asset_cols)
            if start_common and end_common and start_common < end_common:
                # truncate to overlapping range
                df = df.loc[start_common:end_common]
                warnings.append(f"Data truncated to overlapping date range: {start_common.date()} to {end_common.date()}")

    # Final errors list: include any tickers that ultimately have no or insufficient data
    final_errors = []
    for t in all_tickers:
        if t not in df.columns:
            final_errors.append(t)

    return df, final_errors, warnings


def main():
    st.set_page_config(page_title="Interactive Portfolio App", layout="wide")
    st.title("Interactive Portfolio Analytics — (scaffold)")

    # Sidebar inputs
    st.sidebar.header("Inputs")
    tickers_input = st.sidebar.text_area("Enter 3-10 tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    start = st.sidebar.date_input("Start date", value=date.today() - timedelta(days=365 * 3))
    end = st.sidebar.date_input("End date", value=date.today())
    rf_rate = st.sidebar.number_input("Annual risk-free rate (%)", value=2.0, format="%.2f")
    load_btn = st.sidebar.button("Load data")

    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_input.replace(";", ",").split(",") if t.strip()]

    # Layout: tabs for sections (placeholder content)
    tabs = st.tabs(["Inputs & Data", "Exploratory", "Risk", "Correlation", "Portfolio", "Sensitivity", "About"])

    with tabs[0]:
        st.header("Inputs & Data")
        st.write("Tickers:", tickers)
        st.write("Date range:", start, "to", end)
        st.write("Risk-free rate (annual %):", rf_rate)

        if load_btn:
            if len(tickers) < 3 or len(tickers) > 10:
                st.error("Please enter between 3 and 10 tickers.")
            elif (end - start).days < 365 * 2:
                st.error("Please select a date range of at least 2 years.")
            else:
                with st.spinner("Downloading data..."):
                    prices, errors, warnings = download_data(tickers, start, end)
                if errors:
                    st.error(f"Data download issues for: {', '.join(errors)}")
                else:
                    for w in warnings:
                        st.warning(w)
                    st.success("Data downloaded.")
                    st.dataframe(prices.head())
                    st.line_chart(prices.fillna(method="ffill"))

    # Placeholder content for other tabs
    with tabs[1]:
        st.header("Exploratory Analysis")
        st.info("Placeholder: summary statistics, cumulative wealth, distribution plots")

    with tabs[2]:
        st.header("Risk Analysis")
        st.info("Placeholder: rolling volatility, drawdowns, Sharpe/Sortino table")

    with tabs[3]:
        st.header("Correlation & Covariance")
        st.info("Placeholder: correlation heatmap, rolling correlation, covariance matrix")

    with tabs[4]:
        st.header("Portfolio Construction")
        st.info("Placeholder: equal-weight, GMV, tangency, PRC, efficient frontier, custom portfolio")

    with tabs[5]:
        st.header("Estimation Window Sensitivity")
        st.info("Placeholder: compare different lookback windows for optimization inputs")

    with tabs[6]:
        st.header("About / Methodology")
        st.markdown("""
        This app will follow the course instructions: use adjusted close prices, simple returns,
        annualize by 252, convert annual RF to daily by dividing by 252, and cache expensive
        operations with `@st.cache_data`.
        """)


if __name__ == "__main__":
    main()
