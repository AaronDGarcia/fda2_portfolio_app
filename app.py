import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import random
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import minimize


@st.cache_data(ttl=3600)
def download_data(tickers, start_date, end_date, *, max_retries=3, per_ticker_retries=2):
    """Download adjusted close prices for tickers and the S&P 500 benchmark.
    Returns (prices_df, errors_list, warnings_list).
    - errors_list: tickers that completely failed to download
    - warnings_list: messages about truncated ranges or dropped tickers
    """
    warnings = []
    # map ticker -> human-readable failure reason
    reasons = {}
    if not tickers:
        return None, {"<no_tickers>": "No tickers provided"}, warnings

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
                except Exception as e:
                    # record last exception message (trimmed)
                    reasons[t] = f"Network/error during download: {getattr(e, 'args', [''])[0]}"
                    time.sleep(0.5 + random.uniform(0, 0.2))
            if not ok:
                # if we didn't already set a reason above, set a generic one
                reasons.setdefault(t, "Download failed after retries or insufficient data")

        if len(frames) == 0:
            return None, reasons, ["Batched download failed and all per-ticker attempts failed."]

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

        # identify obviously missing tickers and mark reasons
        missing_candidates = []
        for t in all_tickers:
            if t not in df.columns or df[t].dropna().shape[0] < 2:
                # mark as insufficient in batched result and try per-ticker fetch
                reasons[t] = "Insufficient data in batched download"
                missing_candidates.append(t)

        # If some tickers failed, attempt per-ticker retries to isolate failures
        if missing_candidates:
            for t in missing_candidates:
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
                            # resolved — remove any previous reason
                            if t in reasons:
                                del reasons[t]
                            ok = True
                            break
                    except Exception as e:
                        reasons[t] = f"Per-ticker download error: {getattr(e, 'args', [''])[0]}"
                        time.sleep(0.5 + random.uniform(0, 0.2))
                if not ok:
                    reasons.setdefault(t, "Per-ticker retries failed: network or invalid symbol")

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
                reasons[t] = "Dropped: >5% missing data"
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

    # Final per-ticker reasons for any tickers not present in final df
    final_reasons = {}
    for t in all_tickers:
        if t not in df.columns:
            final_reasons[t] = reasons.get(t, "No usable data after processing")

    return df, final_reasons, warnings


def main():
    st.set_page_config(page_title="Interactive Portfolio App", layout="wide")
    st.title("Interactive Portfolio Analytics — (scaffold)")

    # Sidebar inputs
    st.sidebar.header("Inputs")
    # ticker entry UI: add/remove using session state list
    st.sidebar.subheader("Tickers")
    st.sidebar.caption('Example: "AAPL for Apple"')

    if 'tickers_list' not in st.session_state:
        st.session_state['tickers_list'] = ["AAPL", "MSFT", "GOOGL"]

    # text input for a new ticker (stored in session_state via key)
    st.sidebar.text_input("Add ticker individually (enter symbol, click Add Ticker)", value="", max_chars=10, key='new_ticker')

    def handle_add_ticker():
        try:
            t = (st.session_state.get('new_ticker', '') or '').strip().upper()
            import re
            if not t:
                st.sidebar.error("Please enter a ticker symbol before clicking Add.")
                return
            if not re.match(r'^[A-Z0-9\.\-]{1,10}$', t):
                st.sidebar.error("Ticker contains invalid characters. Use letters, numbers, ., or -.")
                return
            if t in st.session_state['tickers_list']:
                st.sidebar.info(f"{t} is already in the list.")
                return
            if len(st.session_state['tickers_list']) >= 10:
                st.sidebar.error("Maximum of 10 tickers allowed.")
                return
            st.session_state['tickers_list'].append(t)
            # clear the input field safely
            st.session_state['new_ticker'] = ''
        except Exception as e:
            st.sidebar.error(f"Failed to add ticker: {e}")

    st.sidebar.button("Add ticker", key='add_ticker', on_click=handle_add_ticker)

    # multiselect uses session_state key 'remove_sel'
    st.sidebar.multiselect("Remove tickers (select then click Remove)", options=list(st.session_state['tickers_list']), key='remove_sel')

    def handle_remove_tickers():
        try:
            remove_sel = st.session_state.get('remove_sel', []) or []
            for s in list(remove_sel):
                if s in st.session_state['tickers_list']:
                    st.session_state['tickers_list'].remove(s)
            # clear selection
            st.session_state['remove_sel'] = []
        except Exception as e:
            st.sidebar.error(f"Failed to remove selected tickers: {e}")

    st.sidebar.button("Remove selected", key='remove_tickers', on_click=handle_remove_tickers)

    st.sidebar.write("Tickers to load:", ", ".join(st.session_state['tickers_list']))

    start = st.sidebar.date_input("Start date", value=date.today() - timedelta(days=365 * 3))
    end = st.sidebar.date_input("End date", value=date.today())
    rf_rate = st.sidebar.number_input("Annual risk-free rate (%)", value=2.0, format="%.2f")
    load_btn = st.sidebar.button("Load data")

    # Use the managed tickers list
    tickers = list(st.session_state['tickers_list'])

    # Layout: tabs for sections (placeholder content)
    tabs = st.tabs(["Inputs & Data", "Exploratory", "Risk", "Correlation", "Portfolio", "Sensitivity", "About"])

    with tabs[0]:
        st.header("Inputs & Data")
        # Show tickers or resolved company names in the main tab
        st.write("Date range:", start, "to", end)
        st.write("Risk-free rate (annual %):", rf_rate)
        st.subheader("Tickers to load")
        if 'ticker_names' in st.session_state and st.session_state.get('ticker_names'):
            names_map = st.session_state['ticker_names']
            for sym in tickers:
                display_name = names_map.get(sym, sym)
                st.write(f"- {sym}: {display_name}")
        else:
            st.write(", ".join(tickers) if tickers else "(no tickers selected)")

        if load_btn:
            if len(tickers) < 3 or len(tickers) > 10:
                st.error("Please enter between 3 and 10 tickers.")
            elif (end - start).days < 365 * 2:
                st.error("Please select a date range of at least 2 years.")
            else:
                with st.spinner("Downloading data..."):
                    prices, error_map, warnings = download_data(tickers, start, end)

                # show any non-fatal warnings first
                for w in warnings:
                    st.warning(w)

                # render per-ticker problems (if any)
                if error_map:
                    st.error("Data download issues for the following tickers:")
                    for t, reason in error_map.items():
                        # treat drops/insufficient/no-data as warnings; others as errors
                        low_serv = any(k in reason for k in ("Dropped", "No data", "Insufficient"))
                        if low_serv:
                            st.warning(f"{t}: {reason}")
                        else:
                            st.error(f"{t}: {reason}")
                else:
                    st.success("Data downloaded.")
                    # persist prices for other tabs
                    st.session_state['prices'] = prices

                    # resolve company names for display in main tab (cache to session_state)
                    names_map = {}
                    for t in tickers:
                        try:
                            info = yf.Ticker(t).info
                            name = info.get('longName') or info.get('shortName') or t
                        except Exception:
                            name = t
                        names_map[t] = name
                    st.session_state['ticker_names'] = names_map

                    st.dataframe(prices.head())
                    # use DataFrame.ffill() for forward-fill compatibility
                    st.line_chart(prices.ffill())

    # Placeholder content for other tabs
    with tabs[1]:
        st.header("Exploratory Analysis")
        st.info("Summary statistics, cumulative wealth, and return distributions")

        prices = st.session_state.get('prices', None)
        if prices is None:
            st.warning("No price data loaded. Go to 'Inputs & Data' and click Load data.")
        else:
            # compute simple returns
            returns = prices.pct_change().dropna(how='all')

            # UI controls
            assets = [c for c in returns.columns]
            selected = st.selectbox("Select asset for distribution / charts", options=assets, index=0)
            # Wealth index overlay toggles (one checkbox per ticker)
            st.markdown("**Wealth index overlay (select tickers to include)**")
            st.caption("Wealth index shows the hypothetical growth of $10,000 invested at the start date for each selected asset (cumulative simple returns).")
            overlay_assets = []
            for c in assets:
                key = f"wealth_overlay_{c}"
                try:
                    sel = st.checkbox(f"Show {c} in Wealth Index", value=True, key=key)
                except Exception:
                    # fallback to key-less checkbox if key causes issues
                    sel = st.checkbox(f"Show {c} in Wealth Index", value=True)
                if sel:
                    overlay_assets.append(c)

            # Summary statistics (always shown)
            def compute_summary(rts, rf_annual_pct):
                rf_daily = rf_annual_pct / 100.0 / 252.0
                mean_d = rts.mean()
                std_d = rts.std()
                skew = rts.skew()
                kurt = rts.kurtosis()  # Fisher by default
                ann_ret = mean_d * 252.0
                ann_vol = std_d * np.sqrt(252.0)
                sharpe = (ann_ret - (rf_annual_pct / 100.0)) / ann_vol.replace(0, np.nan)
                tbl = pd.DataFrame({
                    'Mean (daily)': mean_d,
                    'Std (daily)': std_d,
                    'Skew': skew,
                    'Kurtosis': kurt,
                    'Ann. Return': ann_ret,
                    'Ann. Vol': ann_vol,
                    'Sharpe (ann)': sharpe
                })
                # Convert return values to percentages for display (keep skew/kurtosis/sharpe as-is)
                for col in ['Mean (daily)', 'Std (daily)', 'Ann. Return', 'Ann. Vol']:
                    tbl[col] = tbl[col] * 100.0
                return tbl

            stats_tbl = compute_summary(returns, rf_rate)
            st.subheader("Summary Statistics of Returns")
            st.dataframe(stats_tbl.style.format({
                'Mean (daily)': '{:.2f}%', 'Std (daily)': '{:.2f}%', 'Skew': '{:.2f}', 'Kurtosis': '{:.2f}',
                'Ann. Return': '{:.2f}%', 'Ann. Vol': '{:.2f}%', 'Sharpe (ann)': '{:.2f}'
            }))

            # Cumulative wealth (always shown)
            st.subheader("Cumulative Wealth Index (Start = $10,000)")
            wealth = (1 + returns).cumprod() * 10000.0
            fig_w = go.Figure()
            # plot only selected overlay assets for clarity
            for c in overlay_assets:
                if c in wealth.columns:
                    fig_w.add_trace(go.Scatter(x=wealth.index, y=wealth[c], mode='lines', name=c))
            fig_w.update_layout(yaxis_title='Wealth Index', xaxis_title='Date', height=420)
            st.plotly_chart(fig_w, use_container_width=True)

            # Show Histogram and Q-Q side-by-side
            st.subheader(f"Return Distribution & Q-Q: {selected}")
            series = returns[selected].dropna()
            if series.empty:
                st.warning("No return data for selected asset.")
            else:
                col1, col2 = st.columns(2)
                # Histogram + normal PDF
                with col1:
                    mu = series.mean()
                    sd = series.std()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=series, histnorm='probability density', name='Returns', nbinsx=50))
                    xs = np.linspace(series.min(), series.max(), 200)
                    pdf = stats.norm.pdf(xs, loc=mu, scale=sd)
                    fig.add_trace(go.Scatter(x=xs, y=pdf, mode='lines', name='Normal PDF', line=dict(color='red')))
                    fig.update_layout(xaxis_title='Daily Return', yaxis_title='Density', height=420)
                    st.plotly_chart(fig, use_container_width=True)

                # Q-Q plot
                with col2:
                    (osm, osr), (slope, intercept, r) = stats.probplot(series, dist='norm')
                    qq = go.Figure()
                    qq.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
                    qq.add_trace(go.Scatter(x=osm, y=intercept + slope * osm, mode='lines', name='Reference', line=dict(color='red')))
                    qq.update_layout(xaxis_title='Theoretical Quantiles', yaxis_title='Ordered Values', height=420)
                    st.plotly_chart(qq, use_container_width=True)

    with tabs[2]:
        st.header("Risk Analysis")
        st.info("Rolling volatility, drawdowns, Sharpe / Sortino, VaR")

        prices = st.session_state.get('prices', None)
        if prices is None:
            st.warning("No price data loaded. Go to 'Inputs & Data' and click Load data.")
        else:
            # returns (simple)
            returns = prices.pct_change().dropna(how='all')
            # Exclude benchmark ^GSPC from portfolio asset universe
            assets = [c for c in returns.columns if c != '^GSPC']

            # Controls: rolling window dropdown and drawdown asset dropdown
            cols = st.columns([1, 1, 1])
            with cols[0]:
                rolling_window = st.selectbox("Rolling window (days)", options=[30, 60, 90, 120], index=1)
            with cols[1]:
                dd_asset = st.selectbox("Drawdown asset", options=assets, index=0)
            with cols[2]:
                var_conf = st.selectbox("VaR confidence (%)", options=[90, 95, 99], index=1)

            # Compute rolling volatility (annualized)
            roll_std = returns.rolling(window=rolling_window).std() * np.sqrt(252.0)

            fig_rv = go.Figure()
            for a in assets:
                if a in roll_std.columns:
                    fig_rv.add_trace(go.Scatter(x=roll_std.index, y=roll_std[a], mode='lines', name=a))
            fig_rv.update_layout(title=f"Rolling annualized volatility ({rolling_window}-day)", yaxis_title='Annualized Vol (%)', height=420)
            st.plotly_chart(fig_rv, use_container_width=True)

            # Drawdown for selected asset
            s = returns[dd_asset].dropna()
            wealth = (1 + s).cumprod()
            running_max = wealth.cummax()
            drawdown = wealth / running_max - 1
            max_dd = drawdown.min()
            trough = drawdown.idxmin()
            peak = wealth.loc[:trough].idxmax() if not drawdown.empty else None

            # Prominently display Max Drawdown for selected asset
            if drawdown.empty or pd.isna(max_dd):
                st.subheader(f"Max Drawdown ({dd_asset}): N/A")
            else:
                st.subheader(f"Max Drawdown ({dd_asset})")
                st.metric(label=f"Max Drawdown ({dd_asset})", value=f"{max_dd * 100:.2f}%")

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100.0, mode='lines', fill='tozeroy', name=f'{dd_asset} drawdown'))
            if trough is not None:
                fig_dd.add_vline(x=trough, line=dict(color='red', dash='dash'))
                fig_dd.add_annotation(x=trough, y=(drawdown.loc[trough] * 100.0), text=f"Max DD {max_dd:.2%}\nTrough: {trough.date()}", showarrow=True, yanchor='bottom')
            fig_dd.update_layout(title=f"Drawdown: {dd_asset}", yaxis_title='Drawdown (%)', height=360)
            st.plotly_chart(fig_dd, use_container_width=True)

            # Risk metrics table for all assets
            def compute_metrics(col_series):
                mean_d = col_series.mean()
                std_d = col_series.std()
                ann_ret = mean_d * 252.0
                ann_vol = std_d * np.sqrt(252.0)
                sharpe = (ann_ret - (rf_rate / 100.0)) / ann_vol if ann_vol != 0 else np.nan
                # Sortino: downside std (daily) -> annualize
                downside = col_series[col_series < 0]
                downside_std = downside.std()
                downside_ann = downside_std * np.sqrt(252.0) if not np.isnan(downside_std) else np.nan
                sortino = (ann_ret - (rf_rate / 100.0)) / downside_ann if downside_ann and downside_ann != 0 else np.nan
                # Max drawdown
                w = (1 + col_series).cumprod()
                dd = w / w.cummax() - 1
                mdd = dd.min()
                # VaR (historical, daily) and CVaR
                alpha = 100 - int(var_conf)
                var = np.percentile(col_series.dropna(), alpha)
                cvar = col_series[col_series <= var].mean() if not col_series.dropna().empty else np.nan
                return {
                    'Ann. Vol': ann_vol * 100.0,
                    'Sharpe (ann)': sharpe,
                    'Sortino (ann)': sortino,
                    'Max Drawdown': mdd * 100.0,
                    f'VaR ({var_conf}%)': var * 100.0,
                    f'CVaR ({var_conf}%)': cvar * 100.0
                }

            metrics = {a: compute_metrics(returns[a].dropna()) for a in assets}
            metrics_df = pd.DataFrame(metrics).T

            # Format and display: percentages two decimals, ratios two decimals
            fmt = {}
            for col in ['Ann. Vol', 'Max Drawdown', f'VaR ({var_conf}%)', f'CVaR ({var_conf}%)']:
                if col in metrics_df.columns:
                    fmt[col] = '{:.2f}%'
            for col in ['Sharpe (ann)', 'Sortino (ann)']:
                if col in metrics_df.columns:
                    fmt[col] = '{:.2f}'

            st.subheader('Risk Metrics (per asset)')
            st.dataframe(metrics_df.style.format(fmt))

    with tabs[3]:
        st.header("Correlation & Covariance")
        st.info("Correlation matrix, rolling correlations, and covariance (expand to view)")

        prices = st.session_state.get('prices', None)
        if prices is None:
            st.warning("No price data loaded. Go to 'Inputs & Data' and click Load data.")
        else:
            returns = prices.pct_change().dropna(how='all')
            # Exclude benchmark ^GSPC from portfolio asset universe (benchmark is visual-only)
            assets = [c for c in returns.columns if c != '^GSPC']

            # Controls: asset selection, method, rolling window
            with st.sidebar.expander('Correlation & Covariance settings', expanded=False):
                sel_assets = st.multiselect('Select assets', options=assets, default=assets)
                corr_method = st.selectbox('Correlation method', options=['pearson', 'spearman'], index=0)
                rolling_window_corr = st.selectbox('Rolling window (days)', options=[30, 60, 90, 120], index=1)

            if not sel_assets:
                st.warning('Select at least one asset to compute correlations/covariance.')
            else:
                returns_sel = returns[sel_assets].dropna(how='all')

                # Correlation matrix
                corr = returns_sel.corr(method=corr_method)
                st.subheader('Correlation Matrix')
                fig_corr = px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r')
                fig_corr.update_layout(height=520)
                st.plotly_chart(fig_corr, use_container_width=True)

                # Average pairwise correlation metric
                if len(sel_assets) > 1:
                    tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    avg_pair = tri.stack().mean()
                    st.metric(label='Average pairwise correlation', value=f"{avg_pair:.2f}")

                # Rolling correlation for two user-selected stocks
                if len(sel_assets) >= 2:
                    st.markdown("**Rolling correlation between two selected stocks**")
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        stock1 = st.selectbox('Stock 1', options=sel_assets, index=0)
                    with sc2:
                        # default to second asset if available
                        default_idx = 1 if len(sel_assets) > 1 else 0
                        stock2 = st.selectbox('Stock 2', options=sel_assets, index=default_idx)

                    if stock1 == stock2:
                        st.warning('Select two different assets to view a meaningful rolling correlation.')
                    else:
                        rc_series = returns_sel[stock1].rolling(window=rolling_window_corr).corr(returns_sel[stock2])
                        fig_pair = go.Figure()
                        fig_pair.add_trace(go.Scatter(x=rc_series.index, y=rc_series, mode='lines', name=f'{stock1}/{stock2}'))
                        fig_pair.update_layout(title=f'Rolling correlation: {stock1} vs {stock2} ({rolling_window_corr}-day)', yaxis_title='Correlation', height=360)
                        st.plotly_chart(fig_pair, use_container_width=True)

                # Covariance matrix (annualized) in an expander
                cov_annual = returns_sel.cov() * 252.0
                with st.expander('Covariance matrix (annualized) — expand to view / download', expanded=False):
                    st.write('Annualized covariance of daily returns (decimal units).')
                    st.dataframe(cov_annual.style.format('{:.6f}'))
                    csv = cov_annual.to_csv().encode('utf-8')
                    st.download_button('Download covariance CSV', data=csv, file_name='covariance_annual.csv', mime='text/csv')

                # Allow download of correlation matrix as CSV
                csv_corr = corr.to_csv().encode('utf-8')
                st.download_button('Download correlation CSV', data=csv_corr, file_name='correlation.csv', mime='text/csv')

    with tabs[4]:
        st.header("Portfolio Construction")
        st.info("Build equal-weight, GMV (global minimum variance), and tangency portfolios; backtest with optional rebalancing.")

        prices = st.session_state.get('prices', None)
        if prices is None:
            st.warning("No price data loaded. Go to 'Inputs & Data' and click Load data.")
        else:
            returns = prices.pct_change().dropna(how='all')
            # Benchmark is comparison-only and must not be selectable as a portfolio asset.
            assets = [c for c in returns.columns if c != '^GSPC']

            # UI controls
            cols = st.columns([2, 2, 2])
            with cols[0]:
                sel_assets = st.multiselect('Assets for portfolio', options=assets, default=assets)
            with cols[1]:
                allow_short = st.checkbox('Allow short positions', value=False)
            # no rebalancing UI — backtests are buy-and-hold per project spec

            if not sel_assets:
                st.warning('Select at least one asset to build portfolios.')
            else:
                # Custom portfolio UI inside an expander (collapsible)
                with st.expander('Custom portfolio weights (expand to edit)', expanded=False):
                    st.write('Enter custom weights for each asset. We will normalize on Apply if needed.')
                    # initialize stored custom weights if missing or assets changed
                    if 'custom_weights' not in st.session_state:
                        st.session_state['custom_weights'] = {a: float(1.0 / max(1, len(sel_assets))) for a in sel_assets}
                        st.session_state['custom_applied'] = False
                    else:
                        # if asset list changed, try to preserve matching weights, else reset
                        stored = st.session_state.get('custom_weights', {})
                        if set(stored.keys()) != set(sel_assets):
                            new = {a: float(stored.get(a, 0.0)) for a in sel_assets}
                            # if no positive sum, default to equal weights
                            if abs(sum(new.values())) < 1e-12:
                                new = {a: float(1.0 / max(1, len(sel_assets))) for a in sel_assets}
                            st.session_state['custom_weights'] = new

                    cols_w = st.columns(3)
                    # present numeric inputs compactly across columns
                    for i, a in enumerate(sel_assets):
                        col = cols_w[i % 3]
                        key = f'cw_{a}'
                        default_val = float(st.session_state['custom_weights'].get(a, 0.0))
                        min_val = -10.0 if st.session_state.get('allow_short', allow_short) else 0.0
                        # use number_input for precise entry
                        val = col.number_input(label=a, value=default_val, key=key, format="%.6f", step=0.01, min_value=min_val)
                        st.session_state['custom_weights'][a] = float(val)

                    # Controls: Normalize, Apply, Reset
                    cw_cols = st.columns([1, 1, 1, 2])
                    with cw_cols[0]:
                        if st.button('Normalize'):
                            vals = np.array([st.session_state['custom_weights'][a] for a in sel_assets], dtype=float)
                            s = vals.sum()
                            if abs(s) < 1e-12:
                                st.error('Cannot normalize: sum of weights is zero.')
                            else:
                                vals = vals / s
                                for ai, a in enumerate(sel_assets):
                                    st.session_state['custom_weights'][a] = float(vals[ai])
                                st.success('Weights normalized to sum = 1')
                    with cw_cols[1]:
                        if st.button('Apply custom'):
                            vals = np.array([st.session_state['custom_weights'][a] for a in sel_assets], dtype=float)
                            s = vals.sum()
                            if not allow_short and (vals < 0).any():
                                st.error('Negative weights not allowed when shorting is disabled.')
                            elif abs(s) < 1e-12:
                                st.error('Cannot apply: sum of weights is zero. Normalize or enter valid weights.')
                            else:
                                # normalize to sum=1 for user convenience
                                vals = vals / s
                                for ai, a in enumerate(sel_assets):
                                    st.session_state['custom_weights'][a] = float(vals[ai])
                                st.session_state['custom_applied'] = True
                                st.success('Custom portfolio applied')
                    with cw_cols[2]:
                        if st.button('Reset'):
                            st.session_state['custom_weights'] = {a: float(1.0 / max(1, len(sel_assets))) for a in sel_assets}
                            st.session_state['custom_applied'] = False
                            # try to rerun to refresh number_inputs; if not available, show info
                            if hasattr(st, 'experimental_rerun'):
                                try:
                                    st.experimental_rerun()
                                except Exception:
                                    st.info('Reset applied; please collapse and reopen the expander to refresh inputs.')
                            else:
                                st.info('Reset applied; please collapse and reopen the expander to refresh inputs.')

                    # live preview of custom portfolio metrics
                    preview = np.array([st.session_state['custom_weights'][a] for a in sel_assets], dtype=float)
                    if abs(preview.sum()) > 1e-12:
                        preview = preview / preview.sum()
                        port_daily_preview = returns[sel_assets].dropna().dot(preview)
                        mean_d = port_daily_preview.mean()
                        std_d = port_daily_preview.std()
                        ann_ret_p = mean_d * 252.0
                        ann_vol_p = std_d * np.sqrt(252.0)
                        rf_local = rf_rate / 100.0
                        sharpe_p = (ann_ret_p - rf_local) / ann_vol_p if ann_vol_p > 0 else np.nan
                        downside = port_daily_preview[port_daily_preview < 0]
                        downside_std = downside.std()
                        downside_ann = downside_std * np.sqrt(252.0) if not np.isnan(downside_std) else np.nan
                        sortino_p = (ann_ret_p - rf_local) / downside_ann if downside_ann and downside_ann != 0 else np.nan
                        wealth_p = (1 + port_daily_preview).cumprod()
                        mdd_p = float((wealth_p / wealth_p.cummax() - 1).min()) * 100.0
                        st.write('Preview metrics (normalized)')
                        st.write({'Ann. Return (%)': f"{ann_ret_p*100:.2f}%", 'Ann. Vol (%)': f"{ann_vol_p*100:.2f}%", 'Sharpe': f"{sharpe_p:.2f}", 'Sortino (ann)': f"{sortino_p:.2f}", 'Max Drawdown (%)': f"{mdd_p:.2f}%"})

                # helper utilities
                def equal_weights(n):
                    w = np.ones(n) / float(n)
                    return w

                def gmv_weights(cov, long_only=True):
                    n = cov.shape[0]
                    x0 = np.ones(n) / n
                    bounds = None
                    if long_only:
                        bounds = tuple((0.0, 1.0) for _ in range(n))

                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

                    def obj(w):
                        return float(w.T @ cov @ w)

                    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
                    if not res.success:
                        # fallback to equal weights
                        return x0
                    return res.x

                def tangency_weights(mu, cov, rf, long_only=True):
                    # mu: annual returns (decimal), rf: annual decimal
                    n = len(mu)
                    x0 = np.ones(n) / n
                    bounds = None
                    if long_only:
                        bounds = tuple((0.0, 1.0) for _ in range(n))
                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

                    def neg_sharpe(w):
                        port_ret = w @ mu
                        port_vol = np.sqrt(max(1e-12, float(w.T @ cov @ w)))
                        return - (port_ret - rf) / port_vol

                    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
                    if not res.success:
                        return x0
                    return res.x

                # Prepare inputs: sample mean (annual), cov (annual)
                returns_sel = returns[sel_assets].dropna(how='all')
                mu_daily = returns_sel.mean()
                mu_ann = mu_daily * 252.0
                cov_daily = returns_sel.cov()
                cov_ann = cov_daily * 252.0

                rf_ann = rf_rate / 100.0

                # compute requested portfolios
                st.subheader('Compute portfolios')
                cols2 = st.columns([1, 1, 1])
                with cols2[0]:
                    do_equal = st.checkbox('Equal-weight', value=True)
                with cols2[1]:
                    do_gmv = st.checkbox('Global Min Variance (GMV)', value=True)
                with cols2[2]:
                    do_tan = st.checkbox('Tangency (Max Sharpe)', value=True)

                results = {}
                if do_equal:
                    w = equal_weights(len(sel_assets))
                    results['Equal'] = w
                if do_gmv:
                    w = gmv_weights(cov_ann.values, long_only=(not allow_short))
                    results['GMV'] = w
                if do_tan:
                    w = tangency_weights(mu_ann.values, cov_ann.values, rf_ann, long_only=(not allow_short))
                    results['Tangency'] = w

                # Integrate custom portfolio if applied
                if st.session_state.get('custom_applied', False):
                    try:
                        cw = np.array([st.session_state['custom_weights'][a] for a in sel_assets], dtype=float)
                        if abs(cw.sum()) < 1e-12:
                            # fallback to equal weights
                            cw = equal_weights(len(sel_assets))
                        else:
                            cw = cw / cw.sum()
                        results['Custom'] = cw
                    except Exception:
                        # if custom weights invalid, skip
                        pass

                # Backtest simulator using prices (exact shares to simulate drift + rebalancing)
                def backtest_portfolio(weights, prices_df, freq='None', start_val=10000.0):
                    p = prices_df[sel_assets].dropna()
                    dates = p.index
                    first_prices = p.iloc[0].values
                    shares = (start_val * weights) / first_prices
                    vals = []
                    current_shares = shares.copy()

                    for dt in dates:
                        prices_today = p.loc[dt].values
                        total_val = float((current_shares * prices_today).sum())
                        vals.append(total_val)

                    wealth = pd.Series(data=np.array(vals), index=dates)
                    returns_port = wealth.pct_change().fillna(0.0)
                    return wealth, returns_port

                # Display weights and metrics combined
                st.subheader('Portfolio Weights & Metrics')

                # compute basic metrics for each portfolio so we can combine with weights
                metrics_rows = []
                for name, w in results.items():
                    # normalized weights (in case solver returned slightly off-sum)
                    w = np.array(w)
                    if np.sum(np.abs(w)) == 0:
                        w = equal_weights(len(w))
                    w = w / np.sum(w)

                    # portfolio daily returns (from asset returns)
                    port_daily = returns_sel.dot(w)
                    mean_d = port_daily.mean()
                    std_d = port_daily.std()
                    ann_ret = mean_d * 252.0
                    ann_vol = std_d * np.sqrt(252.0)
                    sharpe = (ann_ret - rf_ann) / ann_vol if ann_vol > 0 else np.nan

                    # Sortino: use downside deviation
                    downside = port_daily[port_daily < 0]
                    downside_std = downside.std()
                    downside_ann = downside_std * np.sqrt(252.0) if not np.isnan(downside_std) else np.nan
                    sortino = (ann_ret - rf_ann) / downside_ann if downside_ann and downside_ann != 0 else np.nan

                    # Max drawdown
                    wealth_p = (1 + port_daily).cumprod()
                    dd = wealth_p / wealth_p.cummax() - 1
                    mdd = float(dd.min()) * 100.0

                    metrics_rows.append((name, ann_ret * 100.0, ann_vol * 100.0, sharpe, sortino, mdd))

                metrics_df = pd.DataFrame(metrics_rows, columns=['Portfolio', 'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe', 'Sortino (ann)', 'Max Drawdown (%)'])
                metrics_df.set_index('Portfolio', inplace=True)
                metrics_t = metrics_df.T

                # weights table (assets x portfolios)
                w_tbl = pd.DataFrame({k: v for k, v in results.items()}, index=sel_assets)

                # combined table: assets (weights) followed by metric rows
                combined = pd.concat([w_tbl, metrics_t], axis=0)

                # Build a display-ready DataFrame with formatted strings
                # use object dtype to allow formatted string assignment on numeric frames
                display_df = combined.copy().astype(object)
                for idx in display_df.index:
                    if idx in sel_assets:
                        # weight rows: 4 decimal places
                        display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.4f}")
                    elif idx in ['Ann. Return (%)', 'Ann. Vol (%)', 'Max Drawdown (%)']:
                        display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.2f}%")
                    elif idx in ['Sharpe', 'Sortino (ann)']:
                        display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.2f}")
                    else:
                        display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{x}")

                st.dataframe(display_df)
                # allow download of combined CSV
                st.download_button('Download combined weights+metrics CSV', data=combined.to_csv().encode('utf-8'), file_name='portfolio_weights_metrics.csv', mime='text/csv')

                # Collapsible pie charts for portfolio weights
                with st.expander('Portfolio Weights (pie charts by portfolio)', expanded=False):
                    st.write('Visual breakdown of asset weights for each computed portfolio.')
                    for pname, pw in results.items():
                        try:
                            w_arr = np.array(pw, dtype=float)
                            df_w = pd.DataFrame({'asset': sel_assets, 'weight': w_arr})
                            # If any negative weights (shorts) present, show signed bar chart instead
                            if (w_arr < 0).any():
                                st.warning(f"Detected negative weights in {pname}; showing signed bar chart instead of pie.")
                                fig_b = px.bar(df_w, x='asset', y='weight', title=f"{pname} Weights (signed)", text='weight')
                                fig_b.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                                fig_b.update_layout(yaxis_title='Weight', xaxis_title='Asset', height=400)
                                st.plotly_chart(fig_b, use_container_width=True)
                            else:
                                # If all weights zero or NaN, skip
                                if np.isclose(w_arr.sum(), 0) or np.allclose(w_arr, 0):
                                    st.info(f"{pname}: weights are all zero or undefined; skipping chart.")
                                    continue
                                fig_p = px.pie(df_w, names='asset', values='weight', title=f"{pname} Weights", hole=0.3)
                                fig_p.update_traces(textposition='inside', textinfo='percent+label')
                                fig_p.update_layout(height=420)
                                st.plotly_chart(fig_p, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not render chart for {pname}: {e}")

                # PRC (Percent Risk Contribution) bar charts
                with st.expander('PRC (Percent Risk Contribution) — bar charts by portfolio', expanded=False):
                    st.markdown("""
                    **Risk Contribution Explanation**

                    Percentage Risk Contribution (PRC) shows how much each asset contributes to the total risk (volatility) of the portfolio, rather than just how much of the portfolio it represents. While portfolio weights tell you how capital is allocated, PRC reveals how risk is distributed.

                    An asset’s risk contribution depends not only on its weight, but also on its volatility and how it moves relative to other assets (its correlations). Because of this, an asset with a relatively small weight can still contribute a large portion of total portfolio risk if it is highly volatile or strongly correlated with other risky assets.

                    For example, a stock with a 10% portfolio weight but a 25% risk contribution is a disproportionate source of portfolio volatility. Conversely, an asset may have a higher weight but contribute less risk if it is more stable or provides diversification benefits.

                    The PRC values across all assets sum to 100%, making it easy to compare how risk is distributed within the portfolio. This helps investors identify which positions are driving overall portfolio risk and make more informed diversification decisions.
                    """)
                    for pname, pw in results.items():
                        try:
                            w = np.array(pw, dtype=float)
                            # ensure covariance matrix aligns with sel_assets
                            cov = cov_ann
                            if isinstance(cov, pd.DataFrame):
                                cov = cov.loc[sel_assets, sel_assets].values
                            else:
                                cov = np.array(cov)

                            port_var = float(w.T @ cov @ w)
                            if port_var == 0 or np.isnan(port_var):
                                st.info(f"{pname}: portfolio variance is zero or undefined; skipping PRC chart.")
                                continue

                            contrib = (w * (cov @ w)) / port_var * 100.0
                            df_prc = pd.DataFrame({'asset': sel_assets, 'prc': contrib})

                            # If signed contributions present, show signed bar chart
                            if (contrib < 0).any():
                                st.warning(f"Detected negative contributions in {pname}; showing signed PRC bar chart.")
                                fig_prc = px.bar(df_prc, x='asset', y='prc', title=f"{pname} %Risk Contribution (signed)", text='prc')
                                fig_prc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                            else:
                                fig_prc = px.bar(df_prc.sort_values('prc', ascending=False), x='asset', y='prc', title=f"{pname} %Risk Contribution", text='prc')
                                fig_prc.update_traces(texttemplate='%{text:.2f}%', textposition='outside')

                            fig_prc.update_layout(yaxis_title='% Contribution', xaxis_title='Asset', height=420)
                            st.plotly_chart(fig_prc, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not render PRC chart for {pname}: {e}")

                # Efficient frontier plot (not collapsible)
                try:
                    if len(sel_assets) >= 2:
                        mu = mu_ann.values
                        cov = cov_ann.values if not isinstance(cov_ann, pd.DataFrame) else cov_ann.loc[sel_assets, sel_assets].values
                        n = len(mu)
                        # bounds depending on shorting
                        if allow_short:
                            bounds_ef = tuple((None, None) for _ in range(n))
                        else:
                            bounds_ef = tuple((0.0, 1.0) for _ in range(n))

                        cons_sum = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

                        # target returns grid between min and max of asset expected returns
                        r_min = float(np.min(mu))
                        r_max = float(np.max(mu))
                        if np.isfinite(r_min) and np.isfinite(r_max) and r_max > r_min:
                            targets = np.linspace(r_min, r_max, 50)
                        else:
                            targets = np.array([])

                        ef_rets = []
                        ef_vols = []

                        for targ in targets:
                            cons = (
                                {'type': 'eq', 'fun': lambda w, targ=targ: float(w @ mu) - targ},
                                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
                            )
                            x0 = np.ones(n) / n
                            def obj_var(w):
                                return float(w.T @ cov @ w)
                            try:
                                res = minimize(obj_var, x0, method='SLSQP', bounds=bounds_ef, constraints=cons)
                                if res.success:
                                    w = res.x
                                    vol = np.sqrt(max(0.0, float(w.T @ cov @ w)))
                                    ef_rets.append(targ)
                                    ef_vols.append(vol)
                            except Exception:
                                continue

                        if ef_rets and ef_vols:
                            ef_rets = np.array(ef_rets)
                            ef_vols = np.array(ef_vols)

                            fig_ef = go.Figure()
                            fig_ef.add_trace(go.Scatter(x=ef_vols * 100.0, y=ef_rets * 100.0, mode='lines', name='Efficient frontier', line=dict(color='royalblue')))

                            # prepare plotting list (include live custom preview if present)
                            plotting_items = list(results.items())
                            # live custom preview: show normalized custom_weights from session_state
                            preview_name = None
                            preview_w = None
                            try:
                                if 'custom_weights' in st.session_state:
                                    cw = np.array([st.session_state['custom_weights'].get(a, 0.0) for a in sel_assets], dtype=float)
                                    if abs(cw.sum()) > 1e-12:
                                        cw = cw / cw.sum()
                                        preview_name = 'Custom (preview)'
                                        preview_w = cw
                            except Exception:
                                preview_w = None
                            if preview_w is not None:
                                plotting_items.append((preview_name, preview_w))

                            # plot computed portfolios (Equal, GMV, Tangency, Custom, plus preview)
                            for pname, pw in plotting_items:
                                w = np.array(pw, dtype=float)
                                if np.sum(np.abs(w)) == 0:
                                    continue
                                w = w / np.sum(w)
                                pret = float(w @ mu)
                                pvol = float(np.sqrt(max(0.0, float(w.T @ cov @ w))))
                                marker = dict(size=10)
                                if pname.lower().startswith('tang'):
                                    marker.update(color='red', symbol='star')
                                elif pname.lower().startswith('gmv'):
                                    marker.update(color='green', symbol='diamond')
                                elif pname.lower().startswith('equal'):
                                    marker.update(color='orange', symbol='circle')
                                else:
                                    marker.update(color='purple', symbol='square')
                                fig_ef.add_trace(go.Scatter(x=[pvol * 100.0], y=[pret * 100.0], mode='markers+text', name=pname, marker=marker, text=[pname], textposition='top center'))

                            # Capital Allocation Line (CAL) from rf through tangency if available or computable
                            try:
                                tang_w = None
                                if 'Tangency' in results:
                                    tang_w = np.array(results['Tangency'], dtype=float)
                                else:
                                    # attempt to compute tangency portfolio for plotting the CML
                                    try:
                                        tang_w = tangency_weights(mu, cov, rf_ann, long_only=(not allow_short))
                                    except Exception:
                                        tang_w = None

                                if tang_w is not None and np.sum(np.abs(tang_w)) > 0:
                                    tang_w = tang_w / np.sum(tang_w)
                                    ret_t = float(tang_w @ mu)
                                    vol_t = float(np.sqrt(max(0.0, float(tang_w.T @ cov @ tang_w))))
                                    if vol_t > 1e-12:
                                        sharpe_t = (ret_t - rf_ann) / vol_t
                                        # extend line slightly beyond frontier for visibility
                                        x_max = max(ef_vols) * 100.0 * 1.2
                                        x_line = np.linspace(0.0, x_max, 200)
                                        y_line = (rf_ann + sharpe_t * (x_line / 100.0)) * 100.0
                                        fig_ef.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='CAL (rf)', line=dict(color='black', dash='dash')))
                                        # mark the risk-free point on the y-axis (vol=0)
                                        fig_ef.add_trace(go.Scatter(x=[0.0], y=[rf_ann * 100.0], mode='markers+text', name='Risk-free', marker=dict(size=8, color='black'), text=['RF'], textposition='bottom right'))
                            except Exception:
                                # non-fatal: continue without CML
                                pass

                            fig_ef.update_layout(title='Efficient Frontier (Annualized)', xaxis_title='Ann. Vol (%)', yaxis_title='Ann. Return (%)', height=480)
                            st.plotly_chart(fig_ef, use_container_width=True)
                except Exception as e:
                    st.warning(f"Efficient frontier plot unavailable: {e}")

                # Explanation expander for the efficient frontier
                with st.expander('What is the efficient frontier?', expanded=False):
                    st.markdown("""
                    **Efficient Frontier & Capital Allocation Line Explanation**

                    The efficient frontier represents the set of optimal portfolios that offer the highest expected return for a given level of risk (standard deviation), or equivalently, the lowest risk for a given return. Portfolios that lie on the frontier are considered “efficient,” while those below it are suboptimal because they deliver less return for the same level of risk.

                    Each point on the chart corresponds to a different portfolio. The individual stocks typically appear off the frontier, while diversified portfolios—such as the Global Minimum Variance (GMV) and tangency portfolios—lie on it. The GMV portfolio is the point with the lowest possible risk, while the tangency portfolio is the point that maximizes the Sharpe ratio (risk-adjusted return).

                    The Capital Allocation Line (CAL) shows the best possible trade-off between risk and return when combining a risk-free asset with a risky portfolio. It starts at the risk-free rate and is tangent to the efficient frontier at the tangency portfolio. Portfolios along the CAL represent different allocations between the risk-free asset and the tangency portfolio, offering the highest expected return for each level of risk.

                    Together, the efficient frontier and CAL help investors understand how to construct portfolios that optimize risk and return, and how diversification improves investment efficiency.
                    """)

                # Backtest and plot wealth series overlay
                st.subheader('Backtest Wealth (Start = $10,000)')
                bt_cols = st.columns([3, 1])
                with bt_cols[1]:
                    run_backtest = st.button('Run backtest')
                if run_backtest:
                    fig = go.Figure()
                    # Backtest plots will show portfolios and the S&P 500 benchmark only

                    # plot S&P 500 benchmark if available
                    try:
                        if '^GSPC' in prices.columns:
                            bench_ret = prices['^GSPC'].pct_change().dropna()
                            bench_wealth = (1 + bench_ret).cumprod() * 10000.0
                            fig.add_trace(go.Scatter(x=bench_wealth.index, y=bench_wealth.values, mode='lines', name='^GSPC (S&P 500)', line=dict(color='black', width=3, dash='dash')))
                    except Exception:
                        # non-fatal: skip benchmark plotting
                        pass

                    # plot portfolio backtests
                    for name, w in results.items():
                        try:
                            w = np.array(w)
                            w = w / np.sum(w)
                            wealth_series, _ = backtest_portfolio(w, prices)
                            fig.add_trace(go.Scatter(x=wealth_series.index, y=wealth_series.values, mode='lines', name=name))
                        except Exception:
                            continue

                    fig.update_layout(yaxis_title='Wealth ($)', xaxis_title='Date', height=480)
                    st.plotly_chart(fig, use_container_width=True)

                # Monte Carlo and rebalancing removed per user request

    with tabs[5]:
        st.header("Estimation Window Sensitivity")
        st.info("Compare optimized portfolios across different lookback windows")

        prices = st.session_state.get('prices', None)
        if prices is None:
            st.warning("No price data loaded. Go to 'Inputs & Data' and click Load data.")
        else:
            # Asset universe (exclude benchmark)
            assets = [c for c in prices.columns if c != '^GSPC']
            if not assets:
                st.warning('No assets available for sensitivity analysis.')
            else:
                # Controls: select assets, allow short, select lookback windows (dropdown-style multiselect)
                cols = st.columns([2, 1])
                with cols[0]:
                    sel_assets = st.multiselect('Assets for sensitivity', options=assets, default=assets)
                with cols[1]:
                    allow_short_sens = st.checkbox('Allow short positions', value=False, key='allow_short_sens')

                st.markdown('**Select lookback windows to compare (dropdown-style)**')
                # present as multiselect (dropdown UI) so users can pick multiple windows to compare
                lookback_opts = ['1 Year', '3 Year', '5 Year', 'Full Sample']
                selected_windows = st.multiselect('Lookback windows', options=lookback_opts, default=['1 Year', '3 Year', 'Full Sample'])

                # Which portfolios to compute
                pcols = st.columns(3)
                with pcols[0]:
                    do_equal_s = st.checkbox('Equal-weight', value=True, key='sens_eq')
                with pcols[1]:
                    do_gmv_s = st.checkbox('GMV', value=True, key='sens_gmv')
                with pcols[2]:
                    do_tan_s = st.checkbox('Tangency', value=True, key='sens_tan')

                # local helper optimizers (same approach as in Portfolio tab)
                def equal_weights_local(n):
                    return np.ones(n) / float(n)

                def gmv_weights_local(cov, long_only=True):
                    n = cov.shape[0]
                    x0 = np.ones(n) / n
                    bounds = None
                    if long_only:
                        bounds = tuple((0.0, 1.0) for _ in range(n))
                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
                    def obj(w):
                        return float(w.T @ cov @ w)
                    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
                    if not res.success:
                        return x0
                    return res.x

                def tangency_weights_local(mu, cov, rf, long_only=True):
                    n = len(mu)
                    x0 = np.ones(n) / n
                    bounds = None
                    if long_only:
                        bounds = tuple((0.0, 1.0) for _ in range(n))
                    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
                    def neg_sharpe(w):
                        port_ret = w @ mu
                        port_vol = np.sqrt(max(1e-12, float(w.T @ cov @ w)))
                        return - (port_ret - rf) / port_vol
                    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
                    if not res.success:
                        return x0
                    return res.x

                # Prepare results containers
                all_columns = []
                all_results_weights = {}
                all_results_metrics = {}

                for win in selected_windows:
                    # determine start date for window
                    if win == 'Full Sample':
                        p_sub = prices[sel_assets].dropna(how='all')
                    else:
                        days = 365 if win == '1 Year' else (365 * 3 if win == '3 Year' else 365 * 5)
                        # ensure we compare like dtypes: convert sidebar `end` (date) to Timestamp
                        end_ts = pd.to_datetime(end)
                        window_start = end_ts - timedelta(days=days)
                        p_sub = prices.loc[prices.index >= window_start, sel_assets].dropna(how='all')

                    if p_sub.empty or p_sub.shape[0] < 10:
                        st.warning(f"Window '{win}' has insufficient data; skipping.")
                        continue

                    rets = p_sub.pct_change().dropna(how='all')
                    mu_ann = rets.mean() * 252.0
                    cov_ann = rets.cov() * 252.0

                    window_col_names = []
                    # compute requested portfolios
                    if do_equal_s:
                        w_eq = equal_weights_local(len(sel_assets))
                        col_name = f"Equal ({win})"
                        all_results_weights[col_name] = w_eq
                        window_col_names.append(col_name)
                    if do_gmv_s:
                        try:
                            w_gmv = gmv_weights_local(cov_ann.values, long_only=(not allow_short_sens))
                        except Exception:
                            w_gmv = equal_weights_local(len(sel_assets))
                        col_name = f"GMV ({win})"
                        all_results_weights[col_name] = w_gmv
                        window_col_names.append(col_name)
                    if do_tan_s:
                        try:
                            w_tan = tangency_weights_local(mu_ann.values, cov_ann.values, rf_rate / 100.0, long_only=(not allow_short_sens))
                        except Exception:
                            w_tan = equal_weights_local(len(sel_assets))
                        col_name = f"Tangency ({win})"
                        all_results_weights[col_name] = w_tan
                        window_col_names.append(col_name)

                    # compute metrics for each computed portfolio in this window
                    for cname in window_col_names:
                        w = np.array(all_results_weights[cname], dtype=float)
                        if np.sum(np.abs(w)) == 0:
                            w = equal_weights_local(len(sel_assets))
                        w = w / np.sum(w)
                        port_daily = rets.dot(w)
                        mean_d = port_daily.mean()
                        std_d = port_daily.std()
                        ann_ret = mean_d * 252.0
                        ann_vol = std_d * np.sqrt(252.0)
                        sharpe = (ann_ret - (rf_rate / 100.0)) / ann_vol if ann_vol > 0 else np.nan
                        downside = port_daily[port_daily < 0]
                        downside_std = downside.std()
                        downside_ann = downside_std * np.sqrt(252.0) if not np.isnan(downside_std) else np.nan
                        sortino = (ann_ret - (rf_rate / 100.0)) / downside_ann if downside_ann and downside_ann != 0 else np.nan
                        wealth_p = (1 + port_daily).cumprod()
                        mdd = float((wealth_p / wealth_p.cummax() - 1).min()) * 100.0
                        # store metrics (Ann. Return and Ann. Vol in percent to match earlier tab)
                        all_results_metrics[cname] = {
                            'Ann. Return (%)': ann_ret * 100.0,
                            'Ann. Vol (%)': ann_vol * 100.0,
                            'Sharpe': sharpe,
                            'Sortino (ann)': sortino,
                            'Max Drawdown (%)': mdd
                        }

                if not all_results_weights:
                    st.info('No valid windows computed; adjust selection or date range.')
                else:
                    # Build combined table: weights (rows=assets) then metrics rows per computed column
                    w_tbl = pd.DataFrame({k: v for k, v in all_results_weights.items()}, index=sel_assets)
                    # metrics as DataFrame where columns match w_tbl columns
                    metrics_df = pd.DataFrame({k: v for k, v in all_results_metrics.items()}).T

                    # assemble combined table similar to Portfolio tab: weights rows followed by metric rows
                    combined = pd.concat([w_tbl, metrics_df.T], axis=0)

                    # Format for display
                    # use object dtype to allow formatted string assignment on numeric frames
                    display_df = combined.copy().astype(object)
                    for idx in display_df.index:
                        if idx in sel_assets:
                            display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.4f}")
                        elif idx in ['Ann. Return (%)', 'Ann. Vol (%)', 'Max Drawdown (%)']:
                            display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.2f}%")
                        elif idx in ['Sharpe', 'Sortino (ann)']:
                            display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{float(x):.2f}")
                        else:
                            display_df.loc[idx] = combined.loc[idx].apply(lambda x: f"{x}")

                    st.subheader('Sensitivity: Weights & Metrics across windows')
                    st.dataframe(display_df)
                    st.download_button('Download sensitivity CSV', data=combined.to_csv().encode('utf-8'), file_name='sensitivity_weights_metrics.csv', mime='text/csv')

                    # Explanation expander for estimation window sensitivity
                    with st.expander('Estimation Window Sensitivity Explanation', expanded=False):
                        st.markdown("""
                        Mean-variance optimization relies heavily on estimated inputs, particularly expected returns and the covariance matrix. These inputs are based on historical data, and even small changes in the selected time period (lookback window) can lead to significantly different portfolio weights and performance metrics.

                        This sensitivity occurs because asset returns and relationships are not stable over time—what appears optimal using one historical window may look very different using another. As a result, portfolios such as the GMV and tangency portfolio can shift meaningfully depending on the data used in the estimation.

                        By comparing results across multiple lookback periods, this section highlights how dependent optimized portfolios are on their underlying assumptions. It reinforces an important takeaway: historical optimization results should be interpreted with caution, as they may not be robust or stable out of sample.
                        """)

                    # Optional weight comparison chart in an expander
                    with st.expander('Weight comparison across windows (bar chart)', expanded=False):
                        try:
                            # normalize any near-zero columns for plotting clarity
                            plot_df = w_tbl.fillna(0.0)
                            fig_w = go.Figure()
                            for col in plot_df.columns:
                                fig_w.add_trace(go.Bar(name=col, x=plot_df.index, y=plot_df[col]))
                            fig_w.update_layout(barmode='group', xaxis_title='Asset', yaxis_title='Weight', height=480)
                            st.plotly_chart(fig_w, use_container_width=True)
                        except Exception as e:
                            st.error(f'Could not render weight comparison chart: {e}')

    with tabs[6]:
        st.header("About / Methodology")
        st.markdown("""
        **Assumptions & Methodology**

        This application applies mean–variance portfolio optimization to construct and evaluate portfolios using historical asset data. The results are driven by several key assumptions and modeling choices outlined below.

        1. Return and Risk Estimation
        Expected returns and the covariance matrix are estimated from historical price data using daily simple returns over a user-selected lookback window. These estimates are assumed to be representative of future performance. Portfolio risk is measured using standard deviation (volatility), and all return and risk metrics are annualized for comparability.

        2. Mean–Variance Framework
        Portfolio construction follows the mean–variance optimization framework, which assumes investors are risk-averse and seek to maximize expected return for a given level of risk. The efficient frontier represents the set of optimal portfolios under this framework, including the Global Minimum Variance (GMV) portfolio and the tangency (maximum Sharpe ratio) portfolio.

        3. Risk-Free Rate and Capital Allocation
        A user-specified risk-free rate is used to compute Sharpe ratios and construct the Capital Allocation Line (CAL). The model assumes investors can borrow and lend at this rate, which may not hold in real-world markets.

        4. Portfolio Constraints
        Portfolios are fully invested, meaning asset weights must sum to 1. The model includes a toggle for short selling, allowing users to permit or restrict negative weights. No additional weight constraints (such as upper or lower bounds on individual assets) are imposed.

        5. Diversification and Correlation
        Diversification benefits arise from imperfect correlations between assets. These correlations are estimated from historical data and assumed to remain stable over the investment horizon, though in practice they may change over time.

        6. Sensitivity to Estimation Window
        Mean-variance optimization is highly sensitive to its inputs, particularly expected returns and covariances. Because these inputs are estimated from historical data, even small changes in the selected lookback period can lead to significantly different portfolio weights and performance outcomes.

        This sensitivity reflects the fact that asset returns and relationships are not stable over time. By comparing results across multiple estimation windows, the application illustrates how dependent optimized portfolios are on historical assumptions. As a result, these portfolios should be interpreted with caution, as they may not be robust or stable out of sample.

        7. Limitations
        This analysis does not account for transaction costs, taxes, liquidity constraints, or market frictions. It also assumes continuous rebalancing without cost and does not incorporate forward-looking information. Extreme market events or structural changes may not be fully captured by historical data.
        """)


if __name__ == "__main__":
    main()
