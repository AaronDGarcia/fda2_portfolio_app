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
                    st.line_chart(prices.fillna(method="ffill"))

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
            assets = [c for c in returns.columns]

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
            assets = [c for c in returns.columns]

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
