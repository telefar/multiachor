# -*-coding:utf-8-*-
#__author__ = 'Palmer.Liu@outlook.com'
__author__ = '878492462@qq.com'
#source：https://mp.weixin.qq.com/s/nI3q6mX9_73whGNIp0Ia0g

#一种如何使用多锚点回归来发现方向、强度和汇聚点的方法
import numpy as np
import pandas as pd

import akshare as ak
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from typing import Tuple
import matplotlib.dates as mdates

import streamlit as st
import time
from datetime import datetime, timedelta

class MultiAnchorRegressionChannels:
    def __init__(self, ticker: str, start_date: str, end_date: str,
                 max_bars: int = 2000, dev_mult: float = 2.0,
                 use_log_scale: bool = False, use_exp_weight: bool = False):
        """
        多锚点回归通道分析

        参数:
            ticker: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_bars: 锚点搜索的最大回溯窗口
            dev_mult: 通道宽度的标准差乘数
            use_log_scale: 是否使用对数价格转换
            use_exp_weight: 是否使用指数权重
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.max_bars = max_bars
        self.dev_mult = dev_mult
        self.use_log_scale = use_log_scale
        self.use_exp_weight = use_exp_weight

        # 锚点配置
        self.anchors = ["BarHighest", "BarLowest", "SlopeZero"]
        self.channel_colors = ["green", "red", "cyan"]

        # 数据存储
        self.data = None
        self.close = None
        self.high = None
        self.low = None
        self.close_t = None

        # 结果存储
        self.channel_info = []

    def transform_price(self, p: pd.Series) -> pd.Series:
        """价格转换（对数转换）"""
        return np.log10(p) if self.use_log_scale else p

    def inverse_transform_price(self, p: np.ndarray) -> np.ndarray:
        """逆转换价格数据"""
        return np.power(10, p) if self.use_log_scale else p

    def exponential_weights(self, n: int, decay: float = 0.9) -> np.ndarray:
        """生成指数权重"""
        return decay ** np.arange(n)[::-1]

    def find_bar_highest(self, high_series: pd.Series) -> int:
        """找到最高点以来的柱数"""
        window = high_series.iloc[-self.max_bars:]
        idxmax = window.idxmax()
        return len(high_series) - high_series.index.get_loc(idxmax)

    def find_bar_lowest(self, low_series: pd.Series) -> int:
        """找到最低点以来的柱数"""
        window = low_series.iloc[-self.max_bars:]
        idxmin = window.idxmin()
        return len(low_series) - low_series.index.get_loc(idxmin)

    def find_slope_zero(self, series: pd.Series) -> int:
        """找到斜率最接近零的窗口长度"""
        best_len = 2
        best_slope = float("inf")

        # 搜索最佳窗口长度
        for L in range(2, min(self.max_bars, len(series))):
            y = series.iloc[-L:]
            x = np.arange(L).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, y.values)
            slope = abs(model.coef_[0])

            if slope < best_slope:
                best_slope = slope
                best_len = L

        return best_len

    def calc_regression_channel(self, series: pd.Series, length: int) -> Tuple:
        """计算回归通道和统计指标"""
        y = series.iloc[-length:]
        x = np.arange(length).reshape(-1, 1)
        model = LinearRegression()

        # 应用指数权重（如果启用）
        if self.use_exp_weight:
            weights = self.exponential_weights(length)
            weights /= weights.sum()
            model.fit(x, y.values, sample_weight=weights)
        else:
            model.fit(x, y.values)

        slope = model.coef_[0]
        intercept = model.intercept_

        # 计算回归线
        reg_line = intercept + slope * x.squeeze()

        # 计算残差和标准差
        residuals = y.values - reg_line
        stdev = np.std(residuals)

        # 计算相关性和R²
        r_val = np.corrcoef(np.arange(length), y.values)[0, 1]
        r2 = r_val ** 2

        # 计算通道边界
        upper_channel = reg_line + self.dev_mult * stdev
        lower_channel = reg_line - self.dev_mult * stdev

        return reg_line, upper_channel, lower_channel, slope, r2, r_val

    def fetch_data(self):
        """获取股票数据"""
        try:
            ticker = self.ticker  # 假设 ticker 是原始输入的代码，比如 '00700'
            start_date = self.start_date
            end_date = self.end_date
            # 判断是否为港股 (5位数字)
            if len(ticker) == 5 and ticker.isdigit():
                # 使用 akshare 获取港股数据
                df = ak.stock_hk_daily(symbol=ticker, adjust="qfq")
                # 转换日期格式并过滤指定区间
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df = df.loc[mask]
            else:
                # 处理 A 股
                df = ak.stock_zh_a_daily(symbol=self.ticker, start_date=start_date, end_date=end_date, adjust='qfq')
            if df.empty:
                raise ValueError("No data returned from akshare")
            df.dropna(inplace=True)

            df['date'] = pd.to_datetime(df['date'])  # 转换为 datetime 类型
            df.set_index('date', inplace=True)  # 设置为索引
            df.sort_index(inplace=True)  # 按日期排序

            self.data = df
            self.close = df["close"]
            self.high = df["high"]
            self.low = df["low"]
            self.close_t = self.transform_price(self.close)

        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def analyze(self):
        """执行多锚点回归分析"""
        if self.data is None:
            self.fetch_data()

        self.channel_info = []

        # 对每个锚点类型进行分析
        for name, color in zip(self.anchors, self.channel_colors):
            # 确定窗口长度
            if name == "BarHighest":
                length = self.find_bar_highest(self.high)
            elif name == "BarLowest":
                length = self.find_bar_lowest(self.low)
            elif name == "SlopeZero":
                length = self.find_slope_zero(self.close_t)

            # 确保长度在有效范围内
            length = max(2, min(length, len(self.close_t)))

            # 计算回归通道
            mid, top, bot, slope, r2, r_val = self.calc_regression_channel(self.close_t, length)

            # 逆转换价格
            mid_plot = self.inverse_transform_price(mid)
            top_plot = self.inverse_transform_price(top)
            bot_plot = self.inverse_transform_price(bot)

            # 确定趋势方向
            trend = "Up" if slope > 0.01 else "Down" if slope < -0.01 else "Flat"

            # 存储通道信息
            info = {
                "anchor": name,
                "length": length,
                "slope": slope,
                "r2": r2,
                "r": r_val,
                "trend": trend,
                "color": color,
                "xvals": self.close.index[-length:],
                "mid": mid_plot,
                "top": top_plot,
                "bot": bot_plot,
                "last_mid": mid_plot[-1],
                "last_top": top_plot[-1],
                "last_bot": bot_plot[-1]
            }
            self.channel_info.append(info)

    def plot_results(self):
        """可视化结果"""
        if not self.channel_info:
            self.analyze()

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_facecolor("#1f1b1b")
        #fig, ax = plt.subplots(figsize=(14, 8))
        #plt.style.use("dark_background")
        #ax.set_facecolor("#1f1b1b")
        ax.grid(True, alpha=0.2)

        # 绘制价格线
        ax.plot(self.close.index, self.close, color="white", linewidth=1.5, label="Close Price")

        # 绘制每个通道
        for info in self.channel_info:
            xvals = info["xvals"]
            ax.plot(xvals, info["mid"], color=info["color"], linewidth=2, label=f"{info['anchor']}")
            ax.plot(xvals, info["top"], color=info["color"], linestyle="--", alpha=0.7)
            ax.plot(xvals, info["bot"], color=info["color"], linestyle="--", alpha=0.7)
            ax.fill_between(xvals, info["bot"], info["top"], color=info["color"], alpha=0.1)

            # 标注统计信息
            ax.annotate(f"{info['anchor']}\nL={info['length']}\nR={info['r']:.2f}",
                        xy=(xvals[-1], info["last_mid"]),
                        xytext=(-100 if info["anchor"] == "BarHighest" else 0,
                                30 if info["anchor"] == "SlopeZero" else 0),
                        textcoords="offset points",
                        color=info["color"],
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5))

        # 检测汇聚点
        confluence_detected = False
        if len(self.channel_info) > 1:
            mids = [info["last_mid"] for info in self.channel_info]
            avg_mid = np.mean(mids)
            stdev_mids = np.std(mids)
            threshold = 0.5 * stdev_mids  # 汇聚阈值

            if np.max(mids) - np.min(mids) < threshold:
                confluence_detected = True
                x_conf = max([info["xvals"][-1] for info in self.channel_info])
                ax.annotate("CONFLUENCE",
                            xy=(x_conf, avg_mid),
                            xytext=(-100, -30),
                            textcoords="offset points",
                            color="gold",
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.7))

        # 检测当前价格状态
        current_price = self.close.iloc[-1]
        tops = [info["last_top"] for info in self.channel_info]
        bots = [info["last_bot"] for info in self.channel_info]

        if current_price > max(tops):
            current_condition = "Overextended Uptrend"
            condition_color = "red"
        elif current_price < min(bots):
            current_condition = "Oversold Downtrend"
            condition_color = "green"
        else:
            current_condition = "Within Channels"
            condition_color = "yellow"

        # 设置标题和标签
        title_text = f"Multi-Anchored Regression Channels: {self.ticker}"
        title_text += f" | {current_condition}"
        title_text += f" | Confluence: {'Detected' if confluence_detected else 'None'}"

        ax.set_title(title_text, color="orange", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加图例
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()

        # 打印报告
        self.print_report(current_condition, confluence_detected)

        self.current_condition = current_condition
        self.confluence_detected = confluence_detected

        #plt.show()
        return fig

    def print_report(self, current_condition: str, confluence_detected: bool):
        """打印分析报告"""
        print("\n" + "=" * 60)
        print(f"Multi-Anchored Regression Analysis Report: {self.ticker}")
        print("=" * 60)
        print(f"Analysis Period: {self.start_date} to {self.end_date}")
        print(f"Current Condition: {current_condition}")
        print(f"Confluence Detected: {'Yes' if confluence_detected else 'No'}")
        print("\nChannel Details:")
        print("-" * 60)
        print("Anchor      | Length | Slope     | R     | R²    | Trend")
        print("-" * 60)

        for info in self.channel_info:
            print(f"{info['anchor']:<12} | {info['length']:<6} | {info['slope']:+.6f} | "
                  f"{info['r']:+.4f} | {info['r2']:.4f} | {info['trend']}")

        print("=" * 60)


    def run(self):
        """执行完整分析流程"""
        try:
            self.fetch_data()
            self.analyze()
            self.plot_results()
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

# 补全市场前缀
def convert_symbol(code: str) -> str:
    code = code.strip()
    if not code.isdigit():
        raise ValueError("股票代码必须是数字")
    # A股代码处理
    if len(code) == 6:
        if code.startswith('6'):
            return f"sh{code}"
        elif code.startswith(('0', '3')):
            return f"sz{code}"
        else:
            raise ValueError(f"不支持的A股代码: {code}")
    # 港股代码处理 (5位)
    elif len(code) == 5:
        return code
    else:
        raise ValueError(f"无效的股票代码长度: {code}")
def display_report_on_st(current_condition: str, confluence_detected: bool, channel_info: list):
    """
    将分析报告展示在 Streamlit 页面上

    参数:
        current_condition: 当前价格状态（Overextended/Oversold/Within）
        confluence_detected: 是否检测到通道汇聚点
        channel_info: 通道信息列表（来自 analyzer.channel_info）
    """
    st.markdown("## 📊 分析报告")

    st.markdown(f"**股票代码**: {channel_info[0]['anchor']}")
    st.markdown(f"**分析时间段**: {channel_info[0]['xvals'][0].date()} 至 {channel_info[0]['xvals'][-1].date()}")
    st.markdown(f"**当前市场状态**: {current_condition}")
    st.markdown(f"**汇聚点检测**: {'✅ 检测到' if confluence_detected else '❌ 未检测到'}")

    st.markdown("### 🔍 通道详情")

    # 构建 DataFrame 展示各通道数据
    report_data = []
    for info in channel_info:
        report_data.append({
            "锚点类型": info["anchor"],
            "窗口长度 (L)": info["length"],
            "斜率 (Slope)": f"{info['slope']:.6f}",
            "相关系数 (R)": f"{info['r']:.4f}",
            "决定系数 (R²)": f"{info['r2']:.4f}",
            "趋势方向": info["trend"]
        })

    # 使用 DataFrame 展示表格
    import pandas as pd
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df.style.highlight_max(subset=["窗口长度 (L)"], color='lightgreen')
                 .highlight_min(subset=["窗口长度 (L)"], color='salmon'),
                 use_container_width=True)

    # 可视化各通道趋势图
    '''
    st.markdown("### 📈 通道趋势图")
    for info in channel_info:
        st.markdown(f"#### {info['anchor']} 通道 - 颜色: `{info['color']}`")
        st.line_chart(pd.DataFrame({
            '中轨': info['mid'],
            '上轨': info['top'],
            '下轨': info['bot']
        }, index=info['xvals']))
    '''
def display_channel_trend():
    ticker_input = st.text_input("请输入股票代码（例如 300124 或 600001）", value="300124", max_chars=6)
    # 新增：用户输入回溯天数，默认为 730 天（2年）
    lookback_days = st.number_input("回溯天数（用于设定开始日期）", min_value=50, max_value=3650, value=550, step=1)

    if st.button("开始分析"):
        try:
            full_ticker = convert_symbol(ticker_input)
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            # 创建分析器并运行
            analyzer = MultiAnchorRegressionChannels(
                ticker=full_ticker,
                start_date=start_date,
                end_date=end_date,
                max_bars=1000,
                dev_mult=2.0,
                use_log_scale=False,
                use_exp_weight=False
            )
            analyzer.run()
            fig = analyzer.plot_results()
            st.pyplot(fig)
            display_report_on_st(
                current_condition=analyzer.current_condition,
                confluence_detected=analyzer.confluence_detected,
                channel_info=analyzer.channel_info
            )
        except Exception as e:
            st.error(f"发生错误: {e}")

def get_codes_lists():
    from stockAnalyze.utils.readstockcodesfromtdx import read_zxg_stock_codes_for_call, read_gn_stock_codes
    from config_private import ZXG_FILE, TOP_20_FILE, TOP_200_FILE

    data_type_options = ['自选tick', '自定20tick', '概念板块', '自选与自定20tick', '自定200tick']
    selected_data_type = st.selectbox('请选择要选股的板块', data_type_options)
    gn_name_input = st.text_input(label='请输入通达信概念板块名', placeholder='例如：通达信88', value='水产品')
    querybutton = st.button('点击开始下载分析')
    if querybutton:
        if selected_data_type == '自选tick':
            stock_code_input = read_zxg_stock_codes_for_call(ZXG_FILE)

        elif selected_data_type == '自定20tick':
            stock_code_input = read_zxg_stock_codes_for_call(TOP_20_FILE)

        elif selected_data_type == '概念板块':
            if gn_name_input:
                stock_code_input = read_gn_stock_codes(gn_name_input)
                pass

        elif selected_data_type == '自选与自定20tick':
            stock_code_input = read_zxg_stock_codes_for_call(ZXG_FILE, TOP_20_FILE)

        elif selected_data_type == '自定200tick':
            stock_code_input = read_zxg_stock_codes_for_call(TOP_200_FILE)


        stock_codes = stock_code_input.split()
        print(stock_codes)

        return stock_codes
def generate_scan_report(stock_list=list, lookback_days=550, dev_mult=2.0):
    '''
    stock_list = get_codes_lists()
    if not stock_list:
        return None
    '''

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    for idx,ticker in enumerate(stock_list):
        if not(ticker.startswith(('6', '0', '3')) and ticker.isdigit()):
            continue
        try:
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            tickersym = convert_symbol(ticker)
            analyzer = MultiAnchorRegressionChannels(
                ticker=tickersym,
                start_date=start_date,
                end_date=end_date,
                max_bars=1000,
                dev_mult=dev_mult
            )
            analyzer.run()

            current_price = analyzer.close.iloc[-1]
            tops = [info["last_top"] for info in analyzer.channel_info]
            bots = [info["last_bot"] for info in analyzer.channel_info]

            if current_price > max(tops):
                signal = "Sell"
            elif current_price < min(bots):
                signal = "Buy"
            else:
                signal = "Neutral"

            # 判断是否所有通道趋势都为 Up
            all_up = all(info["trend"] == "Up" for info in analyzer.channel_info)
            if all_up:
                signal = "AllUp"

            current_condition = analyzer.current_condition
            confluence_detected = "exist" if analyzer.confluence_detected else "none"
            results.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "Signal": signal,
                "Current Condition": current_condition,
                "Confluence Detected": confluence_detected
            })

            # 更新进度条
            progress = (idx + 1) / len(stock_list)
            progress_bar.progress(progress)
            status_text.text(f"正在分析: {ticker} ({idx + 1}/{len(stock_list)})")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    status_text.text("分析完成！")
    time.sleep(0.5)  # 短暂延迟让 UI 更流畅
    progress_bar.empty()
    return pd.DataFrame(results)

def signal_statistics(df):
    if df is not None and not df.empty:
        # 统计 Buy 和 Sell 数量
        signal_counts = df['Signal'].value_counts()

        buy_count = signal_counts.get('Buy', 0)
        sell_count = signal_counts.get('Sell', 0)
        neutral_count = signal_counts.get('Neutral', 0)
        all_up_count = signal_counts.get('AllUp', 0)

        # 显示统计信息
        st.markdown("### 📊 信号统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="✅ Buy 信号数量", value=buy_count)
        col2.metric(label="❌ Sell 信号数量", value=sell_count)
        col3.metric(label="🟰 Neutral 信号数量", value=neutral_count)
        col4.metric(label="📈 AllUp 信号数量", value=all_up_count)

        # 信号筛选器
        if 'selected_signal' not in st.session_state:
            st.session_state.selected_signal = "All"

        option = st.selectbox(
            "选择要查看的信号类型",
            ["All", "Buy", "AllUp", "Sell", "Neutral"],
            index=["All", "Buy", "AllUp", "Sell", "Neutral"].index(st.session_state.selected_signal)
        )
        st.session_state.selected_signal = option

        if option != "All":
            filtered_df = df[df["Signal"] == option]
        else:
            filtered_df = df
        #st.dataframe(filtered_df)
        st.dataframe(filtered_df[['Ticker', 'Current Price', 'Signal', 'Current Condition', 'Confluence Detected']])
def display_page_on_channel_trend_analysis():
    tabanalysis, tabxg = st.tabs(["个股通道分析", "板块级通道批量选股"])

    with tabxg:
        if 'scan_df' not in st.session_state:
            st.session_state.scan_df = None
        if 'last_run_stock_list' not in st.session_state:
            st.session_state.last_run_stock_list = []

        stock_list = get_codes_lists()

        if stock_list:
            #with st.spinner("正在批量分析股票..."):
            df = generate_scan_report(stock_list=stock_list)
            st.session_state.scan_df = df
            st.session_state.last_run_stock_list = stock_list

        if st.session_state.scan_df is not None:
            signal_statistics(st.session_state.scan_df)

    with tabanalysis:
        display_channel_trend()

# 示例使用
if __name__ == "__main__":
    df_report = generate_scan_report(["sz301011", "sh600000", "sz000001", "sz300124"])
    print(df_report)

    TICKER = "sz301011"
    START_DATE = "2024-01-01"
    END_DATE = "2025-06-17"

    # 创建分析器
    analyzer = MultiAnchorRegressionChannels(
        ticker=TICKER,
        start_date=START_DATE,
        end_date=END_DATE,
        max_bars=1000,
        dev_mult=2.0,
        use_log_scale=False,
        use_exp_weight=False
    )

    # 执行分析
    analyzer.run()