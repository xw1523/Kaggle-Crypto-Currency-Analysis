import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import kagglehub
import os
import shutil
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import ccf
import scipy.stats as stats
# --- 1. 页面与样式设置 ---
st.set_page_config(layout="wide", page_title="加密货币与宏观经济分析仪表盘")

def set_app_style():
    """统一设置 Matplotlib/Seaborn 的图表样式，并支持中文。"""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "SimHei",
        "Microsoft YaHei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'figure.titlesize': 16,
        'axes.labelcolor': '#333333',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# 在应用开始时调用样式设置
set_app_style()

# --- 2. 核心数据加载与处理函数  ---

@st.cache_data(show_spinner="正在加载每小时加密货币数据...")
def load_hourly_crypto_data():
    """从 Kaggle Hub 下载并用 DuckDB 聚合为每小时数据。"""
    dataset_handle = "tencars/392-crypto-currency-pairs-at-minute-resolution"
    try:
        download_path = kagglehub.dataset_download(dataset_handle)
    except Exception as e:
        st.error(f"Kaggle 数据下载失败: {e}")
        st.error("请检查您的网络连接和 Kaggle API (kaggle.json) 配置。")
        return None

    target_dir = os.path.join('data', 'crypto')
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    source_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
    for filename in source_files:
        shutil.copy(os.path.join(download_path, filename), target_dir)

    query = """
    SELECT 
        date_trunc('hour', to_timestamp(time / 1000)) AS time_bucket, 
        REGEXP_REPLACE(filename, '.*/|\\.csv', '', 'g') AS coin_name,
        arg_min(open, to_timestamp(time / 1000)) AS open,
        max(high) AS high,
        min(low) AS low,
        arg_max(close, to_timestamp(time / 1000)) AS close,
        sum(volume) AS volume
    FROM 
        read_csv_auto('data/crypto/*.csv', header=true, filename=true, union_by_name=true)
    WHERE time > 0
    GROUP BY 1, 2 ORDER BY 1, 2;
    """
    try:
        hourly_df = duckdb.query(query).to_df()
        hourly_df["time_bucket"] = pd.to_datetime(hourly_df["time_bucket"])
        return hourly_df
    except Exception as e:
        st.error(f"DuckDB 查询失败: {e}")
        return None

@st.cache_data(show_spinner="正在下载宏观经济数据...")
def load_macro_data(start_date, end_date):
    """使用 yfinance 下载宏观经济数据。"""
    tickers = {
        "^GSPC": "SP500",
        "^IXIC": "NASDAQ",
        "^VIX": "VIX",
        "GC=F": "Gold",
        "DX-Y.NYB": "USD_Index",
    }
    eco_data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False)
    if eco_data.empty:
        st.warning("未能下载宏观经济数据，请检查日期范围或 Tickers。")
        return None
    eco_close_data = eco_data["Close"].rename(columns=tickers)
    # 使用前向填充和后向填充处理周末和节假日的数据缺失
    return eco_close_data.ffill().bfill()

@st.cache_data
def prepare_daily_merged_data(_hourly_crypto_df):
    """将加密数据聚合为每日，并与宏观数据合并。"""
    # 1. 聚合加密数据到每日级别
    daily_crypto_df = (_hourly_crypto_df.set_index("time_bucket")
        .groupby("coin_name")
        .resample("D")
        .agg(
            daily_close=("close", "last"),
            daily_volume=("volume", "sum"),
        )
        .dropna()
        .reset_index()
    )
    
    # 2. 将长格式转为宽格式
    daily_crypto_pivot = daily_crypto_df.pivot(
        index="time_bucket", columns="coin_name", values=["daily_close", "daily_volume"]
    )
    daily_crypto_pivot.columns = [
        f"{coin.upper()}_close" if metric == 'daily_close' else f"{coin.upper()}_volume"
        for metric, coin in daily_crypto_pivot.columns
    ]
    # 移除全部是NA的列
    daily_crypto_pivot.dropna(axis=1, how='all', inplace=True)


    # 3. 下载并合并宏观数据
    start_date = daily_crypto_pivot.index.min().strftime("%Y-%m-%d")
    end_date = daily_crypto_pivot.index.max().strftime("%Y-%m-%d")
    eco_close_data = load_macro_data(start_date, end_date)
    
    if eco_close_data is None:
        return None, None

    # 4. 合并数据
    # 确保索引时区一致或都没有时区
    if daily_crypto_pivot.index.tz is not None:
        daily_crypto_pivot.index = daily_crypto_pivot.index.tz_localize(None)
    if eco_close_data.index.tz is not None:
        eco_close_data.index = eco_close_data.index.tz_localize(None)

    merged_df = pd.merge(
        daily_crypto_pivot,
        eco_close_data,
        left_index=True,
        right_index=True,
        how="left"
    )
    merged_df = merged_df.ffill().bfill() # 使用ffill和bfill确保开头没有NaN
    returns_df = merged_df.pct_change().dropna()
    
    return merged_df, returns_df

# --- 3. Streamlit 应用主界面 ---

st.title("加密货币与宏观经济分析")
st.markdown("""
这是一个综合性分析仪表盘，结合了分钟级加密货币数据与每日宏观经济指标。
- **数据源**: [Kaggle - 400+ Crypto Currency Pairs](https://www.kaggle.com/datasets/tencars/392-crypto-currency-pairs-at-minute-resolution) & Yahoo Finance.
- **功能**: 应用分为两大模块：**加密货币EDA分析** 和 **加密货币与宏观经济关联分析**。
""")

# 加载基础数据 
hourly_df = load_hourly_crypto_data()

if hourly_df is not None:
    
    # --- 模块一: 加密货币内部对比分析 (基于小时数据) ---
    with st.expander("模块一：加密货币EDA分析 ", expanded=True):
        st.sidebar.header("模块一：筛选器")
        all_coins = sorted(hourly_df['coin_name'].unique())
        default_coins = ['btcusd', 'ethusd', 'ltcusd']
        selected_coins = st.sidebar.multiselect(
            "选择要对比的加密货币:",
            options=all_coins,
            default=[coin for coin in default_coins if coin in all_coins]
        )

        if not selected_coins:
            st.warning("请在左侧侧边栏中至少选择一种加密货币。")
        else:
            display_df = hourly_df[hourly_df['coin_name'].isin(selected_coins)].copy()

            st.subheader("1. 归一化价格表现 (Normalized Price Performance)")
            st.markdown("将所有选定货币的初始价格设为100，以比较它们的相对表现。")
            
            display_df['normalized_close'] = display_df.groupby('coin_name')['close'].transform(
                lambda x: (x / x.iloc[0]) * 100 if not x.empty else x
            )
            
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=display_df, x='time_bucket', y='normalized_close', hue='coin_name', ax=ax1)
            ax1.set_title('归一化价格表现 (起点 = 100)', fontsize=16)
            ax1.set_ylabel('归一化价格')
            ax1.set_xlabel('日期')
            ax1.axhline(100, color='gray', linestyle='--', linewidth=1)
            ax1.legend(title='货币')
            st.pyplot(fig1)
            with st.container(border=True):
                st.markdown("""
                    #### 现象与分析：
                    这三种加密货币都表现出很高的价格波动性。图表清楚地说明了两个主要的“牛熊周期”：

                    - **2017-2018年**：在2017年，三种货币的价格都经历了爆发式增长，形成了一个急剧的峰值，然后在2018年迅速崩盘。

                    - **2020-2022年**：从2020年底开始，市场进入更大规模的牛市。价格再次飙升，在2021年达到历史新高（图中显示的是“双顶”模式，峰值出现在上半年和下半年）。随后，市场在2022年再次进入深度熊市。

                    **比特币（BTC）**：作为市场的领导者，比特币表现出了最强劲的长期增长。在两个牛市周期中，其正常化价格都达到了最高点。这在2021年牛市期间尤其明显，其正常化价格超过了7万美元，意味着从起点上涨了700多倍。这表明，在整个审查期间，比特币是表现最好的长期投资资产。

                    **以太坊（ETH）**：ETH推出晚于BTC和LTH（大约2015年），但在2017年牛市期间迅速崛起。特别值得注意的是其在2020-2021年牛市中的表现，其增长轨迹非常陡峭，其正常化表现一度紧跟BTC。

                    **莱特币（LTC）**：作为早期的“替代币”之一，LTC的增长明显落后于BTC和ETH。与两大领先资产之间的业绩差距逐渐扩大。这表明，随着市场的成熟，资金和注意力越来越集中在BTC、ETH等顶级资产上，这些资产具有更强的网络效应和更发达的应用生态系统。
                        """)
            
            st.subheader("2. 每小时回报率分布对比")
            st.markdown("通过观察回报率的分布来理解不同货币的波动性。曲线越高、越窄，代表波动性越小。")
            
            display_df['hourly_return'] = display_df.groupby('coin_name')['close'].pct_change()

            fig2, ax2 = plt.subplots(figsize=(12, 7))
            sns.kdeplot(data=display_df.dropna(), x='hourly_return', hue='coin_name', fill=True, common_norm=False, alpha=0.4, ax=ax2)
            ax2.set_title('主要加密货币每小时回报率分布', fontsize=16)
            ax2.set_xlabel('每小时回报率')
            ax2.set_ylabel('密度')
            ax2.set_xlim(-0.05, 0.05)
            #ax2.legend(title='货币')
            st.pyplot(fig2)
            with st.container(border=True):
                st.markdown("""
                #### 现象与分析：
                BTC和ETH的短期价格稳定性相似且高，大部分时间波动性较小。LTC价格在每小时的水平上更容易出现较大的波动幅度。
                """)
        


                
    # --- 模块二: 加密货币与宏观经济关联分析  ---
    with st.expander("模块二：加密货币与宏观经济关联分析 ", expanded=True):
        st.markdown("---")
        st.header("加密货币与宏观经济关联分析")
        
        with st.spinner("正在准备每日数据并进行宏观分析..."):
            merged_df, returns_df = prepare_daily_merged_data(hourly_df)

        if returns_df is None or returns_df.empty:
            st.error("无法生成宏观分析报告，因为数据合并或处理失败。")
        else:
            # 定义分析中常用的列
            crypto_price_cols = [col for col in returns_df.columns if '_close' in col]
            macro_cols = ['SP500', 'NASDAQ', 'VIX', 'Gold', 'USD_Index']
            
            # --- 使用 Tabs 组织不同的分析 ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "BTC vs 股市表现", 
                "相关性矩阵分析", 
                "对冲能力对比", 
                "动态关系分析", 
                "因果与事件研究"
            ])

            with tab1:
                st.subheader("BTC 与 NASDAQ 每日收益率对比")
                if "BTCUSD_close" in returns_df.columns and "NASDAQ" in returns_df.columns:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    returns_df["BTCUSD_close"].plot(ax=ax, label="BTC 每日收益率", color="orange", alpha=0.9)
                    returns_df["NASDAQ"].plot(ax=ax, label="NASDAQ 每日收益率", color="blue", alpha=0.7)
                    ax.set_title("BTC vs. NASDAQ Daily Returns", fontsize=16)
                    ax.set_ylabel("每日收益率")
                    ax.set_xlabel("日期")
                    ax.legend(loc="upper left")
                    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
                    st.pyplot(fig)
                else:
                    st.warning("缺少 'BTCUSD_close' 或 'NASDAQ' 数据，无法绘制对比图。")

                with st.container(border=True):
                    st.markdown("""
                    #### 现象与分析：
                    这张图表展示了BTC和NASDAQ指数每日收益率的关系，比较两种资产的每日涨跌幅。
                    - **高波动性**: BTC每日收益率的振幅（橙色线）远大于NASDAQ（蓝色线），这表明比特币是一种高波动性、高风险的资产。NASDAQ的波动则小得多，代表了传统股票市场相对更成熟、更稳定的特性。
                    - **第一阶段 (大约2019年以前)**: 两条线的走势几乎没有明显的同步性。比特币更多被看作是一个与传统金融市场不相关的独立、小众投机品。
                    - **第二阶段 (大约2020年至今)**: 我们能看到两条线的波动开始出现明显的同步性（同涨同跌）。这表明随着机构投资者的入场，比特币被更多地纳入宏观经济框架，成为宏观经济环境下的一个“风险资产”。
                    """)

                st.subheader("疫情以来 (2019-2022) 的累计回报对比")
                if "BTCUSD_close" in returns_df.columns and "NASDAQ" in returns_df.columns:
                    covid_period_returns = returns_df.loc['2019-01-01':'2022-12-31']
                    if not covid_period_returns.empty:
                        cumulative_returns = (1 + covid_period_returns[['BTCUSD_close', 'NASDAQ']]).cumprod() * 100
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        cumulative_returns['BTCUSD_close'].plot(ax=ax, label="BTC", color="orange")
                        cumulative_returns['NASDAQ'].plot(ax=ax, label="NASDAQ", color="blue")
                        ax.set_yscale('log') # 使用对数坐标轴以便观察
                        ax.set_title("疫情以来累计回报对比 (2019-2022, 初始投资$100)", fontsize=16)
                        ax.set_ylabel("资产价值 (对数坐标)")
                        ax.set_xlabel("日期")
                        ax.axhline(100, color="grey", linestyle="--")
                        ax.legend()
                        st.pyplot(fig)
                    else:
                        st.warning("在 2019-2022 时间范围内无数据。")
                else:
                    st.warning("缺少 'BTCUSD_close' 或 'NASDAQ' 数据，无法绘制累计回报图。")

                with st.container(border=True):
                    st.markdown("""
                    #### 现象与分析：
                    这张图展示了如果在2019年初同时投资$100到比特币和纳斯达克指数，到2022年底资产的最终价值。
                    - **回报数量级**: 使用对数坐标轴可以清晰地看到，尽管两者都经历了剧烈波动，但比特币在该期间的整体增长数量级远超纳斯达克指数。
                    - **风险与机遇**: 比特币的日波动率显著高于纳斯达克，这意味着它的价格波动更剧烈，风险与机遇并存。
                    - **同向移动**: 一个正的相关系数表明，在疫情期间，比特币和纳斯达克市场在一定程度上倾向于同向移动（同涨同跌），这削弱了比特币作为传统金融市场“避险资产”的说法。
                    - **关键节点**:
                        - **疫情冲击 (2020年初)**: 两者都出现了大幅下跌。
                        - **大牛市 (2020年底 - 2021年)**: 比特币经历了爆炸性增长，回报率远超纳斯达克。
                        - **回调/熊市 (2022年)**: 两者都从高点回落。
                    """)
            
            with tab2:
                st.subheader("核心资产相关性矩阵分析")
                st.markdown("""
                通过相关性矩阵，我们可以量化不同资产价格走势的同步性。数值越接近1，表示同涨同跌；越接近-1，表示走势相反；接近0则表示关联性不强。
                """)

                # 关键步骤：定义我们感兴趣的核心资产，并创建一个从原始列名到显示名称的映射
                # 这个定义将统一应用于下面的两个图表
                target_assets_map = {
                    "BTCUSD_close": "BTC",
                    "ETHUSD_close": "ETH",
                    "LTCUSD_close": "LTC",
                    "SP500": "SP500",
                    "NASDAQ": "NASDAQ",
                    "VIX": "VIX",
                    "Gold": "Gold",
                    "USD_Index": "USD_Index"
                }
                # 从您的 returns_df 中，筛选出实际存在的列
                cols_to_correlate = [col for col in target_assets_map.keys() if col in returns_df.columns]
   

                # --- 全周期分析 ---
                st.subheader("核心资产相关性矩阵 (全周期)")
                
                print("\n[分析步骤] 正在计算全周期核心资产相关性矩阵...")

                # 仅选择我们感兴趣的列进行分析
                full_period_returns = returns_df[cols_to_correlate].copy()
                
                # 将列名替换为更简洁的显示名称
                full_period_returns.columns = [target_assets_map[col] for col in full_period_returns.columns]

                if not full_period_returns.empty and full_period_returns.shape[1] > 1:
                    correlation_matrix = full_period_returns.corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title("每日收益率相关性矩阵 (全周期)", fontsize=16)
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                else:
                    st.warning("无足够数据生成全周期核心资产相关性矩阵。")

                # --- 近周期分析 (2021年至今) ---
                st.subheader("核心资产相关性矩阵 (2021年至今)")
                
                print("\n[分析步骤] 正在计算近周期（2021年至今）核心资产相关性矩阵...")

                recent_returns_df = returns_df.loc['2021-01-01':].copy()
                
                if not recent_returns_df.empty:
                    # 【修正点】在这里，我们严格复用上面定义的 cols_to_correlate 和 target_assets_map
                    # 确保筛选的资产范围和重命名逻辑与“全周期”图完全一致
                    recent_cols_in_df = [col for col in cols_to_correlate if col in recent_returns_df.columns]
                    
                    if len(recent_cols_in_df) > 1:
                        recent_period_returns = recent_returns_df[recent_cols_in_df].copy()
                        recent_period_returns.columns = [target_assets_map[col] for col in recent_period_returns.columns]

                        recent_correlation_matrix = recent_period_returns.corr()

                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(recent_correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                        ax.set_title("每日收益率相关性矩阵 (2021年至今)", fontsize=16)
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)
                    else:
                        st.warning("在选定资产中，无足够数据生成近周期相关性矩阵。")
                else:
                    st.warning("无2021年至今的数据，无法生成近周期相关性矩阵。")

                with st.container(border=True):
                    # 分析文本现在可以放心地让用户对比两张图，因为两张图的资产范围是一致的
                    st.markdown("""
                    #### 现象与分析 (对比两图)：
                    通过对比“全周期”与“2021年至今”两张相关性矩阵，我们可以洞察市场结构的演变：
                    - **与股市关联是否加深**: 比较两图中 BTC/ETH 与 SP500/NASDAQ 的相关性数值。如果近期的数值显著高于全周期，则标志着加密货币越来越受传统金融市场情绪和宏观因素影响，其作为“风险分散”工具的独特性在下降。
                    - **内部联动性**: 观察 BTC、ETH、LTC 之间的相关性数值。如果近期数值普遍升高，表明加密货币市场变得更加成熟和一体化，龙头币种的走势联动性更强。
                    - **与美元负相关是否加深**: 比较 BTC 与美元指数(USD_Index)的负相关性。如果近期的负值绝对值更大（如从-0.1变为-0.2），说明其在全球资产配置中的地位提升，对美元强弱更敏感。
                    - **“数字黄金”叙事的变化**: 观察 BTC 与黄金(Gold)的关系。如果相关性始终在0附近徘徊，甚至在近期变得更低，则进一步削弱了“数字黄金”的说法。
                    """)
            
            with tab3:
                # --- 对冲属性分析 - 比特币 vs 黄金 (这是新增的部分) ---
                st.subheader("对冲属性分析：比特币 vs. 黄金")
                
                # 检查所需列是否存在于我们处理好的DataFrame中
                required_cols_hedge = ['BTC', 'SP500', 'Gold']
                if all(col in full_period_returns.columns for col in required_cols_hedge):
                    
                    # 从全周期相关性矩阵中直接获取数值
                    btc_sp500_corr = correlation_matrix.loc['BTC', 'SP500']
                    gold_sp500_corr = correlation_matrix.loc['Gold', 'SP500']

                    # 创建并排的两个子图
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                    fig.suptitle('比特币 vs. 黄金：对冲股市(S&P 500)的能力对比', fontsize=18, fontweight='bold')

                    # 子图1: 比特币 vs. S&P 500
                    sns.regplot(x='SP500', y='BTC', data=full_period_returns, ax=ax1,
                                scatter_kws={'alpha': 0.3, 's': 15},
                                line_kws={'color': 'red', 'linestyle': '--'})
                    ax1.set_title(f'比特币 vs. S&P 500 (相关性: {btc_sp500_corr:.3f})', fontsize=14)
                    ax1.set_xlabel('S&P 500 每日收益率')
                    ax1.set_ylabel('比特币 每日收益率')
                    ax1.axhline(0, color='grey', linestyle='-', linewidth=1)
                    ax1.axvline(0, color='grey', linestyle='-', linewidth=1)

                    # 子图2: 黄金 vs. S&P 500
                    sns.regplot(x='SP500', y='Gold', data=full_period_returns, ax=ax2,
                                scatter_kws={'alpha': 0.3, 's': 15, 'color': 'goldenrod'},
                                line_kws={'color': 'blue', 'linestyle': '--'})
                    ax2.set_title(f'黄金 vs. S&P 500 (相关性: {gold_sp500_corr:.3f})', fontsize=14)
                    ax2.set_xlabel('S&P 500 每日收益率')
                    ax2.set_ylabel('黄金 每日收益率')
                    ax2.axhline(0, color='grey', linestyle='-', linewidth=1)
                    ax2.axvline(0, color='grey', linestyle='-', linewidth=1)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
                    st.pyplot(fig)

                    with st.container(border=True):
                        st.markdown(f"""
                        #### 现象与分析：
                        上图通过散点回归直观地比较了比特币和黄金相对于美股大盘(S&P 500)的关联性。
                        - **比特币 vs. S&P 500**:
                            - 全周期相关性为 **{btc_sp500_corr:.3f}**。
                            - 散点图的趋势线清晰地展示了它们之间的正相关关系。当S&P 500上涨时，比特币倾向于上涨，反之亦然。这表明比特币更像一种高风险的科技股，而非避险资产。
                        - **黄金 vs. S&P 500**:
                            - 全周期相关性为 **{gold_sp500_corr:.3f}**。
                            - 这个接近于零的相关性，以及散点图中几乎水平的趋势线，证明了黄金与股市的走势几乎没有关联，使其成为分散股市风险的传统、有效的对冲工具。
                        """)
                else:
                    st.warning("缺少 BTC, SP500, 或 Gold 数据，无法进行对冲属性分析。")

            with tab4:
                st.header("动态关系与对冲分析")

                # --- 动态关系分析 - 30日滚动相关性 (这是您原有的部分) ---
                st.subheader("动态关系分析 - 30日滚动相关性")
                
                # 检查所需列是否存在
                required_cols_rolling = ["BTCUSD_close", "NASDAQ", "VIX"]
                if all(col in returns_df.columns for col in required_cols_rolling):
                    rolling_corr_nasdaq = returns_df['BTCUSD_close'].rolling(window=30).corr(returns_df['NASDAQ'])
                    rolling_corr_vix = returns_df['BTCUSD_close'].rolling(window=30).corr(returns_df['VIX'])

                    fig, ax = plt.subplots(figsize=(12, 6))
                    rolling_corr_nasdaq.plot(ax=ax, label="BTC vs NASDAQ", color="blue")
                    rolling_corr_vix.plot(ax=ax, label="BTC vs VIX", color="red")
                    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
                    ax.set_title("BTC 与 NASDAQ/VIX 的30日滚动相关性", fontsize=16)
                    ax.set_ylabel("相关系数")
                    ax.set_xlabel("日期")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("缺少 BTCUSD_close, NASDAQ, 或 VIX 数据，无法进行滚动相关性分析。")

                with st.container(border=True):
                    st.markdown("""
                    #### 现象与分析：
                    这张图展示了BTC的日收益率与NASDAQ和VIX指数之间的30天滚动相关性，揭示了它们关系随时间的变化。
                    - **BTC vs NASDAQ (蓝线)**: 观察此线是否在2020年后长期处于正值区间，这通常被解读为BTC的“风险资产”属性。
                    - **BTC vs VIX (红线)**: 观察此线是否长期处于负相关。当市场恐慌情绪上升（VIX上涨）时，若比特币价格倾向于下跌，则证明其不具备避险属性。
                    """)
                
                st.divider() # 添加一个分隔线，让布局更清晰



            with tab5:
                st.header("因果与滞后关系分析")

                # --- VIX vs BTC 领先滞后分析 (Lead-Lag Analysis) ---
                st.subheader("VIX vs BTC 领先滞后分析")
                st.markdown("通过计算不同时间平移下的相关性，我们可以探究一个时间序列的变化是否会领先或滞后于另一个。正滞后天数意味着VIX领先BTC，负滞后天数意味着VIX滞后于BTC。")

                required_cols_lag = ["VIX", "BTCUSD_close"]
                if all(col in returns_df.columns for col in required_cols_lag):
                    
                    # 定义要测试的滞后范围
                    lags_to_test = range(-10, 11)
                    lead_lag_correlations = {}

                    # 使用 shift() 方法手动计算双向相关性
                    for lag in lags_to_test:
                        # 当 lag > 0, VIX.shift(lag) 使用的是过去的数据，代表 VIX 领先
                        # 当 lag < 0, VIX.shift(lag) 使用的是未来的数据，代表 VIX 滞后
                        correlation = returns_df['BTCUSD_close'].corr(returns_df['VIX'].shift(lag))
                        lead_lag_correlations[lag] = correlation
                    
                    # 转换为 Pandas Series 便于分析和绘图
                    lead_lag_series = pd.Series(lead_lag_correlations).dropna()

                    if not lead_lag_series.empty:
                        # 找到相关性绝对值最大的点
                        max_abs_corr_lag = lead_lag_series.abs().idxmax()
                        max_abs_corr_val = lead_lag_series[max_abs_corr_lag]

                        # --- 开始绘图 ---
                        fig, ax = plt.subplots(figsize=(12, 7))
                        
                        # 使用柱状图来清晰地展示每个滞后期的相关性
                        lead_lag_series.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', width=0.8,
                                             label='VIX-BTC Correlation')
                        
                        # 高亮显示最大绝对值相关性的位置
                        ax.axvline(x=list(lags_to_test).index(max_abs_corr_lag), color='red', linestyle='--', 
                                   label=f"最大绝对值相关性在滞后 {max_abs_corr_lag} 天 (Corr={max_abs_corr_val:.3f})")

                        ax.set_title("VIX 收益率 vs BTC 收益率的领先-滞后关系", fontsize=16)
                        ax.set_xlabel("VIX 滞后天数 (正值 = VIX 领先 BTC)")
                        ax.set_ylabel("相关系数")
                        ax.axhline(0, color='black', linewidth=0.8)
                        ax.legend()
                        st.pyplot(fig)

                        with st.container(border=True):
                            st.markdown(f"""
                            #### 现象与分析：
                            上图展示了VIX（恐慌指数）与BTC日收益率在不同时间平移下的相关性。
                            - **核心发现**: 在所有测试的时间偏移中，相关性绝对值最大的点出现在 **VIX 滞后 {max_abs_corr_lag} 天** 的位置，相关系数为 **{max_abs_corr_val:.3f}**传统市场恐慌情绪的变化，可能在几天后才完全反映在比特币的价格波动上。
                            - **解读**:
                                - 如果滞后天数为正（例如 `+1`），意味着VIX今天的变化与BTC明天的变化相关性最强（VIX领先）。
                                - 如果滞后天数为负（例如 `-1`），意味着VIX今天的变化与BTC昨天的变化相关性最强（VIX滞后，或BTC领先）。
                                - 如果滞后天数为 `0`，意味着两者的变化在当天同步性最强。
                            - **结论**: 根据图表，我们可以判断VIX和BTC之间存在一个领先或滞后的“反应时间差”。但请注意，即使存在统计上的领先关系，如果相关系数本身很低（例如绝对值小于0.2），那么其实际预测能力也非常有限。
                            """)
                    else:
                        st.warning("计算出的领先-滞后相关性数据为空，无法绘图。")
                else:
                    missing_cols = [col for col in required_cols_lag if col not in returns_df.columns]
                    st.warning(f"无法进行领先-滞后分析，因为缺少以下数据列: `{', '.join(missing_cols)}`")

                st.divider()

                # --- 事件研究分析 (Event Study) ---
                st.subheader("事件研究分析 (基于2023年数据)")
                st.markdown("通过分析特定宏观事件（如CPI公布、FOMC会议）前后几天的资产价格“异常收益”，我们可以量化这些事件对市场的短期冲击。")

                try:
                    # 1. 识别关键宏观事件：使用2023年真实的美联储议息(FOMC)和CPI公布日期
                    event_dates = {
                        "FOMC": pd.to_datetime(["2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13"]),
                        "CPI": pd.to_datetime(["2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12"]),
                    }
                    
                    # 2. 定义“事件窗口”和目标资产
                    event_window = [-1, 0, 1] # 事件前1天, 当天, 后1天
                    target_asset = "BTCUSD_close" 

                    # 3. 计算“累积异常收益率”Cumulative Abnormal Returns(CAR)
                    all_event_returns = []
                    for event_type, dates in event_dates.items():
                        for date in dates:
                            if date in returns_df.index:
                                for day_offset in event_window:
                                    window_date = date + pd.Timedelta(days=day_offset)
                                    if window_date in returns_df.index:
                                        ret = returns_df.loc[window_date, target_asset]
                                        all_event_returns.append({"event_type": event_type, "event_date": date, "return": ret})

                    if not all_event_returns:
                        st.warning("在您的数据范围内未找到任何定义的事件日期，无法进行事件研究分析。")
                    else:
                        event_returns_df = pd.DataFrame(all_event_returns)
                        car_df = event_returns_df.groupby(["event_type", "event_date"])["return"].sum().reset_index()
                        car_df = car_df.rename(columns={"return": "CAR"})

                        # 4. 可视化结果 - 箱线图
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x="event_type", y="CAR", data=car_df, ax=ax)
                        ax.axhline(0, color="red", linestyle="--")
                        ax.set_title(f"{target_asset} 在宏观事件窗口期内的累积异常收益(CAR)", fontsize=16)
                        ax.set_ylabel(f"窗口期 {event_window} 内的CAR")
                        ax.set_xlabel("事件类型")
                        st.pyplot(fig)

                        # 5. 进行统计检验并展示结果
                        st.markdown("#### 统计显著性检验 (T-test)")
                        st.markdown("我们使用单样本T检验来判断各类事件的平均CAR是否显著不为零。")

                        col1, col2 = st.columns(2)
                        
                        for i, event_type in enumerate(car_df["event_type"].unique()):
                            cars = car_df[car_df["event_type"] == event_type]["CAR"]
                            t_stat, p_value = stats.ttest_1samp(cars, 0)
                            
                            target_col = col1 if i % 2 == 0 else col2
                            with target_col:
                                with st.expander(f"**{event_type} 事件分析结果**", expanded=True):
                                    st.metric(label="平均累积异常收益 (Mean CAR)", value=f"{cars.mean():.4f}")
                                    st.metric(label="P-value", value=f"{p_value:.3f}")
                                    if p_value < 0.05:
                                        st.success("结论: 影响在统计上显著。")
                                    else:
                                        st.warning("结论: 影响在统计上不显著。")

                except Exception as e:
                    st.error(f"在执行事件研究分析时发生错误: {e}")
                    st.info("请检查 `returns_df` 中是否包含 `BTCUSD_close` 列，以及数据时间范围是否覆盖2023年。")
else:
    st.error("数据加载失败，无法显示仪表盘。请检查终端中的错误信息。")