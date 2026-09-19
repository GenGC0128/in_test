from down_stock_data import StockDataIncremental
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import baostock as bs


def get_Sector_pic(stock_list,num):
    downloader = StockDataIncremental()
    stock_list = stock_list
    #stock_list = list(stock_df["code"])
    downloader = StockDataIncremental()
    prices = downloader.get_price_matrix_batch(stock_list=stock_list, start_date="2005-01-01", end_date=None)
    prices_df = prices
    df = prices_df[num:]
    returns_df = df.pct_change().dropna()

    # 2. 计算DTW距离矩阵
    stock_codes = returns_df.columns.tolist()
    n_stocks = len(stock_codes)
    distance_matrix = np.zeros((n_stocks, n_stocks))

    for i in range(n_stocks):
        for j in range(i+1, n_stocks): # 利用对称性，只计算一半
            series_i = returns_df.iloc[:, i].values
            series_j = returns_df.iloc[:, j].values
            distance, _ = fastdtw(series_i, series_j)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance # 对称矩阵

    # 3. 层次聚类
    # 将距离矩阵压缩成凝聚性聚类算法需要的格式
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method='ward') # 使用Ward方法连接

    # 4. 绘制树状图
    plt.figure(figsize=(14, 8))
    dendrogram(Z, labels=stock_codes, leaf_rotation=90)
    plt.title('Stock Clustering Dendrogram')
    plt.tight_layout()
    plt.show()

    # 5. 获取聚类结果 (例如，设定距离阈值t，或指定簇的数量)
    # 方法一：通过高度/距离阈值切割
    clusters = fcluster(Z, t=50, criterion='distance')
    # 方法二：指定簇的数量
    # clusters = fcluster(Z, 10, criterion='maxclust')

    # 将结果存入DataFrame
    result_df = pd.DataFrame({'Stock': stock_codes, 'Cluster': clusters})

    # 6. 可视化某一个簇的走势 (例如，查看簇1)
    cluster_id = 1
    cluster_stocks = result_df[result_df['Cluster'] == cluster_id]['Stock'].tolist()

    plt.figure(figsize=(14, 8))
    for stock in cluster_stocks:
        cum_return = (1 + returns_df[stock]).cumprod() - 1
        plt.plot(cum_return.index, cum_return.values, label=stock)
    plt.title(f'Cumulative Returns of Stocks in Cluster {cluster_id}')
    plt.legend()
    plt.show()





def find_similar_stocks(target_code, stock_codes, returns_df, distance_matrix, threshold):
    """
    根据给定的股票代码，从股票池中选择距离小于特定阈值的相似股票列表
    
    参数:
    target_code: str - 目标股票代码
    stock_codes: list - 所有股票代码列表
    returns_df: DataFrame - 收益率数据框，列为股票代码，索引为日期
    distance_matrix: ndarray - 预计算的距离矩阵
    threshold: float - 距离阈值
    
    返回:
    similar_stocks: list - 相似股票列表，包含目标股票本身
    distances: list - 对应的距离值
    """
    
    # 检查目标股票是否在股票池中
    if target_code not in stock_codes:
        raise ValueError(f"目标股票 {target_code} 不在股票池中")
    
    # 获取目标股票在距离矩阵中的索引
    target_idx = stock_codes.index(target_code)
    
    # 获取目标股票与其他所有股票的距离
    target_distances = distance_matrix[target_idx, :]
    
    # 找出距离小于阈值的股票索引
    similar_indices = np.where(target_distances <= threshold)[0]
    
    # 获取相似股票代码和对应的距离
    similar_stocks = [stock_codes[i] for i in similar_indices]
    distances = [target_distances[i] for i in similar_indices]
    
    # 按距离排序（从近到远）
    sorted_pairs = sorted(zip(similar_stocks, distances), key=lambda x: x[1])
    similar_stocks_sorted, distances_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    return list(similar_stocks_sorted), list(distances_sorted)






def find_similar_stocks_without_matrix(target_code, stock_codes, returns_df, threshold, max_stocks=None):
    """
    无需预计算距离矩阵的版本，按需计算距离（适合股票池较大时）
    
    参数:
    target_code: str - 目标股票代码
    stock_codes: list - 所有股票代码列表
    returns_df: DataFrame - 收益率数据框
    threshold: float - 距离阈值
    max_stocks: int - 返回的最大股票数量（可选）
    
    返回:
    similar_stocks: list - 相似股票列表，包含目标股票本身
    distances: list - 对应的距离值
    """
    
    if target_code not in stock_codes:
        raise ValueError(f"目标股票 {target_code} 不在股票池中")
    
    # 获取目标股票的收益率序列
    target_series = returns_df[target_code].values
    
    similar_stocks = []
    distances = []
    un_finded_code = []
    
    for stock in stock_codes:
        try:
            if stock == target_code:
                # 目标股票自身的距离为0
                similar_stocks.append(stock)
                distances.append(0.0)
            else:
                # 计算与目标股票的距离
                stock_series = returns_df[stock].values
                distance, _ = fastdtw(target_series, stock_series)

                if distance <= threshold:
                    similar_stocks.append(stock)
                    distances.append(distance)
        except:
            un_finded_code.append(stock)
            #print("未找到code")
    
    # 按距离排序
    sorted_pairs = sorted(zip(similar_stocks, distances), key=lambda x: x[1])
    similar_stocks_sorted, distances_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # 如果指定了最大数量，进行截断
    if max_stocks and len(similar_stocks_sorted) > max_stocks:
        similar_stocks_sorted = similar_stocks_sorted[:max_stocks]
        distances_sorted = distances_sorted[:max_stocks]
        
    print(un_finded_code)
    
    return list(similar_stocks_sorted), list(distances_sorted)






def visualize_similar_stocks(target_code, similar_stocks, returns_df, top_n=10):
    """
    可视化相似股票的累积收益率走势
    
    参数:
    target_code: str - 目标股票代码
    similar_stocks: list - 相似股票列表
    returns_df: DataFrame - 收益率数据框
    top_n: int - 显示最相似的前N只股票（包括目标股票）
    """
    
    # 取前top_n只股票进行可视化
    stocks_to_plot = similar_stocks[:min(top_n, len(similar_stocks))]
    
    plt.figure(figsize=(14, 8))
    
    for i, stock in enumerate(stocks_to_plot):
        cum_return = (1 + returns_df[stock]).cumprod() - 1
        
        # 目标股票用粗线突出显示
        if stock == target_code:
            plt.plot(cum_return.index, cum_return.values, 
                    linewidth=3, label=f'{stock} (目标)', alpha=0.9)
        else:
            plt.plot(cum_return.index, cum_return.values, 
                    linewidth=1.5, label=stock, alpha=0.7)
    
    plt.title(f'与 {target_code} 最相似的前{len(stocks_to_plot)}只股票走势', fontsize=14)
    plt.xlabel('日期')
    plt.ylabel('累积收益率')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
