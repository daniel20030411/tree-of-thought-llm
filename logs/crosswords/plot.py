import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = 'dfs+sd_board value'
title = file_name

# 創建 DataFrame 包含你的數據
data = {
    'x': [0.1, 0.4, 0.4, 0.4, 0, 0.8, 0.4, 0.5, 0.2, 0.6, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0, 0, 0.2, 0.3],
    'y': [131.5294769, 106.5948329, 38.18687057, 76.56371498, 139.3860712, 81.13337493, 63.90065002, 47.34281301, 110.5746334, 260.5594647, 42.03254557, 77.67291284, 73.75355768, 43.52544403, 0, 29.5215888, 33.35365224, 0, 96.22045159, 80.1736238]
}

print(f'length x: {len(data["x"])}\nlength y: {len(data["y"])}\n')

df = pd.DataFrame(data)

# 計算不同 x 值的個數
x_counts = df['x'].value_counts().sort_index()

# 計算主要 Y 軸的上限
y_limit_main = x_counts.max() + 1

# 計算次要 Y 軸的上限
y_limit_secondary = df['y'].max() + 50

# 創建一個新的圖表
fig, ax1 = plt.subplots()

# 繪製長條圖
ax1.bar(x_counts.index, x_counts.values, width=0.03, color='#83CBEB', alpha=0.7, label='Count')
ax1.set_xlabel('r_word')
ax1.set_ylabel('Count (n)', color='black')

# 設置 x 軸範圍和刻度
plt.xticks(np.arange(0, 1.0, 0.1))

# 設置主要 Y 軸的上限
ax1.set_ylim(0, y_limit_main)

# 創建次要坐標軸
ax2 = ax1.twinx()

# 繪製散點圖，調整點的大小為 30
ax2.scatter(df['x'], df['y'], color='red', label='Cost Time', s=10)
ax2.set_ylabel('time (s)', color='black')

# 設置次要 Y 軸的上限
ax2.set_ylim(0, y_limit_secondary)

# 添加標題
plt.title(f'{title}, r_word and cost time')

# 添加圖例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# 儲存圖表
plt.savefig(f'C:\EMA\llama2\\text-generation-webui\\tree-of-thought-llm\logs\crosswords\\record\plot\{file_name}.png')

# 顯示圖表
plt.show()
