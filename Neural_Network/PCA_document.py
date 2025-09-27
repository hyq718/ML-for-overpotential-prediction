import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv("dataset_modified.csv") 

# 去除不需要参与分析的列
exclude_cols = ['No', 'Overpotential(mV)', 'tafel(mV/dec)', 'ECSA(cm2)', 'Rct', 'Cdl(mF/cm2)']
features = df.drop(columns=exclude_cols)

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA降维到2维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)

# 获取参考文献分组信息
no_values = df['No']
# 提取整数部分作为文献ID
reference_ids = np.floor(no_values).astype(int)
unique_references = np.unique(reference_ids)
num_references = len(unique_references)

# 创建足够多的颜色 - 使用tab20b(20) + tab20c(20) + 2种额外颜色
colors1 = plt.cm.tab20b(np.linspace(0, 1, 20))
colors2 = plt.cm.tab20c(np.linspace(0, 1, 20))
extra_colors = plt.cm.Set3(np.linspace(0, 1, 2))  # 使用Set3色图的2种颜色
all_colors = np.vstack([colors1, colors2, extra_colors])

# 为每个文献ID分配一个颜色
color_map = dict(zip(unique_references, all_colors[:num_references]))

# 绘图
plt.figure(figsize=(14, 10))
for ref_id, color in color_map.items():
    # 筛选属于当前文献的数据点
    mask = reference_ids == ref_id
    plt.scatter(
        pca_result[mask, 0], pca_result[mask, 1],
        color=color, label=f'Ref {ref_id}', edgecolor='k', s=100
    )

plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.title('PCA Scatter Plot Colored by Reference Source (42 References)', fontsize=14)

# 调整图例位置和大小
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                    title='Reference ID', ncol=2, fontsize=10)
legend.get_title().set_fontsize(12)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()