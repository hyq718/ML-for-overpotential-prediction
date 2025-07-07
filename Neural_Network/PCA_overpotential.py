import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# 获取并归一化过电位值
overpotential = df['Overpotential(mV)']
norm = (overpotential - overpotential.min()) / (overpotential.max() - overpotential.min())

# 绘图
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    pca_result[:, 0], pca_result[:, 1],
    c=norm, cmap='coolwarm', edgecolor='k'
)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Scatter Plot Colored by Overpotential')
plt.colorbar(scatter, label='Normalized Overpotential')
plt.grid(True)
plt.tight_layout()
plt.show()
