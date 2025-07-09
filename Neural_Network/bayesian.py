import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 激活函数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 数据导入
dataset = pd.read_csv('dataset_modified_new.csv')

# 特征和标签
X = dataset.drop(['No', 'tafel(mV/dec)', 'ECSA(cm2)', 'Rct', 'Cdl(mF/cm2)', 'Overpotential(mV)'], axis=1)
y = dataset['Overpotential(mV)'].values.reshape(-1, 1)

# 特征缩放
x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)

# 网络结构
input_size = X.shape[1]
hidden1_size = 32
hidden2_size = 16
output_size = 1

# 超参数
learning_rate = 0.0001
epochs = 2000
l2_lambda = 0.01
kl_weight = 1e-4

# 初始化变分分布的参数（均值和对数方差）
np.random.seed(42)
mu_W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(1. / input_size)
log_var_W1 = np.zeros((input_size, hidden1_size))

mu_b1 = np.zeros((1, hidden1_size))
log_var_b1 = np.zeros((1, hidden1_size))

mu_W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(1. / hidden1_size)
log_var_W2 = np.zeros((hidden1_size, hidden2_size))

mu_b2 = np.zeros((1, hidden2_size))
log_var_b2 = np.zeros((1, hidden2_size))

mu_W3 = np.random.randn(hidden2_size, output_size) * np.sqrt(1. / hidden2_size)
log_var_W3 = np.zeros((hidden2_size, output_size))

mu_b3 = np.zeros((1, output_size))
log_var_b3 = np.zeros((1, output_size))