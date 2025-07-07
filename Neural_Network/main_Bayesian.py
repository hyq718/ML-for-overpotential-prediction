import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from N_N_Bayesian import *

# 数据导入
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

X_train = train_data.drop(['Overpotential(mV)'], axis=1).values
y_train = train_data['Overpotential(mV)'].values

X_test = test_data.drop(['Overpotential(mV)'], axis=1).values
y_test = test_data['Overpotential(mV)'].values

# 模型训练（只需要一次）
model = BayesianNeuralNetwork(
    input_size=X_train.shape[1],
    hidden_layers=[64, 32],
    activation='relu',
    learning_rate=0.001,
    epochs=2000,
    kl_weight=1e-4
)
model.train(X_train, y_train)

# 预测
y_pred = model.predict(X_test, n_samples=20)

# 评估
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

print("\n预测结果 vs 实际值：")
for i in range(len(y_test)):
    print(f"Sample {i+1:>3}: Predicted = {y_pred[i][0]:.2f}, Actual = {y_test[i]:.2f}")