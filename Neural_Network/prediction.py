import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, output_size=1, learning_rate=0.01, epochs=2000, l2_lambda=0.001):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        
        # 初始化权重和偏置
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden1_size))

        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2. / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))

        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * np.sqrt(2. / self.hidden2_size)
        self.b3 = np.zeros((1, self.output_size))
        
        # 标准化器
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def fit(self, X_train, y_train):
        # 数据预处理
        X_scaled = self.x_scaler.fit_transform(X_train)
        y_scaled = self.y_scaler.fit_transform(y_train.reshape(-1, 1))
        
        # 训练过程
        for epoch in range(1, self.epochs + 1):
            # 前向传播
            Z1 = np.dot(X_scaled, self.W1) + self.b1
            A1 = self.relu(Z1)

            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.relu(Z2)

            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = Z3  # 线性输出

            # 损失 + L2 正则
            mse_loss = np.mean((y_scaled - A3) ** 2)
            l2_loss = self.l2_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.W3**2))
            total_loss = mse_loss + l2_loss

            # 反向传播
            m = X_scaled.shape[0]
            dA3 = 2 * (A3 - y_scaled) / m
            dW3 = np.dot(A2.T, dA3) + 2 * self.l2_lambda * self.W3
            db3 = np.sum(dA3, axis=0, keepdims=True)

            dA2 = np.dot(dA3, self.W3.T) * self.relu_derivative(Z2)
            dW2 = np.dot(A1.T, dA2) + 2 * self.l2_lambda * self.W2
            db2 = np.sum(dA2, axis=0, keepdims=True)

            dA1 = np.dot(dA2, self.W2.T) * self.relu_derivative(Z1)
            dW1 = np.dot(X_scaled.T, dA1) + 2 * self.l2_lambda * self.W1
            db1 = np.sum(dA1, axis=0, keepdims=True)

            # 更新参数
            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1

            # 打印损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, MSE Loss: {mse_loss:.6f}, Total Loss: {total_loss:.6f}")

    def predict(self, X_test):
        # 对输入进行标准化
        X_scaled = self.x_scaler.transform(X_test)

        # 前向传播
        Z1 = np.dot(X_scaled, self.W1) + self.b1
        A1 = self.relu(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.relu(Z2)

        Z3 = np.dot(A2, self.W3) + self.b3
        predictions_scaled = Z3

        # 将预测结果反标准化
        predictions = self.y_scaler.inverse_transform(predictions_scaled)
        return predictions

# 数据导入
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')

# 特征和标签
X_train = train_data.drop(['Overpotential(mV)'], axis=1)
y_train = train_data['Overpotential(mV)'].values

X_test = test_data.drop(['Overpotential(mV)'], axis=1)
y_test = test_data['Overpotential(mV)'].values

# 创建神经网络实例
model = NeuralNetwork(input_size=X_train.shape[1])

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
predictions = model.predict(X_test)

# 计算 MSE
mse = mean_squared_error(y_test, predictions)

# 输出所有预测结果和 MSE
print("\n所有预测结果:\n", predictions)
print("\n测试集上的 MSE 值:", mse)
