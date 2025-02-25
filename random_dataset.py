import random
import csv
import pandas as pd

file_path = 'datasetavailable.csv'
df = pd.read_csv(file_path)

#A位元素的列名称，从第4列到第17列
columns_for_Aelements = list(df.columns[3:17])
#B位元素的列名称，从第18列到第21列
columns_for_Belements = list(df.columns[17:21])
#其他需生成的特征符名称，第2、3、22列
columns_for_others = [df.columns[1],df.columns[2],df.columns[21]]
#所有的列
columns_for_all = list(df.columns[1:22])

#对B位元素的随机生成
def generate_b_values(phase):
    feature_values = {feature: 0 for feature in columns_for_Belements}
    if phase == 1:
        target_sum = 2
    elif phase == 3:
        target_sum = 1
    else:
        return feature_values  # 其他情况所有 B 元素为 0

    # 随机生成 4 个数
    random_numbers = [random.uniform(0.1, 0.5) for _ in range(4)]
    total = sum(random_numbers)
    normalized_numbers = [num / total * target_sum for num in random_numbers]

    # 保留两位小数并调整误差
    rounded_numbers = [round(num, 2) for num in normalized_numbers]
    diff = round(target_sum - sum(rounded_numbers), 2)
    max_index = rounded_numbers.index(max(rounded_numbers))
    rounded_numbers[max_index] += diff  # 将误差调整到最大值

    # 填充到对应的 B 位元素
    for feature, value in zip(columns_for_Belements, rounded_numbers):
        feature_values[feature] = value
    
    return feature_values

#随机生成数据
def generate_random_numbers_with_features():
    # 14个特征符
    features = columns_for_Aelements
    
    # 随机选择 4 或 5 个特征符
    num_features = random.choice([4, 5])
    selected_features = random.sample(features, num_features)
    
    # 为选中的特征符生成随机数并归一化
    random_numbers = [random.uniform(0.1, 0.3) for _ in range(num_features)]
    total = sum(random_numbers)
    normalized_numbers = [num / total for num in random_numbers]

    # 保留三位小数并调整最后一个数以确保总和为 1
    rounded_numbers = [round(num, 2) for num in normalized_numbers]
    diff = round(1 - sum(rounded_numbers), 2)
    max_index = rounded_numbers.index(max(rounded_numbers))
    rounded_numbers[max_index] = max(0, rounded_numbers[max_index] + diff)

    # 构造完整的特征符-值对，其中未选中的特征符赋值为 0
    feature_values = {feature: 0 for feature in features}
    for feature, value in zip(selected_features, rounded_numbers):
        feature_values[feature] = value

    # 生成其他特征值
    feature_values[df.columns[1]] = random.randint(1, 7)  # morphology
    phase = random.randint(0, 5)  # phase
    feature_values[df.columns[2]] = phase
    feature_values[df.columns[21]] = random.randint(1, 8)  # WE

    # B 元素随机生成
    b_values = generate_b_values(phase)
    feature_values.update(b_values)

    return feature_values

def generate_and_save_data(filename, num_samples=1000):
    features = columns_for_all
    data = []
    
    # 循环生成数据
    for _ in range(num_samples):
        feature_values = generate_random_numbers_with_features()
        data.append(feature_values)
    
    # 保存到 CSV 文件
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=features)
        writer.writeheader()
        writer.writerows(data)
    print(f"数据已保存到 {filename}")

# 调用函数生成并保存数据
generate_and_save_data("random_features.csv")