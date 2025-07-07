import random
import csv
import pandas as pd

file_path = 'train_dataset.csv'
df = pd.read_csv(file_path)

# A位元素的列名称，从第3列到第17列
columns_for_Aelements = list(df.columns[2:16])
# B位元素的列名称，从第18列到第21列
columns_for_Belements = list(df.columns[16:19])
# 所有的列
columns_for_all = list(df.columns[:21])

# 对B位元素的随机生成
def generate_b_values(phase):
    feature_values = {feature: 0 for feature in columns_for_Belements}
    if phase == 1:
        target_sum = 2
    elif phase == 3:
        target_sum = 1
    else:
        return feature_values  # 其他情况所有 B 元素为 0

    random_numbers = [random.uniform(0.1, 0.5) for _ in range(4)]
    total = sum(random_numbers)
    normalized_numbers = [num / total * target_sum for num in random_numbers]

    rounded_numbers = [round(num, 2) for num in normalized_numbers]
    diff = round(target_sum - sum(rounded_numbers), 2)
    max_index = rounded_numbers.index(max(rounded_numbers))
    rounded_numbers[max_index] += diff

    for feature, value in zip(columns_for_Belements, rounded_numbers):
        feature_values[feature] = value

    return feature_values

# 随机生成数据
def generate_random_numbers_with_features():
    feature_values = {feature: 0 for feature in columns_for_all}

    features = columns_for_Aelements
    num_features = random.choice([4, 5])
    selected_features = random.sample(features, num_features)

    random_numbers = [random.uniform(0.1, 0.3) for _ in range(num_features)]
    total = sum(random_numbers)
    normalized_numbers = [num / total for num in random_numbers]

    rounded_numbers = [round(num, 2) for num in normalized_numbers]
    diff = round(1 - sum(rounded_numbers), 2)
    max_index = rounded_numbers.index(max(rounded_numbers))
    rounded_numbers[max_index] = max(0, rounded_numbers[max_index] + diff)

    for feature, value in zip(selected_features, rounded_numbers):
        feature_values[feature] = value

    # 缩小后的 morphology、phase、WE
    morphology = random.randint(1, 7)
    phase = random.randint(0, 5)
    WE = random.randint(1, 8)

    feature_values[df.columns[0]] = round(morphology * 0.1, 2)  # morphology
    feature_values[df.columns[1]] = round(phase * 0.1, 2)       # phase
    feature_values[df.columns[20]] = round(WE * 0.1, 2)         # WE
    

    b_values = generate_b_values(phase)
    feature_values.update(b_values)

    return feature_values

# 生成并保存数据
def generate_and_save_data(filename, num_samples=1000):
    features = columns_for_all
    data = []

    for _ in range(num_samples):
        feature_values = generate_random_numbers_with_features()
        filtered_values = {key: feature_values.get(key, 0) for key in features}
        data.append(filtered_values)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=features)
        writer.writeheader()
        writer.writerows(data)

    print(f"数据已保存到 {filename}")

# 调用函数
generate_and_save_data("random_features.csv")
