import pandas as pd

# 原始映射规则（字符串到数字）
morphology_mapping = {
    'Nanoparticle': 1, 'Nanoparticles': 1, 'Nanosheet': 2, 'Nanosheets': 2,
    'Nanotube': 3, 'Nanofibre': 4, 'Nanofibers': 4, 'Nanosphere': 5,
    'Nanoflower': 6, 'Thin film': 7, 'Thin Film': 7
}

phase_mapping = {
    'Amorphous': 0, 'amorphous': 0, 'AB2O4': 1, 'A3O4': 2,
    'ABO3': 3, 'Fd-3m': 4, 'Rocksalt': 5
}

we_mapping = {
    'NF': 1, 'GCE': 2, 'CP': 3, 'NA': 4, 'CR': 5, 'Ti foil': 6,
    'Nb:SrTiO3': 7, 'CC': 8
}

# 反转映射规则
morphology_reverse_mapping = {v: k for k, v in morphology_mapping.items()}
phase_reverse_mapping = {v: k for k, v in phase_mapping.items()}
we_reverse_mapping = {v: k for k, v in we_mapping.items()}

# 读取数据
input_file = 'top_10_lowest_overpotential.csv'
output_file = 'converted_top_10_lowest_overpotential.csv'

data = pd.read_csv(input_file)

# 转换列
if 'Morphology' in data.columns:
    data['Morphology'] = data['Morphology'].map(morphology_reverse_mapping)

if 'Phase ' in data.columns:
    data['Phase '] = data['Phase '].map(phase_reverse_mapping)

if 'WE' in data.columns:
    data['WE'] = data['WE'].map(we_reverse_mapping)

# 保存转换后的文件
data.to_csv(output_file, index=False)
print(f"转换后的文件已保存到: {output_file}")