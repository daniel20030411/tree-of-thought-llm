import json
import pandas as pd
import os

# 读取JSON文件
json_file_path = "C:\EMA\llama2\\text-generation-webui\\tree-of-thought-llm\logs\crosswords\\record\\results.json"
with open(json_file_path, "r") as f:
    data = json.load(f)

# 提取数据
extracted_data = []
for entry in data:
    index = entry.get("index", "")
    best_node_id = entry.get("best node id", "")
    node_depth = entry.get("node depth", "")
    chosen_step = entry.get("chosen step", "")
    distance_to_root = entry.get("distance to root", "")
    r_letter = entry.get("accuracy", {}).get("r_letter", "")
    r_word = entry.get("accuracy", {}).get("r_word", "")
    r_game = entry.get("accuracy", {}).get("r_game", "")
    cost_time = entry.get("cost time", "")

    extracted_data.append([index, best_node_id, node_depth, chosen_step, distance_to_root, r_letter, r_word, r_game, cost_time])

# 创建DataFrame
df = pd.DataFrame(extracted_data, columns=["index", "best node", "node depth", "chosen step", "distance to root", "r_letter", "r_word", "r_game", "cost time"])

# 获取桌面路径
output_path = "C:\EMA\llama2\\text-generation-webui\\tree-of-thought-llm\logs\crosswords"

# 输出Excel文件路径
excel_file_path = os.path.join(output_path, "output.xlsx")

# 将数据写入Excel文件
df.to_excel(excel_file_path, index=False)
