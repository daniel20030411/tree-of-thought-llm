import json
import os

record_files_folder = 'C:\EMA\llama2\\text-generation-webui\\tree-of-thought-llm\logs\crosswords\\record'

tree_json_file_name = '{file_path}/tree_crossword_{idx}.json'
node_json_file_name = '{file_path}/node_info_crossword_{idx}.json'
all_json_file_name = '{file_path}/all_tree_crossword.json'
json_result_file_name = '{file_path}/results.json'

acc_file_name = '{file_path}/acc_info_crossword.txt'
record_file_name = '{file_path}/record_crossword_{idx}.txt'

def Init_record_file(file_name, initial_string, idx = 0):
    global record_files_folder
    if not os.path.exists(record_files_folder):
            os.makedirs(record_files_folder)
    with open(file_name.format(file_path = record_files_folder, idx = idx), 'w') as file:
        file.write(initial_string)

def Record_txt(file_name, input_string, idx = 0):
    global record_files_folder
    with open(file_name.format(file_path = record_files_folder, idx = idx), 'a') as file:
        if type(input_string) == str: file.write(input_string)
        elif type(input_string) == list:
             for dictionary in input_string:
                  file.write(str(dictionary) + '\n\n')

def Record_json(file_name, input_dict, idx = 0):
    global record_files_folder
    with open(file_name.format(file_path = record_files_folder, idx = idx), 'a') as file:
        json.dump(input_dict, file, indent = 4)