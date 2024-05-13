import json
from tot.prompts.crosswords import propose_prompt, value_prompt
from tot.models import gpt
from tot.tasks.crosswords import MiniCrosswordsEnv
import re
import copy
import os
import recorder as record
import datetime
import time
import argparse
import textwrap

'''
since distance between nodes aren't defined, the default will be 10, since average cadidate score is approximately 2 when k = 8
and will make distance = 10 - candidate_score
k isn't a variable that could be fixed since only send to generator, so can't control node count of each level
b is actually "breadth limit" instead of "best", so dfs doesn't require a b value, which is similar to b = 1 in bfs (paper p.8)
env_step is ranged from 0 ~ 9

if dfs doesn't have time limit, it could fully travel the tree, so 運氣不好的話, sd will not improve too much accuracy
but maybe improve efficiency
'''

# timer to calculate cost time
class timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            cost_time = 0
        elif self.end_time is None:
            cost_time = time.time() - self.start_time
        else:
            cost_time = self.end_time - self.start_time
        # cost_time = round(cost_time, 5)
        return cost_time

node_id = 1         # variable to store id for each generated node(including nodes which were pruned), start from 1 since 0 is root
all_nodes = dict()  # dictionary to store all nodes
env = MiniCrosswordsEnv()
os.chdir("C:\EMA\llama2\\text-generation-webui\\tree-of-thought-llm")
d_thres = 10000     # threshold for dfs+sd
best_node_id = 0    # store shortest distance node for dfs+sd

# defining node class
class node():
    root = None

    def __init__(self, node_id: int, action: str, value: float, depth: int, parent_node_id: int):
        self.node_id: int = node_id                         # the node id
        self.action: str = action                           # the candidate's word and position, ex: (h2. motor)
        self.value: float = value                           # score of the candidate
        self.depth: int = depth                             # depth in tree, ranged from 0 ~ 9
        self.parent_node_id: int = parent_node_id           # id of parent node

        self.d_parent: float = 20 - value                   # distance to parent node
        self.d_root: float = 0.0                            # distance to root node
        self.children_node_id: list = []                    # list all child node id
        self.pruned: bool = False                           # True if pruned
        self.visited: bool = False                          # True if already visited

    def to_list(self):
        if self.node_id == 0:
            return {"node_id": self.node_id, "action": self.action, "depth": self.depth, "parent_node_id": None, "children_node_id": self.children_node_id}
        else:
            return {"node_id": self.node_id, "action": self.action, "value": self.value, "depth": self.depth, "parent_node_id": self.parent_node_id, "distance_to_parent": self.d_parent, "distance_to_root": self.d_root, "children_node_id": self.children_node_id, "pruned": self.pruned}

    @classmethod
    def create_node(cls, action: str, value: float, depth: int, parent_node_id: int):
        global node_id, all_nodes
        new_node = cls(node_id, action, value, depth, parent_node_id)
        all_nodes[node_id] = new_node
        
        if parent_node_id:
            all_nodes[parent_node_id].children_node_id.append(new_node.node_id)
        if parent_node_id == 0:
            all_nodes[0].children_node_id.append(new_node.node_id)
        
        all_nodes[node_id].d_root = node.get_distance_root(all_nodes[node_id].node_id)
        node_id += 1
        return all_nodes[node_id - 1]
        # existing_node = cls.find_node_id(cls.root, node_id)
        # if existing_node:
        #     existing_node.action = action
        #     existing_node.value = value
        #     existing_node.depth = depth
        #     existing_node.parent_node_id = parent_node_id
        #     existing_node.d_parent = 10 - value
        #     existing_node.d_root = node.get_distance_root(existing_node.node_id)
        #     existing_node.children_node_id = []
        #     return existing_node
        # else:
        #     new_node = cls(node_id, action, value, depth, parent_node_id)
        #     new_node.d_root = node.get_distance_root(new_node.node_id)
            
        #     node_id += 1
        #     parent = cls.find_node_id(cls.root, parent_node_id)
        #     if parent:
        #         child_list = parent.children_node_id
        #         child_list.append(node_id)
        #     return new_node
    
    @staticmethod
    def find_node_id(self, current_node = None, target_id: int = 0):
        global all_nodes
        for i in range(len(all_nodes)):
            if all_nodes[i].node_id == target_id:
                return all_nodes[i]
        return None
        # if current_node.node_id == target_id:
        #     return current_node
        # if current_node.children_node_id:   # if child node exists
        #     for child in current_node.children_node_id:
        #         child_node = node.find_node_id(, target_id)  # find child node
        #         if child_node:
        #             return child_node
        # return None

    def update_node(target_id: int):
        temp_node = all_nodes[target_id]
        global total_nodes
        if temp_node:
            temp_node.d_parent = 20 - temp_node.value
            temp_node.d_root = node.get_distance_root(target_id, 0.0)
            total_nodes[target_id]['value'] = temp_node.value
            total_nodes[target_id]['distance_to_parent'] = temp_node.d_parent
            total_nodes[target_id]['distance_to_root'] = temp_node.d_root

    def clr_child(target_id: int):
        global all_nodes
        all_nodes[target_id].children_node_id.clear()
        # target = cls.find_node_id(cls.root, target_id)
        # if target:
        #     target.children_node_id = []

    @classmethod
    def clr_tree(cls, current_node=None):
        global all_nodes, total_nodes
        all_nodes.clear()
        if current_node is None:
            current_node = cls.root
        if current_node.children_node_id:
            for child_id in current_node.children_node_id:
                child_node = node.find_node_id(node.root, child_id)
                cls.clr_tree(child_node)
            current_node.children_node_id = []
        if current_node.node_id != 0:  # Don't remove the root node
            del current_node
        root_node = node(action="gRoot", node_id=0, value=0.0, depth=0, parent_node_id=None)
        root_node.visited = True
        all_nodes[0] = root_node
        total_nodes.append(root_node.to_list())

    @staticmethod
    def get_distance_root(node_id: int, current_distance: float = 0.0):
        global all_nodes
        if node_id == 0:
            return current_distance
        else:
            temp_distance = current_distance
            temp_distance += all_nodes[node_id].d_parent
            # temp_node = all_nodes[node_id]
            return node.get_distance_root(all_nodes[all_nodes[node_id].parent_node_id].node_id, temp_distance)
        # target_node = node.find_node_id(node.root, node_id)
        # if target_node:
        #     if node_id == 0:
        #         return current_distance
        #     else:
        #         temp_distance = current_distance
        #         temp_node = node.find_node_id(node.root, node_id)
        #         temp_distance += temp_node.d_parent
        #         return node.get_distance_root(temp_node.parent_node_id, temp_distance)
        # else:
        #     return None
    
    @staticmethod
    def prune_node(node_id: int):
        global all_nodes
        all_nodes[node_id].pruned = True
        if all_nodes[node_id].children_node_id:
            for child_id in all_nodes[node_id].children_node_id:
                node.prune_node(all_nodes[child_id].node_id)
        # target_node = node.find_node_id(node.root, node_id)
        # target_node.pruned = True
        # if target_node.children_node_id:
        #     for child_id in target_node.children_node_id:
        #         node.prune_node(child_id)
        # return None

        
node.root = node(action="gRoot", node_id=0, value=0.0, depth=0, parent_node_id=None)

def prompt_wrap(obs):
    return propose_prompt.format(input=obs)

#TODO: Prompt
def parse_line(input_str):
    # regular expression pattern to match the input string format
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

    # use regex to extract the parts of the input string
    match = re.match(pattern, input_str)

    if match:
        # extract the matched groups
        parts = [match.group(1), match.group(2), match.group(3)]
        return parts
    else:
        return None

confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}

def parse_response(response):
    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

    return parsed_lines if len(parsed_lines) >= 1 else None

def get_candidates_to_scores(env):
    obs = env.render()
    if obs in env.cache: 
        print('cache hit')
        return env.cache[obs]
    print('call gpt')
    responses = gpt(prompt_wrap(obs), model='gpt-3.5-turbo', n=8)   # response 8 times(應該就是k)

    candidates_to_scores = {}
    for response in responses:
        parsed_response = parse_response(response)
        print(f'parsed response = {type(parsed_response)}:\n{parsed_response}')

        # sum up candidates and score from 8 parsed_responses => [(candidate, score), (...)]
        if parsed_response:
            for candidate, score in parsed_response:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score    # add new score
        # choose candiate with highest score
    # print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
    # print(f'\ncandidates_to_scores:\n{candidates_to_scores}')
    env.cache[obs] = candidates_to_scores
    return candidates_to_scores

def propose_score(env, idx):
    obs = env.reset(idx)
    done = False
    infos = []
    while not done:
        responses = gpt(prompt_wrap(obs), model='gpt-3.5-turbo', n=5)
        candidates_to_scores = {}
        for response in responses:
            parsed_response = parse_response(response)
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        # choose candiate with highest score
        print(sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True))
        if len(candidates_to_scores) == 0:
            break
        candidates =  sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
        for candidate in candidates:
            env_ = copy.deepcopy(env)
            env_.step(candidate)
            if not any(_ == 2 for _ in env_.status):
                break
        print(candidate)
        # candidate = input()
        obs, r, done, info = env.step(candidate)
        print(obs)
        print(env.steps, info)
        print('-------------------\n\n\n')
        infos.append(info)
    return infos
#TODO: DFS
answer = None       # string to store current board for answer
best_answer = None  # to store best answer for dfs+sd
accuracy = None     # dictionary to store accuracy => {'r_letter': ###, 'r_word': ###, 'r_game': ###}
best_accuracy = None
tree_json = []      # lsit to store dict in order to write to json file
best_dfs = None     # a variable to store 'best' in dfs
best_step = None    # to store best step for dfs+sd
steps_list = None   # to store steps in a list

def dfs(env, actions, infos, time_limit, max_per_state, idx, prune=False, depth = 1, parent_node_id = 0):
    # print(f'\ncalling dfs with:\nactions: {actions}\ninfos: {infos}\ndepth: {depth}\n')

    temp_node = None

    global d_thres, best_node_id
    # d_thres = 10000         # sphere decoding threshold (distance)
    # best_node_id = 0        # leaf node that has shortest distance to root node (which means the best answer)
    # need_recursive = True   # if use another way to generate leaf node, need_recursive = False

    (f'\n\n\n----------------------------------\n\ncalling dfs with d_thres = {d_thres}\n\n----------------------------------\n\n\n')

    enable_output = False

    global visited_node_id
    # get candidate thoughts
    candidates_to_scores = get_candidates_to_scores(env)
    key_candidates_to_scores = sorted(candidates_to_scores.items(), key=lambda x: x[1], reverse=True)
    if len(candidates_to_scores) == 0: 
        return 0, [], []
    print(f'candidates = \n{key_candidates_to_scores}')

    global node_id
    init_node_id = node_id      # each level's first node_id

    # create node for each candidate, node_id = global node_id ~ global node_id + k - 1 (if k = 8 => node = id + 0 ~ id + 7)
    candidate_count = len(key_candidates_to_scores)
    for i in range(candidate_count):
        new_nodes = node.create_node(key_candidates_to_scores[i][0], 0, depth = depth, parent_node_id = parent_node_id)
        global total_nodes
        total_nodes.append(new_nodes.to_list()) #TODO: #TODO: #TODO: need to update
    print(f'init_node_id = {init_node_id}\nend_node_id = {init_node_id + candidate_count - 1}')

    current_node_id = init_node_id  # in order to know which node is running

    # update depth
    current_depth = depth

    # back up current state
    board, status, steps = env.board.copy(), env.status.copy(), env.steps

    # try each candidate
    cnt_per_state = 0
    dict_candidates_to_scores = sorted(candidates_to_scores, key=candidates_to_scores.get, reverse=True)
    print(f'try candidate order:\n{dict_candidates_to_scores}')

    for action in dict_candidates_to_scores:
        obs, r, done, info = env.step(action)
        r = info['r_word']

        temp_node = all_nodes.get(current_node_id)
        if temp_node:
            print(f'found node id {current_node_id} = {temp_node}')
        
        if args.algorithm == 'dfs':
            condition = len(infos) < time_limit and env.steps < 10 and not any(_ == 2 for _ in env.status)
        elif args.algorithm == 'dfs+sd':
            condition = len(infos) < time_limit and env.steps < 10 and not any(_ == 2 for _ in env.status)

        if condition:  # not violating any existing constraints
            cnt_per_state += 1
            if cnt_per_state > max_per_state: break
            count, score = env.prompt_status_average(str(action))         # TODO: tot.tasks.crosswords 的 prompt_status 可以抓到每一個 node 是 sure, maybe, impossible，要用這個去評分(60, 40, 0.0001)
            actions.append(action)              # best answer decided
            print(f'score = {score}, type = {type(score)}\n')
            if temp_node:
                # all_nodes[current_node_id].value = score
                temp_node.value = score
                node.update_node(temp_node.node_id)
            print(f'node {temp_node.node_id} value is {score}')

            # store current answer
            global answer
            board_str = env.render_board()
            line = board_str.splitlines()[1:]   # strip away "Current Board: "
            answer = "\n".join(line)

            # improve terminal readability
            print(f'len(infos) = {len(infos)}')
            print(f'actions = {actions}')     # a list with position and filled words, ex: actions = ['h2. motor', 'v5. drier', 'h1. agend']
            print(f'env.render_board = {env.render_board()}')
            print(f'info = {info}')
            print(f'count = {count}')
            print(f'depth = {current_depth}')
            print(f'total steps = {len(infos)}')

            # store current accuracy
            global accuracy
            accuracy = info

            info = {'total_step': len(infos), 'env_step': env.steps, 'actions': actions.copy(), 'info': info, 'count': count}

            # update d_thres and best_node_id
            if args.algorithm == 'dfs+sd' and depth == 9:       # step limit should be big enough so tree could reach depth = 9
                
                if temp_node is not None:
                    temp_distance = temp_node.d_root
                else: temp_distance = 1000                      # key not found

                if temp_distance < d_thres:
                    print(f'\n\n\n----------------------------------\n\nfound best answer\n\n----------------------------------\n\n\n')
                    global sd_time
                    my_timer.stop()
                    sd_time += my_timer.elapsed_time()

                    enable_output = True

                    d_thres = temp_distance
                    # update best status
                    best_node_id = current_node_id
                    global best_step
                    best_step = info['total_step']

                    global best_answer
                    best_answer = answer

                    global best_accuracy
                    best_accuracy = accuracy
                    print(f'best node id: {best_node_id}\ndistance to root: {temp_distance}\nbest accuracy: {best_accuracy}\nbest answer: \n{best_answer}\n\n')
                    my_timer.start()

            
            info['current_node_id'] = current_node_id
            if args.algorithm == 'dfs+sd':
                if depth == 9: 
                    info['distance to root'] = temp_distance
            elif args.algorithm == 'dfs':
                if temp_node:
                    info['distance to root'] = temp_node.d_root
                else:
                    info['distance to root'] = 10000    # node not found
            info['depth'] = current_depth
            info['d_thres'] = d_thres
            info['answer'] = answer

            # recording
            if args.algorithm == 'dfs': 
                record.Record_txt(record.record_file_name, str(info) + '\n\n', idx = idx)   # straight record
            elif args.algorithm == 'dfs+sd': 
                global steps_list
                steps_list.append(info)

            if enable_output:
                record.Record_txt(record.record_file_name, steps_list, idx = idx)
                steps_list.clear()
                enable_output = False
            
            global tree_json
            tree_json.append(info)

            infos.append(info)  # original design

            global best_dfs
            # find the best 'r_word' in infos, best會有'info'跟'count', 'info'是目前的accuracy, count是目前有幾個sure, maybe, impossible
            if infos:
                best = max(infos, key=lambda x: x['info']['r_word'])    # 有點灌水 = 是跑完100次之後才取最大的 並不是DFS的輸出
                print('best', best)
                best_dfs = best
            print('--------------')
            print()

            if not prune or count['impossible'] < 1:  # only continue if the current status is possible
                print(f'\n\n\n----------------------------------\n\ncontinue\n\n----------------------------------\n\n\n')
                visited_node_id.append(current_node_id)
                # visit_checker = True
                dfs(env, actions, infos, time_limit, max_per_state, idx, prune, current_depth + 1, current_node_id)
            actions.pop()
            print(f'\nactions after pop:\n{actions}')

        current_node_id += 1
        env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
    if visited_node_id:
        current_node_id = visited_node_id[-1] 
        print(f'\nvisited_node_id = {visited_node_id}\ncurrent_node_id = {current_node_id}')
        visited_node_id.pop()
    

visited_node_id = []
total_nodes = []
my_timer = timer()
sd_time = None

# TODO: main
def run(args):
    infoss = []
    total_json = [] # a list to store dict of each game in order to write into all_json_file
    total_json_result = []
    global tree_json
    node.clr_tree()
    current_time = time.localtime()  # get current time
    formatted_time = time.strftime("%H:%M:%S", current_time)  # convert time into hh:mm:ss

    if args.enable_pruning == 1: enable_pruning = True
    else: enable_pruning = False

    # acc_info_crossword.txt initial string, textwrap.dedent will strip all spaces before every line
    acc_init_str = textwrap.dedent(f''' 
    model: {args.backend}
    temperature: {args.temperature}
    algorithm: {args.algorithm}
    start_index = {args.task_start_index}
    end_index = {args.task_end_index}
    pruning = {enable_pruning}
    date: {datetime.date.today()}
    time: {formatted_time}
    ''').strip()    # strip '\n' and spaces before and after the string

    # create total recording file
    record.Init_record_file(record.acc_file_name, f'{acc_init_str}\n')
    record.Init_record_file(record.all_json_file_name, initial_string = '')
    record.Init_record_file(record.json_result_file_name, initial_string = '')

    # in order to meet conndition of environment (i in range(0, 100, 5)), amplify indices by 5 times
    start_idx  = args.task_start_index * 5
    end_idx  = args.task_end_index * 5

    # variable to calculate average 'r_letter', 'r_word', and number of games that is all correct
    acc_letter = 0.0
    acc_word = 0.0
    completed_game = 0

    # variable to sum up total cost time
    total_cost_time = 0
    
    # solving
    for i in range(start_idx, end_idx, 5):
        env.reset(i)
        infos = []
        actions = []
        tree_json = []  # reset tree_json so the latter json file will not include the previous one
        idx = i//5  # a variable for file naming
        global total_nodes, all_nodes
        total_nodes.clear()

        # TODO: tree_section
        global node_id
        node_id = 1     # reset tree
        # clear tree
        node.clr_tree()

        global steps_list
        steps_list = []

        # record_crossword_{idx}.txt initial string 
        record_init_str = textwrap.dedent(f'''
        model: {args.backend}
        temperature: {args.temperature}
        algorithm: {args.algorithm}
        idx: {idx}
        pruning = {enable_pruning}
        date: {datetime.date.today()}
        time: {formatted_time}
        ''').strip()

        # create recording file
        record.Init_record_file(record.record_file_name, f'{record_init_str}\n\n', idx = idx)
        record.Init_record_file(record.tree_json_file_name, initial_string = '', idx = idx)
        record.Init_record_file(record.node_json_file_name, initial_string = '', idx = idx)

        # calling dfs
        global visited_node_id
        visited_node_id.clear()

        # reset
        global d_thres, best_node_id
        d_thres = 10000
        best_node_id = 0
        
        # create timer
        global my_timer
        global sd_time
        sd_time = 0.0

        # start timer
        my_timer.start()

        if args.enable_pruning == 1: 
            dfs(env, actions, infos, args.step_limit, max_per_state=3, idx = idx, prune=True) # pruning
        else: 
            dfs(env, actions, infos, args.step_limit, max_per_state=3, idx = idx, prune=False) # without pruning

        # stop timer
        my_timer.stop()

        # storing cost time
        if args.algorithm == 'dfs': cost_time = my_timer.elapsed_time()
        elif args.algorithm == 'dfs+sd': cost_time = sd_time
        total_cost_time += cost_time

        # recieving answer
        global answer, best_answer, accuracy, best_accuracy, best_dfs, best_step
        if args.algorithm == 'dfs':
            answer = best_dfs['answer']
            accuracy = best_dfs['info']
            best_node_id = best_dfs['current_node_id']
            node_depth = best_dfs['depth']
            chosen_step = best_dfs['total_step']
            distance_root = best_dfs['distance to root']
        elif args.algorithm == 'dfs+sd':
            answer = best_answer
            accuracy = best_accuracy
            chosen_step = best_step
            distance_root = all_nodes[best_node_id].d_root
            node_depth = all_nodes[best_node_id].depth

        # print(all_nodes)
        # print('\n\n\n')
        print('\n-----------------------------------------------------\n')

        print(f'\nbest node id: {best_node_id}')
        print(f'chosen step: {chosen_step}')
        print(f'distance to root: {distance_root}')
        print(f'\nanswer:\n{answer}')
        print(f'accuracy: {accuracy}')


        # collecting accuracy
        acc_letter += float(accuracy['r_letter'])
        acc_word += float(accuracy['r_word'])
        if bool(accuracy['r_game']): completed_game += 1

        # writing into acc_file.txt
        record.Record_txt(record.acc_file_name, f'\nidx: {idx}\nbest node id: {best_node_id}\nnode depth: {node_depth}\nchosen step: {chosen_step}\ndistance to root: {distance_root}\nanswer:\n{answer}\n\naccuracy: {accuracy}\ncost time: {cost_time} s\n--------------------------\n')
        
        # append to total_json list which is later written to all_json_file
        total_json.append(tree_json)
        
        # writing final answer into record_file
        if args.algorithm == 'dfs': record.Record_txt(record.record_file_name, f'-----------------------\n\nbest node id: {best_node_id}\nnode depth: {node_depth}\nchosen step: {chosen_step}\ndistance to root: {distance_root}\nanswer:\n\n{answer}\n\naccuracy: {accuracy}\ncost time: {cost_time} s\n\n---- Task Complete ----', idx = idx)
        elif args.algorithm == 'dfs+sd': record.Record_txt(record.record_file_name, f'-----------------------\n\nbest node id: {best_node_id}\nnode depth: {node_depth}\nchosen step: {chosen_step}\ndistance to root: {distance_root}\nanswer:\n\n{answer}\n\naccuracy: {accuracy}\ncost time: {cost_time} s\n\n---- Task Complete ----', idx = idx)
        json_result = {
            "index": idx,
            "best node id": best_node_id,
            "node depth": node_depth,
            "chosen step": chosen_step,
            "distance to root": distance_root,
            "answer": answer,
            "accuracy": accuracy,
            "cost time": cost_time
        }
        total_json_result.append(json_result)
        tree_json.append(json_result)
        record.Record_json(record.tree_json_file_name, tree_json, idx = idx)
        # print(f'\ntype: {type(total_nodes)}\n\ntotal_nodes: {total_nodes}\n')
        record.Record_json(record.node_json_file_name, total_nodes, idx = idx)

        # original design
        infoss.append(infos)
        if args.enable_pruning:
            with open('logs/crosswords/infoss_dfs_prune.json', 'w') as fout:
                json.dump(infoss, fout)
        else:
            with open('logs/crosswords/infoss_dfs_no_prune.json', 'w') as fout:
                json.dump(infoss, fout)

    # write to all_json_file
    record.Record_json(record.all_json_file_name, total_json)
    # write all results to json file
    record.Record_json(record.json_result_file_name, total_json_result)

    # calculating average accuracy
    game_count = args.task_end_index - args.task_start_index
    acc_letter /= game_count
    acc_word /= game_count
    acc_letter = "{:.3f}".format(acc_letter)
    acc_word = "{:.3f}".format(acc_word)
    record.Record_txt(record.acc_file_name, f'\naverage accuracy: {{\'r_letter\': {acc_letter}, \'r_word\': {acc_word}, \'r_game\': {completed_game}}}\naverage cost time: {total_cost_time / game_count}\ntotal cost time: {total_cost_time} s\n\n---- Record Complete ----\n')

# add arguments
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo'], default='gpt-3.5-turbo')   # for cost calculation use
    args.add_argument('--temperature', type=float, default=0.7) # change to modify randomness

    args.add_argument('--task_start_index', type=int, default = 0)  # range from 0 ~ 20
    args.add_argument('--task_end_index', type=int, default  =20)   # range from 0 ~ 20
    args.add_argument('--algorithm', type=str, default='dfs')   # (dfs, dfs+sd)
    args.add_argument('--enable_pruning', type=int, default = 1)   # whether pruning is allowed, if allowed, 1
    args.add_argument('--step_limit', type=int, default = 100)  # when step count reaches the limit, the game is terminated

    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(f'\n{args}\n')
    print(prompt_wrap(env.reset(0)))
    run(args)

    """
    1. 加上prune
    2. 想辦法將current board的變化畫成圖片，參考林哲丞的crosswords/crossword_draw.py)

    由於作者的演算法會先將candidates進行排序，導致有較高value的candidate在樹的左邊，故進行dfs時不像一般的搜尋演算法來得隨機，能夠在有限的step中找到最佳解
    sd會將距離>d_thres的node直接prune，但由於value高(即distance較短)的node集中在樹的左邊，所以sd能夠縮短的時間並不多
    且正確率提高也有限
    """