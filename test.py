# 方便测试用的一些utility。
# 把测试用例单独编写成文件，用此模块提供的函数来load。

import time
import json
import re
import pandas as pd
from pprint import pprint
import torch

from utils import *

def parse_test(text: str):
    lines = text.strip().split('\n')
    section = None
    now = time.time()

    world = {
        'meta': {
            'system_role': '',
            'user_role': '',
            'last_access_time': now,
            'preferences': {
                'target_language': 'en',
                'source_language': 'zh',  # 添加source_language键并设置默认值
                'world_genre': [],
                'world_prologue': ''
            }
        },
        'static': [],
        'dynamic': [],
    }

    dynamic_sentences = []
    dynamic_times = []

    scene_uid = None
    skill_rows = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            if 'meta' in line.lower():
                section = 'meta'
            elif 'static' in line.lower():
                section = 'static'
            elif 'dynamic' in line.lower():
                section = 'dynamic'
            elif 'scene' in line.lower():
                section = 'scene'
            continue

        if section == 'meta':
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ['system_role', 'user_role']:
                    world['meta'][key] = value
                elif key in ['target_language', 'source_language', 'world_genre', 'world_prologue']:  # 添加source_language
                    if key == 'world_genre':
                        world['meta']['preferences'][key] = [g.strip() for g in value.split(',')]
                    else:
                        world['meta']['preferences'][key] = value

        elif section in ['static', 'dynamic']:
            match = re.match(r'\[(-?\d+)\]\s*(.+)', line)
            if match:
                delta_minutes = int(match.group(1))
                sentence = match.group(2).strip()
                timestamp = now + delta_minutes * 60
            else:
                sentence = line
                timestamp = None  # to be filled later

            if section == 'static':
                if timestamp is None:
                    timestamp = now
                world['static'].append((sentence, timestamp))
            else:
                dynamic_sentences.append(sentence)
                dynamic_times.append(timestamp)

        elif section == 'scene':
            if scene_uid is None:
                scene_uid = int(line)
            else:
                parts = [p.strip() for p in line.split('|')]
                while len(parts) < 3:
                    parts.append('')
                good, bad, prof = parts
                row = {
                    'good_cases': good,
                    'bad_cases': bad,
                    'proficiency': float(prof) if prof else 0.0
                }
                skill_rows.append(row)

    # Fill in missing timestamps for dynamic
    if any(t is not None for t in dynamic_times):
        known = [(i, t) for i, t in enumerate(dynamic_times) if t is not None]
        for (i1, t1), (i2, t2) in zip(known, known[1:]):
            gap = i2 - i1
            if gap > 1:
                step = (t2 - t1) / gap
                for j in range(1, gap):
                    dynamic_times[i1 + j] = t1 + step * j
        first_known = known[0]
        for i in range(first_known[0] - 1, -1, -1):
            dynamic_times[i] = dynamic_times[i + 1] - 600
        last_known = known[-1]
        for i in range(last_known[0] + 1, len(dynamic_times)):
            dynamic_times[i] = dynamic_times[i - 1] + 600
    else:
        base = now - 600 * len(dynamic_sentences)
        for i in range(len(dynamic_sentences)):
            dynamic_times[i] = base + 600 * i

    for sentence, ts in zip(dynamic_sentences, dynamic_times):
        world['dynamic'].append((sentence, ts))

    # Build skill dataframe
    skill_df = pd.DataFrame(skill_rows)
    if not skill_df.empty:
        skill_df.insert(0, 'table', '')  # Default empty, can be filled later

    return world, skill_df, scene_uid

def load_test(worldname='school.txt', root='_tests'):
    filepath = f"{root}/{worldname}"
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return parse_test(text)

def get_test(test='school.txt', language='english', 
             metadata_root='metadata', _test_root='_tests', load_vocab_emb=False, verbose=False):
    world, skill_df, scene_uid = load_test(test)
    # print("scene_uid:", scene_uid)
    progress = {
        'current_scene': scene_uid,
    }
    # print(world, skill_df, scene_uid, sep='\n')

    metadata_path = f"{metadata_root}/{language}"

    vocab = pd.read_csv(f"{metadata_path}/vocab.csv")
    grammar = pd.read_csv(f"{metadata_path}/grammar.csv")
    scenarios = pd.read_csv(f"{metadata_path}/scenarios.csv")
    vocab_emb = torch.load(f"{metadata_path}/vocab_emb.pt") if load_vocab_emb else None
    
    scene_vocab = vocab[vocab["scene_uid"] == scene_uid]
    scene_grammar = grammar[grammar["scene_uid"] == scene_uid]

    # 为当前条目创建skills档案（目前仅限于当前场景的词汇和语法，没有其他场景的内容）
    new_records = []
    for _, row in scene_vocab.iterrows():
        uid = row["uid"]
        #if not ((skill["table"] == "vocab") & (skill["uid"] == uid)).any():
        new_records.append({
            "table": "vocab",
            "uid": uid,
            "good_cases": "",
            "bad_cases": "",
            "proficiency": 0.0,
            "proficiency_goal": INITIAL_PROFICIENCY_GOAL
        })

    for _, row in scene_grammar.iterrows():
        uid = row["uid"]
        #if not ((self.skill["table"] == "grammar") & (self.skill["uid"] == uid)).any():
        new_records.append({
            "table": "grammar",
            "uid": uid,
            "good_cases": "",
            "bad_cases": "",
            "proficiency": 0.0,
            "proficiency_goal": INITIAL_PROFICIENCY_GOAL
        })

    # 添加到 skill 表中
    assert new_records, "No new records to add to skill table."
    skill = pd.DataFrame(new_records)
    # 修改skill表中与testcase对应部分的掌握程度
    n = min(len(skill), len(skill_df))
    if n > 0:

        # 按照行号索引逐列更新（只更新存在的字段）
        for col in ['good_cases', 'bad_cases', 'proficiency']:
            if col in skill_df.columns:
                skill.loc[:n-1, col] = skill_df.loc[:n-1, col].values
    
    if verbose:
        print("\n[info] scene_vocab:")
        print(scene_vocab)          
        print("\n[info] scene_grammar:")
        print(scene_grammar)
        print("\n[info] skill: (records loaded from your testcase)")
        print(skill)
        print("\n[info] world:")
        pprint(world, indent=2, width=80)

    return {
        'scenarios': scenarios,     # metadata 的 场景数据库
        'vocab': vocab,             # metadata 的 词汇数据库
        'grammar': grammar,         # metadata 的 语法数据库
        'vocab_emb': vocab_emb,     # metadata 的 词汇embedding

        'world': world,             # 当前世界，包含dynamic，meta，static
        'skill': skill,             # 当前熟练度对应表
        'progress': progress,       # 当前场景uid
    }

if __name__ == "__main__":
    # "school.txt","restaurant.txt","weather.txt","family.txt",
    testlist = ["apologize.txt"]
    language = 'english'
    for test in testlist:
        print(f"========================= {test} =========================")
        result = get_test(test=test, language=language, verbose=True)       
        