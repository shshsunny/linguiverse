from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime, timedelta
import time
from openai import OpenAI
from copy import deepcopy
from pprint import pprint
from utils import *
from contextlib import nullcontext

# 虚拟世界存档管理器。
# 分为两个主要功能块，
# 一是世界更新（根据最近对话更新世界状态）：
# 输入：
# - 会话上下文，形式为[..., U, S, U, S]，即以最近一次系统回复结尾
# - 当前虚拟世界存档
# 输出：
# - 更新后的虚拟世界存档
#
# 二是世界查询（根据当前世界状态和对话内容查询最相关的局部信息），这部分集成在回应提示器中。
embed_model = None
def init_world_processor():
    global embed_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


def world_processor(context, responser, world, verbose=False, archive_lock=nullcontext()):
    # 注意：本函数会对world就地修改！

    # context的格式：[
    #   {'role': 'system', 'content': '...'}, 
    #   {'role': 'user', 'content': '...'}, 
    #   {'role': 'assistant', 'content': '...'}
    #   ...,
    #   {'role': 'user', 'content': '...'}, 
    #   {'role': 'assistant', 'content': '...'}
    # ]

    # TODO: 完整的对话中每轮assistant回复之前会加上system prompt，之后应当检查确保context中相关信息的提取仍然正确！

    # pipeline:
    # 1. 让模型将对话新增内容转为第三人称简单陈述句，一行一个，记为new_log，分行添加到world.dynamic中。
    # 2. 用语义embedding计算new_log与world.static中每个条目的相似度，将最相似的条目（记作relevants）提取出来（理论上这部分数据量应当远小于world的总大小）。
    # 3. 让模型逐个判断relevants每个条目基于new_facts的操作（不改动 / 修改 / 删除），逐行输出结果；
    #    并且判断relevants是否没有包含new_facts的信息，因此额外加若干新的static条目。
    # 4. 将输出结果整合到world_static中。

    
    if verbose:
        print("world before:")
        pprint(world)

    ### Step 1: 从对话中提取新事实 ###
    # 获取最后一轮完整对话（用户+系统）
    with archive_lock:
        turn_text = format_attended_context(context, world, att_len=2)

    prompt_extract = (
        "You are an assistant that extracts facts from dialogues. "
        "Each user input is a dialogue. "
        "You must consider from the perspective of every entity in the dialogue, "
        "and infer as comprehensive facts that are mentioned or implied in it as possible. "
        "Declare each fact with a third-person declarative sentence which is as simple and clear as possible. "
        "Output one sentence per line."
    )
    messages = [
        {'role': 'system', 'content': prompt_extract},
        {'role': 'user', 'content': turn_text}
    ]

    client = responser['client']
    model = responser['model']

    #stime = time.time()
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            max_tokens=1024,
    )
    #etime = time.time()
    #clen = len_context(messages)

    output_text = response.choices[0].message.content

    new_facts = [line.strip() for line in output_text.split("\n") if line.strip()]

    #print(f"[DEBUG] world_processor stage 1 :{etime - stime:.3f} seconds, clen: {clen}")


    if verbose: print("new_facts:", new_facts)

    #stime = time.time()

    now = time.time()
    # 将 new_facts 添加为 dynamic 条目（带时间）

    with archive_lock:
        for i, fact in enumerate(new_facts):
            world["dynamic"].append((fact, now))
        static_texts = [entry[0] for entry in world["static"]]

    ### Step 2: 相似度匹配，找到可能需要更新的 static 条目 ###
    static_embeddings = embed_model.encode(static_texts, convert_to_tensor=True)
    fact_embeddings = embed_model.encode(new_facts, convert_to_tensor=True)

    # NOTE: 修改评分机制：不再用RELEVANCE_THRESHOLD来筛选，而是根据综合评分排序所有static_indices
    # 选出其中前k个最相关的static_indices，以提高效率、减少开销
    

    
    TOP_RELEVANCE_RANK = 10
    cosine_scores = util.cos_sim(fact_embeddings, static_embeddings)  # 形状: [num_new_facts, num_static]
    max_scores_per_static = cosine_scores.max(dim=0).values

    top_k_indices = torch.topk(max_scores_per_static, k=min(TOP_RELEVANCE_RANK, len(static_texts))).indices
    relevant_static_indices = set(top_k_indices.cpu().numpy())

    # 构建唯一相关的 static 条目集合
    relevant_static = [] # 列表，形如[(index, sentence), ...]
    for static_idx in sorted(relevant_static_indices):
        relevant_static.append((static_idx, static_texts[static_idx]))
    
    #etime = time.time()

    #print(f"[DEBUG] world_processor stage 2 :{etime - stime:.3f} seconds")

    if verbose: print("relevant_static:", relevant_static)

    ### Step 3: 逐条判断如何修改 static ###

    
    newline = '\n'
    prompt_update = (
        "Your task is to update a given knowledge base according to user inputs. "
        "The knowledge base is comprised of several lines, each being a simple sentence, "
        "and you must output one line corresponding to it, "
        "containing one of the following: "
        "1. a sentence which is the modified version of the knowledge base sentence, and is a third-person declarative sentence, and as simple and clear as possible; "
        "2. 'd', meaning the knowledge base sentence is deleted because it's out-dated or contradictory to the input, or " 
        "3. 'k', meaning the knowledge base sentence is kept unchanged. "
        "If the information in the input is new to the knowledge base, "
        "you must append additional lines each containing a sentence of the same style as choice 1, "
        "so that the information is fully updated to the knowledge base. "
        f"\nKnowledge base:\n{newline.join([pair[1] for pair in relevant_static])}"
    )
    messages = [
        {'role': 'system', 'content': prompt_update},
        {'role': 'user', 'content': '\n'.join(new_facts)}
    ]
    # pprint(messages)

    #stime = time.time()
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
    )
    #clen = len_context(messages)
    #etime = time.time()

    output_text = response.choices[0].message.content
    updates = [line.strip() for line in output_text.split("\n") if line.strip()]
    delete_idxs = []

    with archive_lock:
        for i, line in enumerate(updates):
            if i < len(relevant_static):
                if line == 'd': # 记录待删除索引
                    delete_idxs.append(i)
                elif line == 'k': # 保持不变
                    pass
                else: # 修改记录
                    world['static'][relevant_static[i][0]] = (line, now)
            else: # 添加新条目
                world['static'].append((line, now))
        for idx in sorted(delete_idxs, reverse=True):  # 从后往前删除，避免索引错乱
            del world['static'][relevant_static[idx][0]]
    
        if verbose: 
            print("world after:")
            pprint(world)

    
    #print(f"[DEBUG] world_processor stage 3 :{etime - stime:.3f} seconds, clen: {clen}")

    return world



