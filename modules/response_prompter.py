#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime, timedelta
import time
from openai import OpenAI
from copy import deepcopy
from pprint import pprint
from utils import *
import numpy as np
import pandas as pd
from contextlib import nullcontext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_model = None
def init_response_prompter():
    """初始化响应提示器模块"""
    global embed_model
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

def choose_next_to_practise(assistant_context, current_vocab, current_grammar, current_skill):
    current_turn = len(assistant_context)
    # 只考虑还未达到掌握度目标的知识点
    current_skill = current_skill[current_skill['proficiency'] < current_skill['proficiency_goal']].copy()
    if not current_skill.shape[0]:
        return None
    current_skill['last_practice_turn'] = 0
    for idx, turn in enumerate(assistant_context):
        # 默认assistant回应有info字段
        if turn['info']['type'] not in ('demo', 'quiz'):
            continue
        table = turn['info']['table']
        uid = turn['info']['uid']
        current_skill.loc[
            (current_skill['table'] == table) 
            & (current_skill['uid'] == uid), 
            'last_practice_turn'] = idx + 1
            
    current_skill['difficulty'] = 1.0 - current_skill['proficiency']  # 掌握度越低难度越高
    current_skill['freshness'] = (current_turn - current_skill['last_practice_turn']).clip(lower=0)

    # 归一化（注意防止除零）
    current_skill['fresh_norm'] = current_skill['freshness'] / max(current_skill['freshness'].max(), 1)
    current_skill['diff_norm'] = current_skill['difficulty'] / max(current_skill['difficulty'].max(), 0.01)
    # 综合得分
    ALPHA = 0.75
    current_skill['score'] = ALPHA * current_skill['diff_norm'] + (1 - ALPHA) * current_skill['fresh_norm']
    # 按分数降序排列
    current_skill = current_skill.sort_values('score', ascending=False)
    top1 = current_skill.iloc[0]
    return {'table': top1['table'], 'uid': top1['uid']}

def schedule_next_proficiency_goal(current_proficiency, current_goal):
    # 根据知识点已有的掌握度和当前目标，规划下一个目标（仅在目标未达到时有规划作用）
    if current_proficiency < current_goal:
        next_goal = current_goal
        # 后续可以考虑知识点跳过机制，调低过高的目标，使过难知识点被轮换掉，不影响整体进度
    else:
        if current_goal >= FINAL_PROFICIENCY_GOAL:
            next_goal = FINAL_PROFICIENCY_GOAL
        else:
            alpha = 0.1 # 目标上调系数
            next_goal = min(current_proficiency + 
                            max(0.05, alpha * (FINAL_PROFICIENCY_GOAL - current_proficiency))
                                , FINAL_PROFICIENCY_GOAL) # 逼近最终目标时减小目标升幅
    return next_goal




def response_prompter(context, responser, vocab, grammar, vocab_emb, world, skill, progress, archive_lock=nullcontext()):
    
    # context的格式：[
    #   {'role': 'system', 'content': '...'}, 
    #   {'role': 'user', 'content': '...'}, 
    #   {'role': 'assistant', 'content': '...', 'info': {'table': 'vocab / grammar', 'uid': xxx, 'type': 'demo / quiz'}}
    #   ...,
    #   {'role': 'user', 'content': '...'}, 
    # ] 
    # 系统的最新回复尚未生成，以最新用户输入结尾；
    # 系统回复条目有info字段，表示建议针对的知识点和该句的类型（目标句demo / 考验句quiz）

    # attended_len是要聚焦的最近对话条数，用于提取world.static中的相关局部信息，并参与续写建议的生成。

    assert vocab_emb != None, "vocab_emb must be provided for response prompter to work."
    """
    输出：
    - 要聚焦的知识点（含uid及具体建议文本）
    - 语义建议（接下来的续写内容）
    - 难度建议（用词白名单/推荐）
    """

    # pipeline:
    # 1. 计算context和world.static的语义embedding相似度，从而提取出相关的局部信息
    #  （context中每句的相关度随着距当前的轮次指数折扣）
    # 2. 根据skill, vocab, grammar选择接下来要针对的知识点（可以先简单根据掌握度+最近一次学习的轮次，非LLM实现），
    #   提供其元数据信息，并结合context建议句子类型为目标句/考验句
    # 3. 根据以上信息生成续写建议：续写内容+更具体的句式（目标句——陈述？感叹？疑问？考验句——祈使？）
    # 4. 进一步生成用词建议：计算续写建议与vocab, grammar中相关词汇/语法形式的语义相似度（由于vocab表很大，有可能要在DataManager中预处理降低计算量），
    #   同时考虑skill中对应知识的掌握度，来选出最适合用于回复的词汇和语法
    # 最终返回续写建议和知识运用建议
    # 测试时需要通过某种机制生成一个模拟的skill，体现不同难度、话题词汇的掌握度差别，确认本模块能基于用户掌握度给出用词建议

    print('[INFO] ------------ in response_prompter ------------')
    ### Step 1: 提取world相关信息

    HISTORY_ATTENTION_LEN = 8 # 从最近最多x条对话中提取相关信息（用户输入及系统回复；不包括prompt的部分）

    with archive_lock:
        pure_context = get_context_without_prompt(context)
        role_name = {
            'assistant': world['meta']['system_role'],
            'user': world['meta']['user_role']
        }

        attended_history = [role_name[item['role']] + ": " + item['content'] for item in pure_context[-HISTORY_ATTENTION_LEN:]] # 最多提取x条
        static_texts = [entry[0] for entry in world["static"]]


    static_embeddings = embed_model.encode(static_texts, convert_to_tensor=True)
    history_embeddings = embed_model.encode(attended_history, convert_to_tensor=True)

    
    gamma = 0.3 ** (1 / HISTORY_ATTENTION_LEN) # 被关注的最早的对话大约只有30%重要度
    # denom = (1 - GAMMA ** len(attended_history)) / (1 - GAMMA)
    """
    RELEVANCE_THRESHOLD = 0.4 # 按近重远轻的原则对每个static_text的相关性取加权平均，超过此阈值则被注意到

    cosine_scores = torch.zeros(len(static_texts), device=device)

    for i, history_emb in enumerate(history_embeddings):
        cosine_scores_i = util.cos_sim(history_emb, static_embeddings)[0] * gamma ** i
        cosine_scores = torch.where(cosine_scores > cosine_scores_i, cosine_scores, cosine_scores_i)
    
    relevant_static = []
    for idx, score in enumerate(cosine_scores):
        if score.item() >= RELEVANCE_THRESHOLD:
            relevant_static.append((idx, static_texts[idx]))"""
    
    TOP_RELEVANCE_RANK = 10

    cosine_sim_matrix = util.cos_sim(history_embeddings, static_embeddings)  # [len_history, num_static]
    weights = torch.tensor(
            [gamma ** i for i in range(len(attended_history))],
            device=device
        ).unsqueeze(1) # 列向量，为不同远近的历史消息加权
    
    weighted_sim = cosine_sim_matrix * weights
    max_scores = weighted_sim.max(dim=0).values  # 每个static_text的最大加权相似度
    top_k_indices = torch.topk(max_scores, k=min(TOP_RELEVANCE_RANK, len(max_scores)), dim=0).indices
    relevant_static = [
        (idx, static_texts[idx]) for idx in top_k_indices.cpu().numpy()
    ]

    
    # debugging
    # print("relevance cosine_scores:", cosine_scores)
    print("extracted relevant static sentences:", relevant_static)

    ### Step 2: 决定回应句型和知识点
    # 2.1 整理 skill 中每个知识点的最新练习轮次
    # 选出当前场景的词汇和语法点
    current_vocab = vocab[vocab['scene_uid'] == progress['current_scene']].copy()
    current_grammar = grammar[grammar['scene_uid'] == progress['current_scene']].copy()

    # 得到对应的掌握度信息
    current_skill = skill[
        ((skill['table'] == 'vocab') & (skill['uid'].isin(current_vocab['uid'])
        )) | ((
        (skill['table'] == 'grammar') & (skill['uid'].isin(current_grammar['uid']))
        ))].copy()
    
    # 回应提示策略：
    # - 每隔若干轮允许生成一次任意句
    # - 若尚无目标句，则据综合评分选择知识点，生成第一个目标句
    # - 若有最近一次目标句，则持续针对该知识点生成考验句，直到达到其目前的掌握度目标
    # - 达到掌握度目标后，结束当前知识点的学习环节。如果掌握度目标还没达到满意值（最大值），则根据规划器调高掌握度目标
    # - 选择下一个知识点（尚未达到掌握度目标、综合评分最高），生成新的目标句
    # - 若所有知识点均已达掌握度目标，输出type=hooray的句子，表示本场景已完成，可进入下一个场景，剩下句子只生成任意句

    assistant_context = get_context_assistant_only(context)

    ANY_EVERY = 3 # 每多少非任意句之后允许生成一次任意句
    last_any = 0
    for item in reversed(assistant_context):
        if item['info']['type'] != 'any':
            last_any += 1
        else:
            break

    target_info = None # 接下来要学的知识点信息
    # if last_any >= ANY_EVERY:
    #    sentence_type = 'any'
    # else:

    # 更新：取消了any句型，由main agent自行决定是否让demo/quiz句生效
    last_demo_idx = None
    for i in reversed(range(len(assistant_context))):
        if assistant_context[i]['info']['type'] == 'demo':
            last_demo_idx = i
            break

    if last_demo_idx is None:
        target_info = choose_next_to_practise(assistant_context, current_vocab, current_grammar, current_skill)
        sentence_type = 'demo' if target_info != None else 'hooray'
    else:
        target_info = assistant_context[last_demo_idx]['info']
        table, uid = target_info['table'], target_info['uid']
        target = current_skill[(current_skill['table'] == table) & (current_skill['uid'] == uid)].iloc[0]
        if target['proficiency'] < target['proficiency_goal']:
            sentence_type = 'quiz'
        else: # 重新规划当前知识点的proficiency_goal，并重新选择知识点
            # NOTE: 注意，所有更新必须应用到原始存档skill中！current_skill是当前skill的一个局部（当前场景）副本
            # NOTE: 注意使用正确的方式更新dataframe！
            skill.loc[(skill['table'] == table) & (skill['uid'] == uid), 'proficiency_goal'] \
                = schedule_next_proficiency_goal(target['proficiency'], target['proficiency_goal'])
            target_info = choose_next_to_practise(assistant_context, current_vocab, current_grammar, current_skill)
            sentence_type = 'demo' if target_info != None else 'hooray'
    
    # debugging
    print('sentence_type:', sentence_type)
    if sentence_type not in ('any', 'hooray') and target_info is not None:
        print('target_info:', target_info)
        table, uid = target_info['table'], target_info['uid']
        print('target:', current_skill[(current_skill['table'] == table) & (current_skill['uid'] == uid)].iloc[0])

    ### Step 3: 生成续写建议

    client = responser['client']
    model = responser['model']

    if sentence_type == 'any':
        task = 'be coherent. '
        knowledge = ''
        k_type = ''
    elif sentence_type == 'hooray':
        task = f'praise {role_name["user"]} for achieving their language goal in the current scene. '
        knowledge = ''
        k_type = ''
    else:
        k_type = 'expression' if target_info['table'] == 'vocab' else 'grammar point'
        if target_info['table'] == 'vocab':
            target = current_vocab[current_vocab['uid'] == target_info['uid']].iloc[0]
            desc = (
                f"expression: {target['word']}\n"
                +(f"guideword: {target['guideword']}\n" if target['guideword'] and pd.notna(target['guideword']) else '')
                +(f"part of speech: {target['type']}\n" if target['type'] and pd.notna(target['type']) else '')
            )
        else:
            target = current_grammar[current_grammar['uid'] == target_info['uid']].iloc[0]
            desc = (
                f"can-do standard: {target['grammar']}\n"
                +(f"guideword: {target['guideword']}\n" if target['guideword'] and pd.notna(target['guideword']) else '')
                +(f"category: {target['type']}\n" if target['type'] and pd.notna(target['type']) else '')
            )
        
        if sentence_type == 'demo':
            task = 'demonstrate the usage of the following '
            knowledge = f'{desc}\n'
        else: # 'quiz'
            task = f'pose a question or request that lures {role_name["user"]} into using the following '
            knowledge = f'{desc}\n'


    prompt_content = (
        f"{role_name['assistant']} is a helpful language mentor, who is guiding {role_name['user']} by conversation. "
        f"You'll see their conversation history and background information in the input. "
        f"Your task is to advise {role_name['assistant']} to make a response, which "
        f"is creative, based on the history and the background, and should {task} {k_type}\n{knowledge}"
        "Give your advice on the response content "
        "using only brief imperative English sentence(s). "
        "No more than 25 words in total. "
    )

    input_text = f"history:\n" + '\n'.join(attended_history) + \
    f"\nbackground:\n" + '\n'.join([item[1] for item in relevant_static])

    messages = [
        {'role': 'system', 'content': prompt_content},
        {'role': 'user', 'content': input_text}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,
        max_tokens=1024,
    )

    content_advice = response.choices[0].message.content
    print('content_advice:',content_advice)


    ### Step 4: 生成用词建议

    """
    # 采用简化方式：直接根据scene的级别选取用词，并按与content_advice的文本相似度建议用词
    # 暂不考虑skill中的具体掌握度
    level = scenarios[scenarios['uid'] == progress['current_scene']].iloc[0]['level']
    cond = vocab['level'] == level
    level_vocab = vocab[cond]['word'].tolist()
    level_vocab_emb = vocab_emb[torch.as_tensor(cond, dtype=torch.bool)]

    content_advice_emb = embed_model.encode([content_advice], convert_to_tensor=True)[0]
    cosine_scores = util.cos_sim(content_advice_emb, level_vocab_emb)[0]
    _, indices = torch.topk(cosine_scores, k=min(10, len(cosine_scores)))
    vocab_advice = [level_vocab[i] for i in indices]

    print('vocab_advice:', vocab_advice)
    """

    # 修订：不再用整个skill中擅长的词汇计算掌握度，而是根据skill和当前回复句型选择对当前知识点的延申

    def get_expression_advice(score, k_type):
        """根据掌握度分数生成表达建议"""
        if k_type:
            txt = f"the {k_type} and the history"
        else:
            txt = "the history"
        if score < 0.3:
            return f"Use common vocabulary related to {txt}. Keep sentences simple and straightforward."
        elif score < 0.7:
            return f"Use moderately complex vocabulary and sentence structures related to {txt}."
        else:
            return f"Use diverse and expansive vocabulary and sentence structures related to {txt}."

    # 计算掌握度分数
    if sentence_type in ('demo', 'quiz') and target_info is not None:
        # 针对特定知识点的掌握度
        table, uid = target_info['table'], target_info['uid']
        target_skill = current_skill[(current_skill['table'] == table) & (current_skill['uid'] == uid)]
        if not target_skill.empty:
            score = target_skill['proficiency'].iloc[0]
        else:
            score = 0.0  # 未学过的知识点
        focus_context = f"targeting specific knowledge point"
    else:
        # 整个场景的平均掌握度
        if not current_skill.empty:
            score = current_skill['proficiency'].mean()
        else:
            score = 0.0
        focus_context = f"based on overall scene proficiency"

    # 生成表达建议
    expression_advice = get_expression_advice(score, k_type) #  if sentence_type in ('demo', 'quiz') else ''

    print(f'proficiency score: {score:.2f} ({focus_context})')
    print(f'expression advice: {expression_advice}')

    print('[INFO] ----------------------------------------------')

    return {
        'info': {
            'table': target_info['table'] if target_info else None,
            'uid': target_info['uid'] if target_info else None,
            'type': sentence_type,
            'show_knowledge':knowledge # 展示给前端的知识信息
        },
        'task': task,
        'knowledge': knowledge,
        'content_advice': content_advice,
        'expression_advice': expression_advice
    }
    