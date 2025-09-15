from openai import OpenAI
import json
import numpy as np
import pandas as pd
from utils import get_context_without_prompt, len_context
import time

from sentence_transformers import SentenceTransformer, util
import torch
from contextlib import nullcontext

# TODO: 减短prompt，直接根据掌握度范围选择对应的解释方案字符串作为prompt
# 尽可能把能用精确逻辑判断完成的事情全部交给编程，只有模糊逻辑交给LLM，
# 这样能提高LLM回复效率，减少token消耗

# TODO: sentence_checker生成的prompt过长，影响LLM进行高效的量化判断，修改原则同上：
# - 尽可能拆解和精简语言表达，并避免将大段弱关联信息嵌入prompt中（如world的全部内容和整段context）
# - 考虑采用response_prompter中计算相似度提取相关信息，只嵌入真正有用的信息
# - 避免让大模型进行数值评分，只给出掌握度的粗略分级，程序中进行换算得到对应的数值
# - 元数据各字段仅仅是可能用到所以设计，不意味着全部都要用上，不要全部嵌入
# - 总之，尽可能细分pipeline，偏精确逻辑交给小LM和编程，除了模糊逻辑的部分之外，不要让LLM做决策
# - 具体pipeline之后商讨

# 掌握度分级模板系统
PROFICIENCY_TEMPLATES = {
    'beginner': "学习者对该词汇的掌握处于初级阶段，需要更多练习和基础解释。",
    'intermediate': "学习者对该词汇有一定了解，但在使用准确性和语境适切性上仍需提高。",
    'advanced': "学习者能够较为熟练地使用该词汇，但在语义丰富性方面还有提升空间。",
    'proficient': "学习者对该词汇的掌握已达到熟练水平，能够准确、恰当地在各种语境中使用。"
}

embed_model = None
def init_sentence_checker():
    global embed_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


# 掌握度分级映射
def get_proficiency_level(score):
    if score < 0.3:
        return 'beginner'
    elif score < 0.6:
        return 'intermediate'
    elif score < 0.8:
        return 'advanced'
    else:
        return 'proficient'
    
def extract_top3_relevant_static(context, world, top_k=3):
    """提取与当前对话上下文最相关的前3个static信息
    
    参数:
    context: 对话上下文列表
    world: 包含static信息的世界对象
    top_k: 返回的相关信息数量，默认为3
    
    返回:
    list: 包含前3个相关static信息的列表
    """
    # 使用与response_prompter相同的嵌入模型
    
    # 从上下文中提取纯对话内容
    pure_context = get_context_without_prompt(context)
    
    # 获取角色名称
    role_name = {
        'assistant': world['meta']['system_role'],
        'user': world['meta']['user_role']
    }
    
    # 构建对话历史文本
    HISTORY_ATTENTION_LEN = 20  # 关注最近的20条对话
    attended_history = [role_name[item['role']] + ": " + item['content'] for item in pure_context[-HISTORY_ATTENTION_LEN:]]
    
    # 提取static信息文本
    static_texts = [entry[0] for entry in world["static"]]
    if not static_texts:
        return []
    
    # 计算嵌入向量
    static_embeddings = embed_model.encode(static_texts, convert_to_tensor=True)
    history_embeddings = embed_model.encode(attended_history, convert_to_tensor=True)
    
    # 计算相似度（带时间衰减）
    gamma = 0.3 ** (1 / HISTORY_ATTENTION_LEN)  # 衰减因子
    cosine_scores = torch.zeros(len(static_texts), device=static_embeddings.device)
    
    for i, history_emb in enumerate(history_embeddings):
        # 计算当前历史条与所有static的相似度，并应用时间衰减
        cosine_scores_i = util.cos_sim(history_emb, static_embeddings)[0] * gamma ** i
        # 取每个static的最高相似度分数
        cosine_scores = torch.maximum(cosine_scores, cosine_scores_i)
    
    # 获取top_k个最相关的static信息
    top_indices = torch.topk(cosine_scores, k=min(top_k, len(static_texts))).indices.tolist()
    
    # 返回相关的static信息
    return [static_texts[idx] for idx in top_indices]
    
# 列表格式化函数
def brief_list(items, prefix=""):
    if not items:
        return "None"
    return ", ".join(f"{prefix}{item}" for item in items)

# 精简的世界设定
def build_world_description(context, world_genre, world_prologue, world, dynamic): 
    # 只保留最新的3个静态和动态元素
    recent_static = extract_top3_relevant_static(context, world)
    recent_dynamic = dynamic[-3:] if len(dynamic) > 3 else dynamic

    static_str = brief_list([item[0] for item in recent_static])
    dynamic_str = brief_list([item[0] for item in recent_dynamic])
    genre_str = brief_list(world_genre)
    prologue_str = world_prologue[:100] + "..." if len(world_prologue) > 100 else world_prologue

    return f"World Genre: {genre_str}\nBackstory: {prologue_str}\nStatic: {static_str}\nDynamic: {dynamic_str}"

# 精简的对话上下文
def build_dialogue_context(context, system, user):
    # 只保留最近3轮对话
    # NOTE: 确保context为pure
    recent_context = context[-6:] if len(context) > 6 else context
    dialogue_lines = []

    if recent_context:
        for i, utterance in enumerate(recent_context):
            is_system_turn = (utterance['role'] == 'assistant')
            speaker = system if is_system_turn else user
            content = utterance['content'][:50] + "..." if len(utterance['content']) > 50 else utterance['content']
            dialogue_lines.append(f"{speaker}: {content}")

    return "\n".join(dialogue_lines) if dialogue_lines else "No data"

def prompt_maker(scenario, grammar, vocab, world, context):
    # 精简场景描述
    target_language = world['meta']['preferences']['target_language']
    source_language = world['meta']['preferences']['source_language']
    world_genre = world['meta']['preferences']['world_genre']
    world_prologue = world['meta']['preferences']['world_prologue']
    static = world['static']
    dynamic = world['dynamic']
    system = world['meta']['system_role']
    user = world['meta']['user_role']

    scenario_desc = "No scenario data"
    if not scenario.empty:
        scenario_data = scenario.iloc[0]
        scenario_desc = f"Topic: {scenario_data['topic']}\n CEFR Level: {scenario_data['level']}"
        if scenario_data['objective'] and pd.notna(scenario_data['objective']): 
            scenario_desc += f"\n Objective: {scenario_data['objective']}"
    # 包含所有语法要点

    
    knowledge_points = []
    for _, row in vocab.iterrows():
        knowledge_points.append(
            f"expression: {row['word']}"
            + (f" | guideword: {row['guideword']}" if row['guideword'] and pd.notna(row['guideword']) else "")
        )
    
    for _, row in grammar.iterrows():
        examples = row['example'].replace('\n', ' ').strip()
        knowledge_points.append(
            f"grammar: {row['grammar']}" + (f" | examples: {examples}" if examples and pd.notna(examples) else '')
        )
    
    knowledge_points = '\n'.join([f"{i}. {txt}" for i, txt in enumerate(knowledge_points)])

    # 构建世界描述
    world_description = build_world_description(context, world_genre, world_prologue, world, dynamic)

    # 构建对话上下文
    latest_reply = context[-1:]
    context = context[:-1]
    dialogue_context = build_dialogue_context(context, system, user)
    latest_reply = build_dialogue_context(latest_reply, system, user)
    # 构建完整的评估prompt（新增：要求返回 grammar_confidence 0..1；若未检测到语法命中则为 0）
    evaluation_prompt = f"""
You're a language proficiency evaluator. 
You need to assess the proficiency of given knowledge points exhibited in the user's LATEST REPLY.
Output format: a JSON dict with the following keys and values:
- proficiency: a dict
    (Assess IF AND ONLY IF the knowledge point is used in the LATEST REPLY)
    - key: knowledge point number (stringified integer)
    - value: a word indicating the proficiency (beginner/intermediate/advanced/proficient)
- issues: brief, no more than 30 words, in the source language. Empty if no issues.
- advice: brief, imperative sentence(s), no more than 10 words, in English. A reminder for the system to guide the user, if they didn't use the given knowledge.
- improved: brief, no more than 30 words, in the target language. Revision of LATEST REPLY if there are issues. Otherwise empty.
- grammar_confidence: a float between 0 and 1. If and only if the LATEST REPLY demonstrates any grammar point listed in knowledge points below, output your confidence of that grammar detection; otherwise output 0.
Attention: for grammar, focus on structure, not spelling.

System role: {system}; user role: {user}

Target language: {target_language}; source language: {source_language}

Knowledge points:
{knowledge_points}

Previous dialogue:
{dialogue_context}

LATEST REPLY:
{latest_reply}

Scenario:
{scenario_desc}

World setting:
{world_description}
"""

    return evaluation_prompt
    
# 从LLM的粗略评估转换为数值评分
def convert_proficiency_to_score(level):
    level_map = {
        'beginner': 0.3,
        'intermediate': 0.6,
        'advanced': 0.8,
        'proficient': 1.0
    }
    
    result = level_map.get(level.lower(), 0)  # 0 是 默认值，但其实大概率用不上
    assert result, f"Error in Sentence Checker: proficiency level {level} is not valid!"
    
    return result 

def sentence_checker(context: list, responser:dict, scenarios: dict, vocab: dict, grammar: dict, world: dict, skill: dict, progress: dict, archive_lock=nullcontext()):
    # progress中含当前scene_uid
    
    # archives
    # world: 'meta' / 'static' / 'dynamic'
    # skill: df, pd.DataFrame(columns=['table', 'uid', 'good_cases', 'bad_cases', 'proficiency']) 
    
    # current_scene_uid
    scene_uid = progress['current_scene']               # int
    
    # scenarios, vocab, grammar -> pd csv (extracted)
    ex_scenario = scenarios[scenarios['uid'] == scene_uid]      # df csv
    ex_vocab = vocab[vocab['scene_uid'] == scene_uid]           # df csv
    ex_grammar = grammar[grammar['scene_uid'] == scene_uid]     # df csv
    
    
    # 提取技能列表（只包含当前场景的知识），用于索引并更新skill
    skill_list = []
    for _, row in ex_vocab.iterrows():
        skill_list.append(('vocab', row['uid'], row['word']))
    for _, row in ex_grammar.iterrows():
        skill_list.append(('grammar', row['uid'], row['grammar']))

    context = get_context_without_prompt(context)
    # 默认context包含>=2条对话，且最后一条是用户的最新回复

    latest_user_text = ""
    is_latest_user_turn = False
    if isinstance(context, list) and context:
        try:
            last_turn = context[-1] if isinstance(context[-1], dict) else None
            if isinstance(last_turn, dict) and last_turn.get('role') == 'user':
                latest_user_text = last_turn.get('content', '')
                is_latest_user_turn = True
            else:
                # 最后一条不是用户消息：后续将跳过所有 vocab 更新
                latest_user_text = ""
                is_latest_user_turn = False
        except Exception:
            latest_user_text = ""
            is_latest_user_turn = False

    # Early debug marker to ensure this function version is running
    try:
        with open('debug.log', 'a', encoding='utf-8') as f:
            f.write('[SC-ENTRY] is_latest_user_turn=' + str(is_latest_user_turn) + '\n')
    except Exception:
        pass

    # client
    client = responser['client']
    
    # 语句检查器
    with archive_lock:
        prompt = prompt_maker(
            scenario=ex_scenario,
            grammar=ex_grammar,
            vocab=ex_vocab,
            world=world,
            context=context
        )
    
    # PROMPT TEST
    test_file_path = "test_output.txt"

    with open(test_file_path, 'w', encoding='utf-8') as file:
        file.write(prompt)
    
    # response
    messages = [
        {'role': 'user', 'content': prompt}
    ]  
    print("checking sentences...", end="", flush=True)
    
    #stime = time.time()
    response = client.chat.completions.create(
            model=responser['model'],
            messages=messages,
            max_tokens=1024,            # 限制回复长度
            temperature=0.7,            # 控制回复的随机性
            top_p=0.8,                  # 控制多样性的核心采样
    )

    #etime = time.time()
    #clen = len_context(messages)
    #print(f"[INFO] sentencec_checker LLM call time: {etime - stime:.2f}s, clen: {clen}")
    
    output_text = response.choices[0].message.content
    
    # 移除Markdown代码块标记
    output_text = output_text.strip()
    if output_text.startswith('```json'):
        output_text = output_text[7:]
    if output_text.endswith('```'):
        output_text = output_text[:-3]
    output_text = output_text.strip()
    
    # print(f"\nresult:\n {output_text}")
    
    # 解析LLM输出（健壮处理：清洗控制字符、截取JSON片段、失败则回退默认空结构）
    def _safe_parse_json(text: str):
        import re as _re
        # 去除不可见控制字符（但保留换行、制表等常见空白）
        cleaned = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        try:
            return json.loads(cleaned)
        except Exception as e1:
            # 尝试提取最外层花括号的JSON片段
            try:
                m = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
                if m:
                    return json.loads(m.group(0))
            except Exception:
                pass
            _dbg(f"[SC-DEBUG] JSON parse failed: {repr(e1)}; raw(head)=" + cleaned[:200].replace('\n','\\n'))
            return {
                'proficiency': {},
                'issues': [],
                'advice': '',
                'improved': ''
            }

    # 解析LLM输出
    output_json = _safe_parse_json(output_text)

    proficiency = output_json.get('proficiency', {})
    issues = output_json.get('issues', [])              # 以用户母语给出词汇、语法分析
    advice = output_json.get('advice', [])              
    improved = output_json.get('improved', [])          # 以目标语言给出句子更正示例
    # 新增：解析 LLM 给出的整体语法置信度（0..1），缺省为 0
    grammar_confidence = 0.0
    try:
        gc = output_json.get('grammar_confidence', 0)
        if isinstance(gc, (int, float)):
            grammar_confidence = max(0.0, min(1.0, float(gc)))
        elif isinstance(gc, str):
            grammar_confidence = max(0.0, min(1.0, float(gc.strip())))
    except Exception as e:
        _dbg(f"[SC-DEBUG] grammar_confidence parse error: {e!r}; raw={output_json.get('grammar_confidence', None)!r}")

    # 计算质量分数，反映整体语言表现
    # quality_score = sum(updates_with_scores.values()) / len(updates_with_scores)
    
    # print(f"origin result: \n skill:{skill}")

    # 直接在skill原档上更新技能分数
    quality_score = 0
    updates = []
    debug_msgs = []

    # 简单归一化与匹配函数，避免把助手用词算到用户
    import re
    def _dbg(msg: str):
        try:
            with open('debug.log', 'a', encoding='utf-8') as f:
                f.write(msg + '\n')
        except Exception:
            pass
        try:
            debug_msgs.append(msg)
        except Exception:
            pass
    def _normalize_text(s: str) -> str:
        s = s.lower()
        # 保留字母/数字/空格/撇号，去掉其它标点，便于匹配如 "mr." -> "mr"
        return re.sub(r"[^\w\s']+", "", s)

    norm_latest = _normalize_text(latest_user_text)
    # 分词：用正则抽取词元，做全词匹配，避免子串误判（如 "white" 命中 "hi"）
    # 保留含重音字符的字母（\w 在Unicode下已覆盖），以及撇号连接的词
    token_pattern = re.compile(r"[\w]+(?:'[\w]+)?", re.UNICODE)
    latest_tokens = set(token_pattern.findall(norm_latest)) if norm_latest else set()
    _dbg(f"[SC-DEBUG] latest_user_text={latest_user_text!r} tokens={sorted(latest_tokens)} user_turn={is_latest_user_turn}")

    # 预构建多路映射：index / uid / content(规范化)
    index_map = {i: (tbl, int(uid), str(cnt)) for i, (tbl, uid, cnt) in enumerate(skill_list)}
    uid_map = {}
    content_map = {}
    for i, (tbl, uid, cnt) in enumerate(skill_list):
        uid_map[str(int(uid))] = (tbl, int(uid), str(cnt))
        norm_cnt = _normalize_text(str(cnt))
        content_map[norm_cnt] = (tbl, int(uid), str(cnt))

    # 判断 issues 是否提到了指定词（仅在这种情况下跳过该词的更新）
    def _issues_mention_word(issues_obj, target_word: str) -> bool:
        try:
            if not issues_obj:
                return False
            norm_target = _normalize_text(target_word)
            if not norm_target:
                return False
            target_tokens = set(token_pattern.findall(norm_target))
            if not target_tokens:
                return False
            # 支持字符串或列表（其他结构保守转字符串）
            texts = []
            if isinstance(issues_obj, (list, tuple)):
                texts = [str(x) for x in issues_obj]
            else:
                texts = [str(issues_obj)]
            for txt in texts:
                norm_txt = _normalize_text(txt)
                txt_tokens = set(token_pattern.findall(norm_txt))
                # 需要目标词的所有 token 都被包含（处理复合词）
                if target_tokens.issubset(txt_tokens):
                    return True
            return False
        except Exception:
            return False

    # 预过滤：若最后一条不是用户消息，则移除所有 vocab 评分
    # 语法判定现仅依赖 LLM 输出，不再使用本地关键词/模式规则
    def _is_definition_question(user_text: str, target_word: str) -> bool:
        if not user_text:
            return False
        t = _normalize_text(user_text)
        w = _normalize_text(target_word)
        return ("what is" in t) and (w in t)

    # 词频/搭配等通用工具（vocab使用），保留
    def _token_occurs(tokens_set: set[str], target: str) -> int:
        # 统计目标词元在 latest_tokens 中的出现次数（按词元集合近似，若需要更精确可改为在原文中用regex计数）
        tnorm = _normalize_text(target)
        tks = token_pattern.findall(tnorm) if tnorm else []
        # 至少一个词元在集合中则认为出现一次，多词元全部命中则提高强度
        if not tks:
            return 0
        hit = all((tk in tokens_set) for tk in tks)
        return 1 if hit else 0
    def _has_collocation(tokens_set: set[str], head: str, tail: str) -> bool:
        # 简单搭配检测，如 cup of coffee -> 需要 'cup','of','coffee' 都在
        needed = {_normalize_text(head), _normalize_text(tail)}
        return needed.issubset(tokens_set)
    #（本地 grammar 关键词/模式检测相关的工具函数已移除）
    filtered_items = []
    for raw_key, level in proficiency.items():
        # proficiency仅包含那些被检出的知识点的掌握度level
        mapped = None
        path = None
        # 路径1：枚举下标
        try:
            i = int(raw_key)
            if i in index_map:
                mapped = index_map[i]
                path = 'index'
        except Exception:
            pass
        # 路径2：UID
        if mapped is None and str(raw_key) in uid_map:
            mapped = uid_map[str(raw_key)]
            path = 'uid'
        # 路径3：内容字符串（规范化后全词匹配）
        if mapped is None:
            norm_key = _normalize_text(str(raw_key))
            if norm_key in content_map:
                mapped = content_map[norm_key]
                path = 'content'
        if mapped is None:
            _dbg(f"[SC-DEBUG] drop unknown key from LLM: key={raw_key!r}")
            continue
        table, uid, content = mapped
        _dbg(f"[SC-DEBUG] map proficiency key {raw_key!r} -> ({table},{uid},{content!r}) via {path}")

        # 针对vocab做严格过滤：
        # 1) 必须是用户的最新回复；2) 必须全词命中，禁止子串匹配
        if table == 'vocab':
            if not is_latest_user_turn or not latest_tokens:
                # 无用户最新回复或为空：跳过 vocab 更新
                _dbg(f"[SC-DEBUG] skip vocab (no user turn or empty): word={content!r}, latest_tokens={sorted(latest_tokens)}")
                continue
            norm_word = _normalize_text(str(content))
            # 使用相同正则抽取词元；若为复合词，全部 token 均需出现
            norm_word_tokens = set(token_pattern.findall(norm_word)) if norm_word else set()
            if not norm_word_tokens or not norm_word_tokens.issubset(latest_tokens):
                # 丢弃该条更新，防止误将助手说的词记到用户，或子串误判
                _dbg(f"[SC-DEBUG] skip vocab (not whole-word match): word={content!r}, word_tokens={sorted(norm_word_tokens)}, latest_tokens={sorted(latest_tokens)}")
                continue
            else:
                _dbg(f"[SC-DEBUG] accept vocab: word={content!r}, word_tokens={sorted(norm_word_tokens)}, latest_tokens={sorted(latest_tokens)}")

        # 针对grammar的接受判定：仅依据 LLM（该项出现在 proficiency 即接受），并加入置信度门槛；保留“必须是用户轮次”的前置条件。
        if table == 'grammar':
            if not is_latest_user_turn:
                _dbg(f"[SC-DEBUG] skip grammar (no user turn): uid={uid}, latest_tokens={sorted(latest_tokens)}")
                continue
            # 仅 LLM 证据 + 置信度阈值（提升为 0.7）
            grammar_conf_threshold = 0.7
            if grammar_confidence >= grammar_conf_threshold:
                _dbg(f"[SC-DEBUG] accept grammar: uid={uid}, evidence=llm-only, confidence={grammar_confidence:.3f} >= threshold={grammar_conf_threshold}")
            else:
                _dbg(f"[SC-DEBUG] skip grammar: uid={uid}, evidence=llm-only, confidence={grammar_confidence:.3f} < threshold={grammar_conf_threshold}")
                continue

        # 注意：i 仅用于日志，不再作为索引依赖
        # 使用四档等级的启发式映射：
        # - vocab:
        #   beginner(0.3): 定义提问（what is X）
        #   intermediate(0.5): 仅一次简单提及（无明显搭配/结构证据）
        #   advanced(0.8): 正确使用于句中，或搭配明显（如 cup of coffee）
        #   proficient(1.0): 强证据（多次命中/多搭配/复合短语完整出现）
        override_level = level
        if table == 'vocab':
            # 避免“查词/定义提问”被计为使用
            if _is_definition_question(latest_user_text, content):
                _dbg(f"[SC-DEBUG] skip vocab (definition question): word={content!r}")
                continue

            # 仅当 issues 明确提到该词时，才跳过该词的更新
            if _issues_mention_word(issues, content):
                _dbg(f"[SC-DEBUG] skip vocab (issues mention this word): word={content!r}, issues={issues!r}")
                continue

            occ = _token_occurs(latest_tokens, content)
            collocation = False
            # 仅示例：对 coffee 给出一个搭配样例
            if _normalize_text(content) == 'coffee':
                collocation = _has_collocation(latest_tokens, 'cup', 'coffee') and ('of' in latest_tokens)

            # 未出现则不更新，避免无使用也涨分
            if occ == 0 and not collocation:
                _dbg(f"[SC-DEBUG] skip vocab (no occurrence): word={content!r}")
                continue

            # 启发式等级（仅作上限，不抬过 LLM 判定）
            heur_level = 'proficient' if (occ >= 2 or collocation) else 'advanced'
            level_order = {'beginner': 0, 'intermediate': 1, 'advanced': 2, 'proficient': 3}
            # 保护：若 LLM 给出未知标签，回退到 heur_level
            llm_level = level if level in level_order else heur_level
            override_level = llm_level if level_order[llm_level] <= level_order[heur_level] else heur_level
        elif table == 'grammar':
            override_level = level

        filtered_items.append((table, uid, content, override_level))

    for table, uid, content, level in filtered_items:
        
        score = convert_proficiency_to_score(level)
        quality_score += score

        updates.append((table, uid, content, score))
        rows = skill.loc[
            ((skill['table'] == table) & (skill['uid'] == uid)), 
            'proficiency'
            ]
        # 误用强罚：当 issues 明确提到该词，或最终等级为 beginner，直接扣大量熟练度
        is_vocab = (table == 'vocab')
        is_beginner = (str(level).lower() == 'beginner')
        misuse = is_vocab and (_issues_mention_word(issues, content) or is_beginner)
        if misuse:
            PENALTY = 0.30  # 重罚幅度，可根据体验调节（0.2~0.4）
            _dbg(f"[SC-DEBUG] heavy penalty applied: word={content!r}, level={level}, penalty={PENALTY}")
            skill.loc[
                ((skill['table'] == table) & (skill['uid'] == uid)), 
                'proficiency'
                ] = rows.apply(lambda x: max(x - PENALTY, 0.0))
        else:
            # 正常情况：EMA 更新掌握度值
            skill.loc[
                ((skill['table'] == table) & (skill['uid'] == uid)), 
                'proficiency'
                ] = rows.apply(
                    lambda x: 
                    max(min(x * 0.3 + score * 0.7, 1), 0)
                    )
        
    # 基于本轮 updates 计算质量分，更贴近实际本轮表现
    if not updates:
        quality_score = None
    else:
        # updates: [(table, uid, content, score), ...]
        quality_score = sum(s for (_, _, _, s) in updates) / len(updates)
        
    # print(f"final result: \n issues:{issues}, \n advice:{advice}, \n improved:{improved}, \n updated_skill:{skill}, \n updates:{updates}, \n quality_score:{quality_score}")
        
    return {
        'issues': issues, # 词汇和语法运用问题
        'advice': advice, # 对系统的建议，例如引导用户专注于当前知识学习
        'improved': improved, # 对用户回复的参考修正，如果不用修正则留空
        'updates': updates, # [(table, uid, content, this time's score), ...]
        'quality_score': quality_score, # 衡量本次回复知识运用质量
        'debug_msgs': debug_msgs
    } 