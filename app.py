import gradio as gr
from backend import init_state, init_main_dialogue, init_subdialogue, handle_user_input, switch_dialogue, close_dialogue
from utils import StateDict, get_context_without_prompt
import re
import json
from copy import deepcopy
import html
import os
import urllib
from pathlib import Path
import argparse
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_suffix_regex
import re
import unicodedata
import time, hashlib
from functools import partial
from pprint import pprint
import requests
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import plotly.graph_objects as go
import numpy as np, pandas as pd
# 全局状态
state = StateDict()

#  python app.py --config default --reset_to_scene 1580


def initialize_app(config_file=None, kwargs=dict()):
    """初始化应用"""
    init_state(state, config_file, kwargs=kwargs)

    # 更新：如果kwargs中有reset_to_scene参数，则经用户确认后将属于当前target_language的history删除，并重置world和skill

    if 'reset_to_scene' in kwargs:
        scene_uid = int(kwargs['reset_to_scene'])
        if scene_uid in state.data_manager.get_metadata()['scenarios']['uid'].values:
            if input(f"------------------------------------\n[WARNING] --reset_to_scene received, this will DELETE existing history and progress for the current target language [{state.target_lang}]. Type 'yes' to confirm: ").strip().lower() == 'yes':
                print(f"[INFO] Resetting to scene {scene_uid} as requested...")
                state.data_manager.reset_archives(scene_uid)
        else:
            print('[DEBUG] "--reset_to_scene" received invalid scene UID, skipping')
            
    init_main_dialogue(state)
    state.backend_ready = True
    return state

def custom_tokenizer(nlp):
    """Use spaCy defaults but ensure em/en dashes are split and keep hyphenated/apostrophe words.

    - token_match: allows words like don't, mother-in-law (ASCII scope)
    - infix/suffix: add rules to split on em-dash/en-dash sequences
    """
    # Allow internal '-' or apostrophes within letter chunks, but not em-dash
    token_match = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*").match

    # Start from spaCy defaults
    infixes = list(nlp.Defaults.infixes)
    suffixes = list(nlp.Defaults.suffixes)

    # Add em-dash/en-dash (and consecutive runs) as split points
    dash_re = r"[\u2014\u2013]+"  # em dash U+2014, en dash U+2013
    infixes += [dash_re]
    suffixes += [dash_re]

    infix_re = compile_infix_regex(infixes)
    suffix_re = compile_suffix_regex(suffixes)

    return Tokenizer(
        nlp.vocab,
        token_match=token_match,
        infix_finditer=infix_re.finditer,
        suffix_search=suffix_re.search,
    )

# 初始化 nlp
nlp = spacy.blank("en")
nlp.tokenizer = custom_tokenizer(nlp)

def extract_words_from_text(text):
    """Tokenize with spaCy then further split leading/trailing punctuation runs.

    Fixes cases like words sticking to em-dashes: "word——" -> ["word", "——"].
    Also separates quotes/brackets attached to words.
    """
    if not isinstance(text, str):
        return []

    doc = nlp(text)
    base = [{
        'word': t.text,
        'start': t.idx,
        'end': t.idx + len(t.text)
    } for t in doc]

    final_words = []
    # runs of trailing punctuation including em/en dashes, quotes, and ellipsis
    trailing_re = re.compile(r"[\?\!\,\.:;…\u2014\u2013'\"“”‘’]+$")
    leading_re = re.compile(r"^[\(\)\[\]\{\}<>'\"“”‘’\u2014\u2013]+")
    # splitter for internal em/en dash runs
    internal_dash_split = re.compile(r"([\u2014\u2013]+)")

    for w in base:
        s = w['start']
        e = w['end']
        token = w['word']

        # Split leading punctuation run
        m_lead = leading_re.search(token)
        if m_lead:
            lead = m_lead.group(0)
            if lead:
                final_words.append({'word': lead, 'start': s, 'end': s+len(lead)})
                token = token[len(lead):]
                s = s + len(lead)

        # Split trailing punctuation run
        m_tail = trailing_re.search(token)
        if m_tail:
            tail = m_tail.group(0)
            core = token[:-len(tail)] if len(tail) < len(token) else ''
            if core:
                # further split internal em/en dashes inside the core
                pos = s
                for part in internal_dash_split.split(core):
                    if not part:
                        continue
                    final_words.append({'word': part, 'start': pos, 'end': pos+len(part)})
                    pos += len(part)
            if tail:
                final_words.append({'word': tail, 'start': e-len(tail), 'end': e})
        else:
            if token:
                # further split internal em/en dashes
                pos = s
                for part in internal_dash_split.split(token):
                    if not part:
                        continue
                    final_words.append({'word': part, 'start': pos, 'end': pos+len(part)})
                    pos += len(part)

    return final_words
# def extract_words_from_text(text):
#     """从文本中提取单词及其位置，支持中英混合"""
#     if not isinstance(text, str):
#         return []
#     words = []
#     pattern = re.compile(r'([\u4e00-\u9fa5]+|[a-zA-Z]+|[^\u4e00-\u9fa5a-zA-Z\s]+)')
#
#     for match in pattern.finditer(text):
#         if match.group().strip():
#             words.append({
#                 'word': match.group(),
#                 'start': match.start(),
#                 'end': match.end()
#             })
#     return words


def create_clickable_content(content):
    """为内容中的单词创建可点击的HTML"""
    if not isinstance(content, str):
        return str(content)

    return create_clickable_words(content)

def extract_bold_marks(text):
    """
    提取文本中所有成对的⟪⟫标记，返回处理后的文本及待加粗范围
    
    参数:
        text: 原始文本字符串
    返回:
        tuple: (处理后的文本, 待加粗索引范围列表)
            - 处理后的文本: 已去除所有⟪⟫标记的文本
            - 待加粗索引范围列表: 每个元素为(start, end)的元组，代表在处理后文本中
              需要加粗的内容的闭区间索引范围
    """
    # 标准化文本编码
    normalized_text = unicodedata.normalize('NFKC', text)
    
    # 查找所有成对的⟪⟫标记（假设无嵌套）
    bracket_pattern = re.compile(r'⟪(.*?)⟫')
    matches = list(bracket_pattern.finditer(normalized_text))
    
    # 提取标记位置和内容
    bracket_info = [
        (match.start(), match.end() - 1, match.group(1))  # (左标记索引, 右标记索引, 内容)
        for match in matches
    ]
    
    # 构建去除标记后的文本并计算加粗范围
    processed_parts = []
    bold_ranges = []
    last_processed_pos = 0  # 记录上一次处理到的位置
    
    for left_pos, right_pos, content in bracket_info:
        # 添加标记前的文本
        processed_parts.append(normalized_text[last_processed_pos:left_pos])
        
        # 计算当前内容在处理后文本中的起始索引
        content_start = len(''.join(processed_parts))
        
        # 添加标记内的内容
        processed_parts.append(content)
        
        # 计算当前内容在处理后文本中的结束索引（闭区间）
        content_end = content_start + len(content) - 1
        bold_ranges.append((content_start, content_end))
        
        # 更新处理位置
        last_processed_pos = right_pos + 1
    
    # 添加剩余文本
    processed_parts.append(normalized_text[last_processed_pos:])
    processed_text = ''.join(processed_parts)
    
    return processed_text, bold_ranges

def contains_text(text): # 是否包含实义字母/数字，而不是全为标点、EMOJI或其他
    return any(unicodedata.category(c).startswith(('L', 'N')) for c in text)

def create_clickable_words(text):
    """将文本按单词分割并创建可点击的HTML"""
    text = unicodedata.normalize('NFKC', text) # 确保特殊字符编码一致
    text, bold_ranges = extract_bold_marks(text)
    result = ""
    last_end = 0
    word_data = extract_words_from_text(text)

    # NEW: 新增预处理阶段，

    for word_info in word_data:
        word_start = word_info['start']
        word_end = word_info['end']
        word = word_info['word']

        result += text[last_end:word_info['start']]
        formatted_word = html.escape(word)
        safe_word = formatted_word.replace("'", "\\'")

        # 判断是否加粗
        is_bold = any(word_start <= end and word_end >= start for start, end in bold_ranges)
        bold = "font-weight: 900;" if is_bold else "" # 加到最粗
        
        # 判断是否可点击
        if contains_text(word):
            header = f'class="clickable-word" onclick="clickWord(\'{safe_word}\')"'
        else:
            header = 'class="common-word"'
        
        formatted_word = (
            f'<span {header} '
            f'style="{bold}">{formatted_word}</span>'
        )

        result += formatted_word
        last_end = word_info['end']

    result += text[last_end:]
    return result


def handle_main_input(user_input):
    """
    输出：[main_input, main_chatbot, error_display, radar_chart]
    """
    if user_input.strip():
        handle_user_input(state, user_input, level=0) # 输入主会话
    main_context = get_main_display()
    radar_fig = generate_radar_chart()
    # 返回顺序与 outputs 对齐：清空可见输入，同时把原始输入放入隐藏缓冲
    return "", main_context, "", radar_fig, (user_input or "")

def handle_sub_input(user_input):
    """
    输出：[sub_input, sub_chatbot, error_display]
    """
    if user_input.strip():
        # 仅当active_dialogue_level有效时输入子会话
        if 0 < state.active_dialogue_level < len(state.dialogue_chain):
            handle_user_input(state, user_input, level=state.active_dialogue_level)
    sub_context = get_sub_display()
    radar_fig = generate_radar_chart()
    # 返回顺序与 outputs 对齐：清空可见输入，同时把原始输入放入隐藏缓冲
    return "", sub_context, "", radar_fig, (user_input or "")

def pre_echo_main_input(user_input):
    """
    轻量级：仅把用户输入立即加入主会话上下文并返回显示，用于在加载动画出现前/同时先弹出用户消息。
    输出：[main_input, main_chatbot, error_display, radar_chart]
    """
    try:
        if user_input and user_input.strip():
            # 保证主会话已初始化
            if len(state.dialogue_chain) == 0:
                init_main_dialogue(state)
            # 只追加用户消息，info 稍后由句法检查阶段补齐
            state.dialogue_chain[0]['history'].append({
                'role': 'user',
                'content': user_input
            })
    except Exception as e:
        print(f"[WARN] pre_echo_main_input error: {e}")
    main_context = get_main_display()
    radar_fig = generate_radar_chart()
    # 输出 5 个值：[main_input, main_chatbot, error_display, radar_chart, main_last_input]
    return "", main_context, "", radar_fig, (user_input or "")

def process_main_input(user_input):
    """
    重活：调用后端处理，但不再次追加用户消息（append_user=False），避免重复。
    输出：[main_input, main_chatbot, error_display, radar_chart]
    """
    if user_input and user_input.strip():
        handle_user_input(state, user_input, level=0, sub_interpreter='interactive', append_user=False)
    main_context = get_main_display()
    radar_fig = generate_radar_chart()
    return "", main_context, "", radar_fig

def pre_echo_sub_input(user_input):
    """
    轻量级：仅把用户输入立即加入当前子会话上下文。
    输出：[sub_input, sub_chatbot, error_display, radar_chart]
    """
    try:
        if user_input and user_input.strip():
            if 0 < state.active_dialogue_level < len(state.dialogue_chain):
                state.dialogue_chain[state.active_dialogue_level]['history'].append({
                    'role': 'user',
                    'content': user_input
                })
    except Exception as e:
        print(f"[WARN] pre_echo_sub_input error: {e}")
    sub_context = get_sub_display()
    radar_fig = generate_radar_chart()
    # 输出 5 个值：[sub_input, sub_chatbot, error_display, radar_chart, sub_last_input]
    return "", sub_context, "", radar_fig, (user_input or "")

def process_sub_input(user_input):
    """
    重活：对子会话做后端处理，避免重复追加（append_user=False）。
    输出：[sub_input, sub_chatbot, error_display, radar_chart]
    """
    if user_input and user_input.strip():
        if 0 < state.active_dialogue_level < len(state.dialogue_chain):
            handle_user_input(state, user_input, level=state.active_dialogue_level, append_user=False)
    sub_context = get_sub_display()
    radar_fig = generate_radar_chart()
    return "", sub_context, "", radar_fig

def handle_word_click_event(trigger_value):
    print(f"[DEBUG] Word click trigger received: {trigger_value}")

    if not trigger_value: # NOTE: 设置为空时也可能再次触发事件，应拦截
        return "", get_main_display(), get_sub_display(), get_navigation_display()
    root_context = deepcopy(state.dialogue_chain[0]['history'])

    new_level = len(state.dialogue_chain)
    print(f"[DEBUG] Creating sub-dialogue at level {new_level} for word '{trigger_value}'")


    init_subdialogue(state, root_context, new_level, trigger_value)

    # 将活跃对话级别设置为新创建的子对话
    state.active_dialogue_level = new_level

    print(f"[DEBUG] Sub-dialogue initialized successfully")

    main_context = get_main_display()
    nav_html = get_navigation_display()
    sub_context = get_sub_display()

    print(f"[DEBUG] Sub-dialogue display updated")
    return "", main_context, sub_context, nav_html



# 新增：处理导航按钮点击事件
def handle_nav_click(level_str):
    """切换到指定层级的对话"""
    level = int(level_str)
    print(f"[DEBUG] Switching to dialogue level: {level}")
    switch_dialogue(state, level)

    main_context = get_main_display()
    sub_context = get_sub_display()
    nav_html = get_navigation_display()

    return main_context, sub_context, nav_html


# 新增：处理导航按钮关闭事件
def handle_nav_close(level_str):
    """删除指定层级的对话"""
    level = int(level_str)
    print(f"[DEBUG] Closing dialogue level: {level}")
    close_dialogue(state, level)

    main_context = get_main_display()
    sub_context = get_sub_display()
    nav_html = get_navigation_display()

    return main_context, sub_context, nav_html

def get_main_display():
    assert len(state.dialogue_chain) > 0
    main_dialogue = state.dialogue_chain[0]
    pure_context = get_context_without_prompt(main_dialogue['history'])
    formatted_context = []

    for message in pure_context:
        new_message = deepcopy(message)
        if new_message['role'] == 'assistant':
            content = create_clickable_content(new_message['content'])

            # 动态 tooltip
            show_knowledge = ""
            if isinstance(new_message.get("info"), dict):
                show_knowledge = new_message["info"].get("show_knowledge", "")
            show_knowledge = html.escape(str(show_knowledge)) if show_knowledge else ""
            
            # 判断 info.type 决定颜色；对于无知识点的回答不显示 info 图标
            info_type = (
                new_message.get("info", {}).get("type", "").lower()
                if isinstance(new_message.get("info"), dict) else ""
            )
            if info_type in ('demo', 'quiz') and show_knowledge:
                content += f' <span class="info-icon assistant-{info_type}" title="{show_knowledge}">ⓘ</span>'

            # 若有audio，添加播放/暂停按钮
            audio_path = new_message.get("info", {}).get("audio_path", "") if isinstance(new_message.get("info"), dict) else ""
            if audio_path:
                content += create_audio_player_html(audio_path, 'main')

            new_message['content'] = content


        # User 消息处理
        elif new_message['role'] == 'user':
            content = html.escape(str(new_message['content']))
            info_data = new_message.get("info", {})
            issues = info_data.get("issues", "")
            improved = info_data.get("improved", "")

            tooltip_parts = []
            if issues:
                tooltip_parts.append(issues)
            if improved:
                tooltip_parts.append(improved)

            tooltip_text = "\n".join(tooltip_parts) if tooltip_parts else ""
            if tooltip_text:
                tooltip_text = html.escape(tooltip_text)
                content += f' <span class="user-info-icon" title="{tooltip_text}">ⓘ</span>'
            new_message['content'] = content

        formatted_context.append(new_message)


    scene_completed = state.data_manager.get_archive()['progress']['scene_completed']
    print("[DEBUG] progress:", state.data_manager.get_archive()['progress'])
    if scene_completed:
        print("[INFO] (get_main_display) scene completed, adding next scene button")
        button_html = '''
        <div style="text-align: center; margin: 10px 0;">
            <button class="next-scene-btn" onclick="triggerNextScene()">Proceed to the next scene...</button>
        </div>
        '''
        formatted_context.append({
            'role': 'assistant',  # 自定义角色用于样式控制
            'content': button_html
        })

    return formatted_context

def handle_next_scene(trigger_value):
    """处理切换到下一场景的事件"""
    if trigger_value:
        print("[DEBUG] Next scene button clicked")
        state.data_manager.save_history(state.dialogue_chain[0]['history'])
        if state.data_manager.next_scene():
            # 若场景切换成功，重新初始化主会话（自动从新场景开始）
            init_main_dialogue(state)
    return "", get_main_display(), generate_radar_chart()

def create_audio_player_html(audio_path, dialogue):
    """为音频创建播放/暂停按钮的HTML"""
    assert dialogue in ('main', 'sub')
    # 生成唯一按钮ID：时间戳 + 路径哈希值
    timestamp = int(time.time())
    path_hash = hashlib.md5(audio_path.encode()).hexdigest()[:8]

    #posix_path = Path(audio_path).as_posix()

    #file_url = '/file=' + audio_path  # 假设 audio_path 是相对于某个静态文件目录的路径
    file_url = '/gradio_api/file=' + audio_path.replace('\\', '/') #urllib.parse.quote(posix_path, safe='/')

    # print(f"[DEBUG] before: {audio_path}, after: {posix_path}, {file_url}")
    button_id = f"{dialogue}-audio-btn-{timestamp}-{path_hash}"

    #escaped_path = repr(audio_path)[1:-1]  # 已经正确转义，勿修改
    escaped_path = repr(file_url)[1:-1]
    print(f"[DEBUG] preprocessed audio path: {escaped_path}")
    return f'''
    <button class="audio-control-btn" id="{button_id}" onclick="toggleAudio('{escaped_path}', '{button_id}')">▶️</button>
    '''



#interface = None
#audio_finish_trigger = None
local_url = "http://127.0.0.1:8760"

def audio_finish_callback(button_id):
    
    # TODO: 待修正
    """assert local_url is not None
    url = f"{local_url}/run/audio_finish"
    payload = {
        "data": [button_id]
    }
    response = requests.post(url, json=payload)
    print("\n[Audio Finish Response Details]")
    print(f"Status Code: {response.status_code}")  # 响应状态码（200表示成功）
    print(f"Response Headers: {response.headers}")  # 响应头
    try:
        # 尝试解析JSON响应体
        response_json = response.json()
        print("Response Body (JSON):")
        pprint(response_json)  # 格式化打印JSON内容
    except json.JSONDecodeError:
        # 如果响应不是JSON格式，直接打印文本
        print(f"Response Body (Text): {response.text}")
    
    return response  # 可选：返回响应对象供后续处理
    """

def handle_audio_control(audio_path_and_command):
    """处理音频播放/暂停按钮点击事件"""

    """if not audio_path_and_command:
        return ""
    
    dic = json.loads(audio_path_and_command)
    audio_path = dic['audioPath']
    command = dic['command']
    buttonId = dic['buttonId']
    gui_callback = partial(audio_finish_callback, buttonId)
    print("handling:", audio_path, command)
    audio_generator = state.audio_generator
    if not audio_generator:
        return ""

    if command == "play":
        audio_generator.stop_stream_player() # 暂停可能正在进行的流式播放
        if audio_generator.audio_player.audio_file != audio_path:
            audio_generator.audio_player.set_audio_file(audio_path)
        if audio_generator.audio_player.status() == 'paused':
            audio_generator.audio_player.resume()
        else:
            audio_generator.audio_player.play(gui_callback)
    elif command == "pause":
        # 如果当前播放的音频与按钮对应的音频一致，暂停播放，否则操作无效
        if audio_generator.audio_player.audio_file == audio_path:
            audio_generator.audio_player.pause()"""

    # 前端按钮状态已经在JS中改变，不需要额外更新
    return ""

 

def get_sub_display():
    if len(state.dialogue_chain) > 1 and state.active_dialogue_level > 0:
        current_dialogue = state.dialogue_chain[state.active_dialogue_level]
        pure_context = get_context_without_prompt(current_dialogue['history'])
        formatted_context = []
        for message in pure_context:
            if message['role'] == 'assistant':
                new_message=deepcopy(message)
                if isinstance(new_message['content'], list) and all(isinstance(item, str) for item in new_message['content']):
                    # instance interpretation
                    is_image_list = all(os.path.exists(item) and (
                                item.endswith('.png') or item.endswith('.jpg') or item.endswith('.jpeg')) for item in
                                        new_message['content'])
                    if is_image_list:
                        # 渲染图片
                        for i in range(len(new_message['content'])):
                            temp_message=deepcopy(new_message)
                            temp_message['content'] = [new_message['content'][i]]
                            formatted_context.append(temp_message)
                        # 渲染图片对应的例句
                        if 'example' in new_message.get('info', {}):
                            example_content = create_clickable_content(new_message['info']['example'])

                            # 若有audio，添加播放按钮
                            audio_path = new_message.get("info", {}).get("audio_path", "")
                            example_content += create_audio_player_html(audio_path, 'sub')

                            formatted_context.append({
                                'role': 'assistant',
                                'content': example_content
                            })

                    else: # 视为文本列表，依次渲染
                        new_message['content'] = create_clickable_content(' '.join(new_message['content']))
                elif isinstance(new_message['content'], str):
                    # interactive interpretation
                    new_message['content'] = create_clickable_content(new_message['content'])

                    # 若有audio，添加播放按钮
                    audio_path = new_message.get("info", {}).get("audio_path", "")
                    new_message['content'] += create_audio_player_html(audio_path, 'sub')

                    formatted_context.append(new_message)
            else:
                formatted_context.append(message)
        # print("[INFO] subdialogue context:", formatted_context)
        return formatted_context
    return []

def get_navigation_display():
    """获取导航栏显示内容，修复按钮显示字典的问题"""
    # if len(state.dialogue_chain) <= 1:
    #     return ""

    nav_buttons = []
    for i, dialogue in enumerate(state.dialogue_chain):
        # 1. 获取 keyword 字段的值
        if i == 0: continue # Main dialogue不需要对应的按钮
        keyword_data = dialogue.get('keyword', None)

        # 2. 从 keyword_data 中提取可显示的字符串
        display_text = ""
        if isinstance(keyword_data, dict) and 'content' in keyword_data:
            display_text = str(keyword_data['content'])
        elif isinstance(keyword_data, str):
            display_text = keyword_data

        # 3. 如果提取失败，使用默认值
        if not display_text:
            display_text = f'Keyword {i}'

        is_active = (i == state.active_dialogue_level)
        button_class = "nav-button active" if is_active else "nav-button"

        close_btn_html = f'<span class="close-btn" onclick="event.stopPropagation(); closeNavButton({i})">×</span>' if i > 0 else ''

        button_html = f'''
        <div class="{button_class}" onclick="clickNavButton({i})">
            <span>{html.escape(display_text)}</span>
            {close_btn_html}
        </div>
        '''
        nav_buttons.append(button_html)
    nav_html = f'''
    <div class="navigation-panel">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 12px; color: #6c757d;">Conversation Topics:</span>
            <button class="instance-interpreter-btn" onclick="instanceInterpreter()">Explain with pictures...</button>
        </div>
        <div class="nav-button-container">{"".join(nav_buttons)}</div>
    </div>
    '''

    return nav_html


def generate_radar_chart():
    """生成雷达图，展示当前场景的词汇和语法掌握情况"""
    if not hasattr(state, 'data_manager'):
        return None
    
    try:
        # 获取当前场景状态数据
        status = state.data_manager.get_current_scene_status()
        current_vocab = status.get('current_vocab', pd.DataFrame())
        current_grammar = status.get('current_grammar', pd.DataFrame())
        current_skill = status.get('current_skill', pd.DataFrame())
        current_scenario = status.get('current_scenario', None)
        if current_scenario is not None:
            scene_uid = current_scenario['uid']
            scene_topic = current_scenario['topic']
            scene_level = current_scenario['level']
            title = f"Scene #{scene_uid} : {scene_topic} (Level {scene_level})"
        else:
            title = ""

        # 准备雷达图数据
        categories = []
        values = []
        full_names = []
        types = []
        proficiencies = []
        
        # 处理词汇数据
        for _, vocab_row in current_vocab.iterrows():
            vocab_id = vocab_row['uid']
            # 查找对应的掌握度
            skill_row = current_skill[(current_skill['table'] == 'vocab') & 
                                     (current_skill['uid'] == vocab_id)]
            
            proficiency = skill_row.iloc[0]['proficiency'] if not skill_row.empty else 0.0
            
            # 截断过长的名称
            name = vocab_row['word']
            truncated_name = name[:20] + '...' if len(name) > 20 else name
            
            categories.append(truncated_name)
            values.append(proficiency)
            full_names.append('Expression: ' + name)
            types.append(f"Part of speech: {vocab_row.get('type', 'n/a')}")
            proficiencies.append(f"{proficiency:.2f}")
        
        # 处理语法点数据
        for _, grammar_row in current_grammar.iterrows():
            grammar_id = grammar_row['uid']
            # 查找对应的掌握度
            skill_row = current_skill[(current_skill['table'] == 'grammar') & 
                                     (current_skill['uid'] == grammar_id)]
            
            proficiency = skill_row.iloc[0]['proficiency'] if not skill_row.empty else 0.0
            
            # 截断过长的名称
            name = grammar_row['grammar']
            truncated_name = name[:20] + '...' if len(name) > 20 else name
            
            categories.append(truncated_name)
            values.append(proficiency)
            full_names.append('Grammar: ' + name)
            types.append(f"Type: {grammar_row.get('type', 'n/a')}")
            proficiencies.append(f"{proficiency:.2f}")
        
        if not categories:  # 没有数据时返回空
            return None
            
        # 创建雷达图
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(54, 162, 235)')
        ))
        
        # 设置雷达图布局
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 16},
                'y': 0.95,  # 标题位置（0-1之间，1为顶部）
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
                    tickfont={'size': 10},
                    #angle=45,
                    dtick=0.2
                ),
                angularaxis=dict(
                    tickfont={'size': 11},
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50),
            height=550
        )
        
        # 添加自定义悬停信息
        fig.update_traces(
            customdata=np.stack([full_names, types, proficiencies], axis=-1),
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Proficiency: %{customdata[2]}<extra></extra>',
        )
        
        return fig
        
    except Exception as e:
        print(f"Error generating radar chart: {str(e)}")
        return None



# 创建Gradio界面
with open("app.css", "r", encoding='utf-8') as css_file:
    CSS = css_file.read()

def handle_instance_interpreter(trigger_value):
    if trigger_value == 'toggle':
        if 0 < state.active_dialogue_level < len(state.dialogue_chain):
            print("[DEBUG] INSTANCE INTERPRETER TRIGGERED")
            handle_user_input(state, None, level=state.active_dialogue_level, sub_interpreter='instance')
    sub_context = get_sub_display()
    return "", sub_context, "" # [instance_interpreter_trigger, sub_chatbot, error_display]


def create_interface(config_file=None, kwargs=dict()):

    #gr.set_static_paths(paths=[Path("generated_audios")])

    interface = gr.Blocks(
        title="Linguiverse",
        css=CSS
    )


    with interface:
        # ====== [LOADING OVERLAY - START] 精致加载页与进度条 ======
        gr.HTML(
            value='''
            <div class="lv-loading-overlay" id="lv-loading-overlay">
              <div class="lv-stars"></div>
              <div class="lv-slogan">Linguiverse - Your Best Language Learning Assistant</div>
              <div class="lv-loading-card">
                <div class="lv-loading-header">
                  <div class="lv-logo"></div>
                  <div>
                    <div class="lv-title">Linguiverse 正在启动...</div>
                    <div class="lv-subtitle">正在加载模型与场景数据，请稍候 <span class="lv-dots"><span></span><span></span><span></span></span></div>
                  </div>
                </div>
                <div class="lv-progress-shell">
                  <div class="lv-progress-bar" id="lv-progress-bar"></div>
                </div>
                <div class="lv-progress-info" id="lv-progress-info">0%</div>
                <div class="lv-current-task" id="lv-current-task">正在准备资源...</div>
                <div class="lv-tips">
                  <span class="lv-tip">语料初始化</span>
                  <span class="lv-tip">知识图谱装载</span>
                  <span class="lv-tip">界面渲染</span>
                </div>
              </div>
            </div>
            ''',
            elem_id="lv-loading-overlay-root"
        )
        # ====== [LOADING OVERLAY - END] ======

        gr.Markdown(
            "# 💥 Linguiverse - natural language acquisition AI")

        with gr.Row(equal_height=True, elem_classes="main-container"):
            with gr.Column(scale=1, elem_classes="dialogue-container"):
                gr.Markdown("## 💬 Main Dialogue")
                main_chatbot = gr.Chatbot(
                    sanitize_html=False,
                    render_markdown=False,
                    type="messages",
                    height=560
                )
                with gr.Row(elem_classes="input-area"):
                    main_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4,
                                            container=False, lines=1)
                    main_submit = gr.Button("Send", scale=1, variant="primary")

            with gr.Column(scale=1, elem_classes="dialogue-container"):
                gr.Markdown("## 🔍 Knowledge Exploration")
                
                with gr.Tabs():
                    with gr.Tab("Subdialogue"):
                        navigation_display = gr.HTML(value="", elem_classes="navigation")
                        sub_chatbot = gr.Chatbot(
                            sanitize_html=False,
                            render_markdown=False,
                            type="messages"
                        )
                        with gr.Row(elem_classes="input-area"):
                            sub_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4,
                                                container=False, lines=1)
                            sub_submit = gr.Button("Send", scale=1, variant="primary")
                    with gr.Tab("Skills Radar"):
                        radar_chart = gr.Plot(label="Vocabulary & Grammar Proficiency")
 
        
        # 隐藏的触发器，用于捕获单词点击事件和导航点击事件
        word_click_trigger = gr.Textbox(value="", elem_id="word-click-trigger", elem_classes="hidden-trigger")
        nav_click_trigger = gr.Textbox(value="", elem_id="nav-click-trigger", elem_classes="hidden-trigger")
        nav_close_trigger = gr.Textbox(value="", elem_id="nav-close-trigger", elem_classes="hidden-trigger")
        audio_control_trigger = gr.Textbox(value="", elem_id="audio-control-trigger", elem_classes="hidden-trigger")
        global audio_finish_trigger
        audio_finish_trigger = gr.Textbox(value="", elem_id="audio-finish-trigger", elem_classes="hidden-trigger")
        error_display = gr.Textbox(visible=False)
        instance_interpreter_trigger = gr.Textbox(value="", elem_id="instance-interpreter-trigger", elem_classes="hidden-trigger")
        next_scene_trigger = gr.Textbox(value="", elem_id="next-scene-trigger", elem_classes="hidden-trigger")
        # 新增：隐藏缓冲区，用于在链式调用中传递原始输入
        main_last_input = gr.Textbox(value="", visible=False)
        sub_last_input = gr.Textbox(value="", visible=False)

        # 采用链式：先轻量预回显，再进行耗时处理
        main_submit.click(
            fn=pre_echo_main_input,
            inputs=[main_input],
            outputs=[main_input, main_chatbot, error_display, radar_chart, main_last_input]
        ).then(
            fn=process_main_input,
            inputs=[main_last_input],
            outputs=[main_input, main_chatbot, error_display, radar_chart]
        )
        main_input.submit(
            fn=pre_echo_main_input,
            inputs=[main_input],
            outputs=[main_input, main_chatbot, error_display, radar_chart, main_last_input]
        ).then(
            fn=process_main_input,
            inputs=[main_last_input],
            outputs=[main_input, main_chatbot, error_display, radar_chart]
        )
        sub_submit.click(
            fn=pre_echo_sub_input,
            inputs=[sub_input],
            outputs=[sub_input, sub_chatbot, error_display, radar_chart, sub_last_input]
        ).then(
            fn=process_sub_input,
            inputs=[sub_last_input],
            outputs=[sub_input, sub_chatbot, error_display, radar_chart]
        )
        sub_input.submit(
            fn=pre_echo_sub_input,
            inputs=[sub_input],
            outputs=[sub_input, sub_chatbot, error_display, radar_chart, sub_last_input]
        ).then(
            fn=process_sub_input,
            inputs=[sub_last_input],
            outputs=[sub_input, sub_chatbot, error_display, radar_chart]
        )
        
        

        word_click_trigger.change(fn=handle_word_click_event, inputs=[word_click_trigger],
                                  outputs=[word_click_trigger, main_chatbot, sub_chatbot, navigation_display])

        # 绑定新的导航事件
        nav_click_trigger.change(fn=handle_nav_click, inputs=[nav_click_trigger],
                                 outputs=[main_chatbot, sub_chatbot, navigation_display])
        nav_close_trigger.change(fn=handle_nav_close, inputs=[nav_close_trigger],
                                 outputs=[main_chatbot, sub_chatbot, navigation_display])

        # 新增：绑定音频控制事件
        audio_control_trigger.change(fn=handle_audio_control, inputs=[audio_control_trigger],
                                     outputs=[audio_control_trigger])
        
        audio_finish_trigger.change(fn=lambda x:"", inputs=[audio_finish_trigger], outputs=[audio_finish_trigger], js="""
            (data) => {
                console.log("received:", data);
                const button = document.getElementById(data);
                if (button) {
                    button.textContent = '▶️';
                }
                return data;
            }
        """, api_name="audio_finish")

        instance_interpreter_trigger.change(
            fn=handle_instance_interpreter,
            inputs=[instance_interpreter_trigger],
            outputs=[instance_interpreter_trigger, sub_chatbot, error_display]
        )

        next_scene_trigger.change(
            fn=handle_next_scene,
            inputs=[next_scene_trigger],
            outputs=[next_scene_trigger, main_chatbot, radar_chart]
        )

        def initialize_display():
            if not state.get('backend_ready', False):
                initialize_app(config_file=config_file, kwargs=kwargs)
            main_context = get_main_display()
            nav_html = get_navigation_display()
            sub_context = get_sub_display()
            radar_fig = generate_radar_chart()
            return main_context, sub_context, nav_html, radar_fig

        interface.load(
            fn=initialize_display,
            inputs=None,
            outputs=[main_chatbot, sub_chatbot, navigation_display, radar_chart],
            js="""
            () => {
                // ====== [LOADING OVERLAY - START] 进度与隐藏逻辑 ======
                const overlay = document.getElementById('lv-loading-overlay');
                const bar = document.getElementById('lv-progress-bar');
                const info = document.getElementById('lv-progress-info');
                const taskEl = document.getElementById('lv-current-task');
                let progress = 0;
                let running = true;
                const tasks = [
                    '正在加载词汇数据',
                    '正在装载语法知识',
                    '正在准备场景资源',
                    '正在连接音频模块',
                    '正在初始化界面组件'
                ];
                let taskIdx = 0;
                const rotateTask = () => {
                    if (!taskEl) return;
                    taskEl.textContent = tasks[taskIdx % tasks.length] + '...';
                    taskIdx++;
                };
                rotateTask();
                const taskIv = setInterval(rotateTask, 1200);

                // ====== 高特效星空与流星（随机生成） ======
                const starsBox = document.querySelector('.lv-stars');

                const rnd = (min, max) => Math.random() * (max - min) + min;
                const irnd = (min, max) => Math.floor(rnd(min, max));

                // 生成随机星星，避免平铺花纹
                const createStars = (count = 180) => {
                    if (!starsBox) return;
                    starsBox.innerHTML = '';
                    for (let i = 0; i < count; i++) {
                        const s = document.createElement('span');
                        s.className = 'star';
                        const x = rnd(0, 100);
                        const y = rnd(0, 100);
                        const scale = rnd(0.6, 1.25);
                        const twinkle = rnd(2.4, 4.2) + 's';
                        const delay = rnd(-3, 3) + 's';
                        s.style.left = x + '%';
                        s.style.top = y + '%';
                        s.style.transform = `scale(${scale})`;
                        s.style.setProperty('--twinkle', twinkle);
                        s.style.animationDelay = delay;
                        starsBox.appendChild(s);
                    }
                    // 额外添加“大亮星”，更亮更大、更慢闪烁
                    const giants = Math.max(6, Math.floor(count * 0.03));
                    for (let g = 0; g < giants; g++) {
                        const s = document.createElement('span');
                        s.className = 'star giant';
                        const x = rnd(0, 100);
                        const y = rnd(0, 100);
                        const scale = rnd(1.6, 2.4);
                        const twinkle = rnd(3.6, 6.0) + 's';
                        const delay = rnd(-4, 4) + 's';
                        s.style.left = x + '%';
                        s.style.top = y + '%';
                        s.style.transform = `scale(${scale})`;
                        s.style.setProperty('--twinkle', twinkle);
                        s.style.animationDelay = delay;
                        starsBox.appendChild(s);
                    }
                };
                createStars(340);

                const tick = () => {
                    if (!running) return;
                    // 缓慢增至 90%
                    const target = 90;
                    const step = Math.max(0.2, (target - progress) * 0.03);
                    progress = Math.min(target, progress + step);
                    if (bar) bar.style.width = progress.toFixed(1) + '%';
                    if (info) info.textContent = Math.round(progress) + '%';
                };
                const iv = setInterval(tick, 40);

                const hideOverlay = () => {
                    if (!overlay) return;
                    running = false;
                    clearInterval(iv);
                    clearInterval(taskIv);
                    progress = 100;
                    if (bar) bar.style.width = '100%';
                    if (info) info.textContent = '100%';
                    if (taskEl) taskEl.textContent = '就绪';
                    overlay.classList.add('lv-fade-out');
                    setTimeout(() => {
                        if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
                    }, 500);
                };

                // 使用 MutationObserver：仅当出现助手内容/导航/雷达图时才隐藏
                const container = document.querySelector('.gradio-container');
                try {
                    const observer = new MutationObserver(() => {
                        const msgs = Array.from(document.querySelectorAll('.chatbot .message'));
                        // 粗略判断是否有助手消息：带有 role-assistant 类或包含助手相关标记
                        const hasAssistant = msgs.some(el => {
                            try {
                                return el.classList.contains('role-assistant')
                                    || el.querySelector('.assistant-demo')
                                    || el.querySelector('.assistant-quiz')
                                    || el.querySelector('.info-icon.assistant-demo')
                                    || el.querySelector('.info-icon.assistant-quiz');
                            } catch { return false; }
                        });
                        const hasNav = document.querySelector('.navigation-panel');
                        const hasPlot = document.querySelector('div.plotly');
                        if (hasAssistant || hasNav || hasPlot) {
                            observer.disconnect();
                            hideOverlay();
                        }
                    });
                    if (container) observer.observe(container, { childList: true, subtree: true });
                } catch {}

                // 最多等待 18 秒自动隐藏，避免卡住
                setTimeout(hideOverlay, 18000);
                // ====== [LOADING OVERLAY - END] ======

                // ====== [CHAT JELLY STAGGER - START] 队列与弹性出现 ======
                (function(){
                    const MIN_INTERVAL = 500; // ms
                    // 结合你的页面实际结构：消息节点是 span.md.chatbot
                    const CHAT_ROOT_SELECTOR = 'body';
                    const MESSAGE_SELECTOR = '.md.chatbot, span.md.chatbot, div.md.chatbot';

                    let isLoading = true;          // 仍在加载遮罩期间
                    let queue = [];                // 先进先出队列（用于助理和初始历史）
                    let ticking = false;
                    let lastAt = 0;
                    let alternator = false;        // 当无法判定角色时，左右交替
                    const processed = new WeakSet(); // 已处理的消息节点，避免重复

                    const getChatRoots = () => {
                        const hosts = Array.from(document.querySelectorAll(CHAT_ROOT_SELECTOR));
                        if (hosts.length === 0) return [document];
                        const roots = [];
                        hosts.forEach((h) => {
                            if (h && h.shadowRoot) roots.push(h.shadowRoot);
                            roots.push(h);
                        });
                        return roots.length ? roots : [document];
                    };

                    const ensureShadowStyles = () => {
                        const css = `
/* injected by stagger */
.message.lv-hidden, .lv-hidden { opacity: 0 !important; transform: translateX(0) scale(0.98); display: inline-block; will-change: transform, opacity; }
.message.lv-jelly.role-assistant, .lv-jelly.role-assistant { animation: lvJellyRight 620ms cubic-bezier(0.175, 0.885, 0.32, 1.275) both; display: inline-block; will-change: transform, opacity; }
.message.lv-jelly.role-user, .lv-jelly.role-user { animation: lvJellyLeft 620ms cubic-bezier(0.175, 0.885, 0.32, 1.275) both; display: inline-block; will-change: transform, opacity; }
.message.lv-jelly, .lv-jelly { animation: lvJellyRight 620ms cubic-bezier(0.175, 0.885, 0.32, 1.275) both; display: inline-block; will-change: transform, opacity; }
@keyframes lvJellyRight { 0%{opacity:0;transform:translateX(14px) scale(0.92);} 45%{opacity:1;transform:translateX(-6px) scale(1.02);} 70%{transform:translateX(3px) scale(0.995);} 100%{transform:translateX(0) scale(1);} }
@keyframes lvJellyLeft { 0%{opacity:0;transform:translateX(-14px) scale(0.92);} 45%{opacity:1;transform:translateX(6px) scale(1.02);} 70%{transform:translateX(-3px) scale(0.995);} 100%{transform:translateX(0) scale(1);} }
`;
                        getChatRoots().forEach((root) => {
                            try {
                                if (!root) return;
                                const doc = root instanceof ShadowRoot ? root : root;
                                if (!doc) return;
                                if (doc.getElementById && doc.getElementById('lv-stagger-style')) return;
                                const style = document.createElement('style');
                                style.id = 'lv-stagger-style';
                                style.textContent = css;
                                (doc.head || doc).appendChild(style);
                            } catch {}
                        });
                    };

                    const queryAllInRoots = (selector) => {
                        const out = [];
                        getChatRoots().forEach((root) => {
                            try { out.push(...root.querySelectorAll(selector)); } catch {}
                        });
                        return Array.from(new Set(out));
                    };

                    const observeRoots = (cb) => {
                        getChatRoots().forEach((root) => {
                            try {
                                const obs = new MutationObserver(cb);
                                obs.observe(root, { childList: true, subtree: true });
                            } catch {}
                        });
                    };

                    const findMessageEl = (n) => {
                        if (!n || n.nodeType !== 1) return null;
                        if (n.matches && n.matches(MESSAGE_SELECTOR)) return n;
                        if (n.closest) {
                            const c = n.closest(MESSAGE_SELECTOR);
                            if (c) return c;
                        }
                        return n;
                    };

                    const detectRole = (el) => {
                        if (!el || !el.classList) return null;
                        const cls = Array.from(el.classList).join(' ').toLowerCase();
                        const attrTest = (name) => (el.getAttribute && (el.getAttribute(name)||'').toLowerCase()) || '';
                        const testids = [attrTest('data-testid'), attrTest('data-role')].join(' ');
                        const hasChild = (sel) => {
                            try { return !!el.querySelector(sel); } catch { return false; }
                        };
                        // 你的 DOM：助理消息内有 .assistant-demo 图标，用户消息内有 .user-info-icon
                        if (hasChild('.assistant-demo')) return 'assistant';
                        if (hasChild('.user-info-icon')) return 'user';
                        const textMark = (el.textContent || '').slice(0, 0); // 不使用文本判定，避免误伤
                        const has = (s) => cls.includes(s) || testids.includes(s);
                        if (has('role-user') || has(' user ') || has('human') || has('me') || has('user-message')) return 'user';
                        if (has('role-assistant') || has('assistant') || has('bot') || has('ai') || has('bot-message')) return 'assistant';
                        return null;
                    };

                    const ensureRoleClass = (el) => {
                        let role = detectRole(el);
                        if (!role) {
                            // 无法检测时，交替指定，保证方向动画存在
                            alternator = !alternator;
                            role = alternator ? 'user' : 'assistant';
                        }
                        el.classList.toggle('role-user', role === 'user');
                        el.classList.toggle('role-assistant', role === 'assistant');
                        return role;
                    };

                    const markHidden = (el) => {
                        if (!el.classList.contains('lv-seen')) {
                            el.classList.add('lv-hidden');
                        }
                    };
                    const animateReveal = (el) => {
                        try { console.debug('[stagger] reveal', el); } catch {}
                        el.classList.remove('lv-hidden');
                        // 根据角色决定方向
                        el.classList.add('lv-jelly');
                        el.classList.add('lv-seen');
                        try { processed.add(el); } catch {}
                        // 清理动画类，避免重复触发时堆积
                        el.addEventListener('animationend', function handler(){
                            el.classList.remove('lv-jelly');
                            el.removeEventListener('animationend', handler);
                        });
                    };

                    const step = () => {
                        if (ticking) return;
                        if (queue.length === 0) return;
                        const now = Date.now();
                        const wait = Math.max(0, MIN_INTERVAL - (now - lastAt));
                        ticking = true;
                        setTimeout(() => {
                            const msg = queue.shift();
                            if (msg && !msg.classList.contains('lv-seen')) animateReveal(msg);
                            lastAt = Date.now();
                            ticking = false;
                            if (queue.length > 0) step();
                        }, wait);
                    };

                    const enqueue = (msg) => {
                        if (!msg || msg.nodeType !== 1) return;
                        if (msg.classList.contains('lv-seen')) return;
                        // 不在此处使用 processed，允许同一 DOM 节点在内容更新后重新进入流程
                        ensureRoleClass(msg);
                        markHidden(msg);
                        queue.push(msg);
                        try { console.debug('[stagger] enqueue, size=', queue.length, msg); } catch {}
                        if (!isLoading) step();
                    };

                    const immediateRevealUser = (msg) => {
                        if (!msg || msg.classList.contains('lv-seen')) return;
                        // 遵守冷却：若仍在冷却期，则进入队列；否则立刻显示
                        const now = Date.now();
                        const inCooldown = (now - lastAt) < MIN_INTERVAL;
                        if (inCooldown) return enqueue(msg);
                        // 不用 processed 阻断，确保用户消息可以再次即时显示
                        ensureRoleClass(msg);
                        // 仍给出动画，但不进入队列且无延迟
                        markHidden(msg);
                        try { console.debug('[stagger] immediate user reveal'); } catch {}
                        requestAnimationFrame(() => animateReveal(msg));
                    };

                    // 扫描当前 DOM，找出尚未处理的新消息（即便不是以新增节点形式出现）
                    const scanForNew = () => {
                        const nodes = queryAllInRoots(MESSAGE_SELECTOR);
                        nodes.forEach((el) => {
                            if (!el || processed.has(el) || el.classList.contains('lv-seen')) return;
                            const role = ensureRoleClass(el);
                            const isUser = role === 'user';
                            if (isLoading) {
                                enqueue(el);
                            } else {
                                if (isUser) immediateRevealUser(el);
                                else enqueue(el);
                            }
                        });
                    };

                    // 初次扫描：在遮罩存在时不急于展示，统一进入队列
                    const bootstrapBacklog = () => {
                        const nodes = queryAllInRoots(MESSAGE_SELECTOR);
                        try { console.debug('[stagger] bootstrap size=', nodes.length); } catch {}
                        nodes.forEach((n) => enqueue(findMessageEl(n)));
                        // 不再因“存在任意消息”而隐藏遮罩，等待助手/导航/雷达图出现
                    };

                    // 监听后续新增消息
                    const watchRoots = () => {
                        // 监听：包括文本变化，兼容“同节点内容更新”
                        getChatRoots().forEach((root) => {
                            try {
                                const obs = new MutationObserver(() => scanForNew());
                                obs.observe(root, { childList: true, subtree: true, characterData: true });
                            } catch {}
                        });
                        // 监听键盘回车（发送），在短窗口内多次扫描，确保用户消息即时显示
                        window.addEventListener('keydown', (e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                const kicks = [0, 40, 120, 240, 400];
                                kicks.forEach((t) => setTimeout(scanForNew, t));
                            }
                        }, true);
                    };

                    const startWhenOverlayGone = () => {
                        const check = () => {
                            const ov = document.getElementById('lv-loading-overlay');
                            if (!ov) {
                                isLoading = false;
                                // 已入队的历史消息（含用户与助手）按队列逐条弹出
                                step();
                                return;
                            }
                            setTimeout(check, 200);
                        };
                        check();
                    };

                    // 初始化
                    ensureShadowStyles();
                    bootstrapBacklog();
                    watchRoots();
                    startWhenOverlayGone();
                })();
                // ====== [CHAT JELLY STAGGER - END] ======

                window.clickWord = function(word) {
                    console.log('=== Word click event ===');
                    const input = document.querySelector('#word-click-trigger textarea');
                    if (input) {
                        input.value = word;
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                };
                
                window.instanceInterpreter = function() {
                    console.log('=== Change model click ===');
                    const input = document.querySelector('#instance-interpreter-trigger textarea');
                    if (input) {
                        input.value = "toggle";
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                };

                window.clickNavButton = function(level) {
                    console.log('=== Nav button click event ===');
                    const input = document.querySelector('#nav-click-trigger textarea');
                    if (input) {
                        input.value = level;
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                };

                window.closeNavButton = function(level) {
                    console.log('=== Nav close event ===');
                    const input = document.querySelector('#nav-close-trigger textarea');
                    if (input) {
                        input.value = level;
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                };

                // Ensure toggleAudio is globally accessible
                window._lv_shared_audio = window._lv_shared_audio || new Audio();
                window._lv_current_button = window._lv_current_button || null;

                window.toggleAudio = function(audioPath, buttonId) {
                    const button = document.getElementById(buttonId);
                    if (!button) return;
                    const fileUrl = audioPath;
                    const audio = window._lv_shared_audio;

                    // 如果播放的是不同的文件
                    if (audio.src !== location.origin + fileUrl) {
                        audio.pause();
                        audio.currentTime = 0;
                        if (window._lv_current_button) window._lv_current_button.textContent = '▶️';
                        window._lv_current_button = button;

                        audio.src = fileUrl;
                        audio.load();
                    } else {
                        window._lv_current_button = button;
                    }

                    if (audio.paused) {
                        audio.play().then(() => {
                            button.textContent = '⏸';
                        }).catch((e) => {
                            console.warn('Audio play failed', e);
                        });
                    } else {
                        audio.pause();
                        button.textContent = '▶️';
                    }

                    audio.onended = function() {
                        if (window._lv_current_button) window._lv_current_button.textContent = '▶️';
                    };
                };

                window.triggerNextScene = function() {
                    console.log('=== Next scene triggered ===');
                    const input = document.querySelector('#next-scene-trigger textarea');
                    if (input) {
                        input.value = "next";
                        input.dispatchEvent(new Event('input', {bubbles: true}));
                        input.dispatchEvent(new Event('change', {bubbles: true}));
                    }
                };
            }
            """
        )

 

    return interface


# 启动应用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linguiverse')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration YAML file name (without extension) in config/ directory')
    args, unknown_args = parser.parse_known_args()
    unknown_kwargs = {}
    for i in range(0, len(unknown_args), 2):
        if i + 1 < len(unknown_args) and unknown_args[i].startswith('--'):
            key = unknown_args[i][2:]  # 去掉'--'前缀
            value = unknown_args[i + 1]
            unknown_kwargs[key] = value

    AUDIOS_DIR = (Path.cwd() / "generated_audios").resolve()
    gr.set_static_paths(paths=[AUDIOS_DIR])
    
    interface = create_interface(config_file=args.config, kwargs=unknown_kwargs)
    

    server, local_url, share_url = interface.launch(
        server_name="127.0.0.1",
        server_port=8760,
        prevent_thread_lock=True, # 非阻塞，以保证能访问app相关信息
        show_error=True,
        show_api=True, # 开通通过FASTAPI发起的url requests
        allowed_paths=[AUDIOS_DIR]
    )
    # 打印可访问地址
    try:
        print(f"[Gradio] Local: {local_url}", flush=True)
        if share_url:
            print(f"[Gradio] Public: {share_url}", flush=True)
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)  # 保持主线程活着
    except KeyboardInterrupt:
        print("退出程序")