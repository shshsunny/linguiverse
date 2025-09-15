  #=======note========
# 原main.py已改名为terminal.py，通过前端访问主程序应执行streamlit run app.py
# 此处将terminal.py中的逻辑模块化，以便前端调用

import os
import json, yaml
from openai import OpenAI

from utils import *
from modules import *
from copy import deepcopy
from pprint import pprint
from functools import partial
import time
from time import perf_counter  # ==== CASCADE PATCH: profiling ====
import logging
import emoji
import threading


# ==== CASCADE PATCH: logging setup ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ==== CASCADE PATCH END ====

AUDIO_MODEL = "PiperTTS" # alternatives: "Kokoro" (deprecated), None

if AUDIO_MODEL == 'PiperTTS':
    from modules.audiogen_pipertts import AudioGenerator



def generate_and_play_audio(state, text, callback):
    if AUDIO_MODEL and state.audio_generator:
        # 预处理：去除text中的EMOJI，以免被朗读
        text = ''.join([char for char in text if not emoji.is_emoji(char)])

        # 先暂停可能正在进行的流式音频播放或完整音频播放
        state.audio_generator.stop_stream_player()
        state.audio_generator.audio_player.pause()

        state.audio_generator.async_generate_and_play_audio(
            text=text,
            save_filename=f"{state.target_lang}-{int(time.time())}",
            play_audio=True,
            first_chunk_time_callback=None,
            language=state.target_lang,
            callback=callback
        )

def update_audio_path(audio_path, context_item):
    """回调函数：更新context的info字典中的audio_path"""
    if audio_path:
        update_context_info(context_item, {'audio_path': audio_path})
        print(f"[INFO] Audio path added to context: {audio_path}")
        # pprint(context_item)


def init_state(state, config_file='default', kwargs=dict()):
    if not config_file:
        config_file = 'default'
    with open(f'config/default.yaml', 'r', encoding='utf-8') as f:
        default_config = yaml.safe_load(f)
    if config_file == 'default':
        config = default_config
    else:
        with open(f'config/{config_file}.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            default_config.update(config)
            config = default_config
    
    config.update(kwargs) # 若命令行有特殊指定的参数，则覆盖yaml配置文件中的参数值
    target_lang = config.get('target_lang', 'english')
    # 注意：source_language保存在world archive中，而不是单独指定；
    # target_lang直接关联一整套world, skill, progress存档文件
    #debugging = config.get('debugging', True)
    #debug_scene = config.get('debug_scene', 5)
    apikey_path = "config/api.key"
    api_key = read_apikey(apikey_path)


    ############# initialize modules #############
    # init data manager
    data_manager = DataManager(target_lang, debugging=False)

    # init audio generator
    audio_generator = None
    if AUDIO_MODEL:
        audio_generator = AudioGenerator(model_cache_dir=config['model_cache_dir'])
        audio_generator.init_tts_pipeline(language=target_lang)


    init_sentence_checker()
    init_response_prompter()
    init_world_processor()

    state.update({
        # global configs
        'target_lang': target_lang,
        'apikey_path': apikey_path,

        # LLM API
        'api_key': api_key,
        'model': config['text_model'],
        'flux_model': config['flux_model'],
        'client': OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        'model_cache_dir': config['model_cache_dir'],
    
        'data_manager': data_manager,
        'dialogue_chain': [],
        'active_dialogue_level': 0,

        'archive_lock': threading.Lock(),  # 用于多线程访问archive时的锁

        # audio generator
        'audio_generator': audio_generator,
    })

    # initialize configuration / pipeline for the instance interpreter
    state.flux_cfg = init_flux_pipeline(state.api_key, config['flux_model'])
    # state.sd_pipeline = init_sd_pipeline(model_name='stabilityai/stable-diffusion-xl-base-1.0')
    #state.sd_pipeline = init_sd_pipeline(model_name='stabilityai/stable-diffusion-2-1')
    #state.sd_pipeline = init_sd_pipeline(model_name="CompVis/stable-diffusion-v1-4")


def init_main_dialogue(state):
    world = state.data_manager.get_archive()['world']
    context = [
        {
            "role": "system", 
            "content": f"You are {world['meta']['system_role']}, who is the language mentor of "
                    f"{world['meta']['user_role']} (the user) in a virtual world. "
                    "Your mission is to help them master proposed language points "
                    "while building you two's story in this world throughout the conversation. "
                    "The narration genre is like: " + ", ".join(world['meta']['preferences']['world_genre']) + ". "
                +   "And the story background is as follows: " + world['meta']['preferences']['world_prologue']
                +   f"Inputs starting with {PROMPT_PREFIX} are inserted prompts on adjusting your response. "
                    f"ALWAYS respond as {world['meta']['system_role']} in {world['meta']['preferences']['target_language']} like chatting. "
                    f"Try to be logically coherent. "
        },

        {
            "role": "user",
            "content": PROMPT_PREFIX + "start your conversation in no more than 20 words. "
        }
    ]

    # 如果保存了历史对话，则追加该scene中已有的对话内容，并且不让系统作出新的回应
    if old_context := state.data_manager.load_history():
        context.extend(get_context_without_prompt(old_context)) # 获取不含prompt的会话，但仍保留info（涉及音频等信息）
    
    else:
        # TODO: 让系统输出真实的第一条消息，基于当前world.static和world.dynamic，更有时效性

        # ==== CASCADE PATCH START: profiling first LLM call ====
        _t0 = perf_counter()
        first_response = get_response(state.client, state.model, context, temperature=0.7, top_p=0.8)
        
        logger.info(f"[PROFILE] init_main_dialogue.get_response: {(perf_counter()-_t0)*1000:.1f} ms")
        # ==== CASCADE PATCH END ====
        update_context(context, 'assistant', first_response, info={
            'table': None,
            'uid': None,
            'type': 'any'
        })

        # generate audio async
        generate_and_play_audio(state, first_response, partial(update_audio_path, context_item=context[-1]))

    main_dialogue = {
        "level": 0,
        "keyword": None,
        "history": context
    }

    if len(state.dialogue_chain) == 0: # 这是第一次调用init_main_dialogue，即程序初始化时
        state.dialogue_chain.append(main_dialogue)
        state.active_dialogue_level = 0
    else:
        state.dialogue_chain[0] = main_dialogue # 这是从前端重新加载主会话时调用的，替换掉主会话
        # 不更新active_dialogue_level，因为只重新加载主会话，不管子会话
    
    print('[INFO] main agent initialized')

def init_subdialogue(state, context, level, subject):
    """
    context: 继承自父会话的上下文
    level: 新建子对话的深度
    subject: 被点击的关键词，或知识点的info dict, 用于进行针对性讲解
        若直接点击文字，则是str
        也可传入dict(table, uid, content))
    """
    context = deepcopy(context)
    
    # responser 构建一下
    responser = {
        'client': state.client,
        'model': state.model
    }
    
    data_manager = state.data_manager
    
    sub_context = init_subdialogue_context(
            root_context=context,
            responser=responser,
            vocab=data_manager.vocab,
            grammar=data_manager.grammar,
            world=data_manager.world,
            skill=data_manager.skill,
            subject=subject
        )
    
    context_item = sub_context[-1]
    generate_and_play_audio(state, context_item['content'], partial(update_audio_path, context_item=context_item))  

    state.dialogue_chain.append({
        "level": level,
        "keyword": subject,
        "history": sub_context
    })
    state.active_dialogue_level = level
    print(f'[INFO] sub agent, level: {level} initialized')
        
    
def handle_user_input(state, user_input, level, sub_interpreter='interactive', append_user=True):
    """
    将最新用户输入加入指定level的会话，并生成系统回复
    目前认为level是会话的唯一ID
    """
    assert level in range(0, len(state.dialogue_chain)), "Invalid dialogue level."
    data_manager = state.data_manager
    data = {**data_manager.get_metadata(), **data_manager.get_archive()}
    scenarios, vocab, grammar, vocab_emb, world, skill, progress = \
        data['scenarios'], data['vocab'], data['grammar'], data['vocab_emb'], \
        data['world'], data['skill'], data['progress']
    api_key = state.api_key
    client = state.client
    model = state.model
    responser = {
        'client': client,
        'model': model
    }

    context = state.dialogue_chain[level]['history']
    print("[INFO] processing at dialogue level:", level)
    print("[INFO] user input:", user_input)
    # NOTE: update_context有副作用，直接修改context
    # 当采用前端流式立即回显用户消息时，可将 append_user 设为 False，避免重复添加
    if (
        append_user
        and user_input is not None
        and not (0 < level < len(state.dialogue_chain) and sub_interpreter == 'instance')
    ):
        update_context(context, 'user', user_input)

   

    if level == 0: # 主会话agent

        world_processor_context = deepcopy(context)
        # NOTE: 为并行化，world_processor不再等main agent完成后再处理current input + current response，
        # 而是处理last response + current input

        # world_processor处理对话历史，更新虚拟世界存档
        def async_world_processor():
            _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
            world_processor(world_processor_context, responser, world, archive_lock=state.archive_lock)
            logger.info(f"[PROFILE] world_processor: {(perf_counter()-_t0)*1000:.1f} ms")
            if data_manager.save_archives():
                print(f"[INFO] world archive updated: {len(data_manager.world['static'])} static sentences, {len(data_manager.world['dynamic'])} dynamic sentences")

        threading.Thread(target=async_world_processor, daemon=True).start()
        
        # 1. sentence_checker检查用户语句、更新语言能力存档（为宽容子会话中用户的表达，只在主会话中更新掌握度）
        _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
        check_info = sentence_checker(context, responser, scenarios, vocab, grammar, world, skill, progress, archive_lock=state.archive_lock)
        logger.info(f"[PROFILE] sentence_checker: {(perf_counter()-_t0)*1000:.1f} ms")
        print("[INFO] sentence checking results:")
        pprint(check_info)
        # 将check_info中对当前用户输入的评估添加到context中
        assert context[-1]['role'] == 'user', "Last message should be user input."
        context[-1]['info'] = check_info # 包含issues, improved，以及其他checking相关信息

        # 2. response_prompter提示语句状态，主会话agent生成回应
        _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
        advice = response_prompter(context, responser, vocab, grammar, vocab_emb, world, skill, progress, archive_lock=state.archive_lock)
        logger.info(f"[PROFILE] response_prompter: {(perf_counter()-_t0)*1000:.1f} ms")
        prompt = PROMPT_PREFIX + ' '.join([ # 只在demo/quiz时插入建议，response_prompter在非demo/quiz时不给建议
            "Advice:", advice['content_advice'],
            advice['expression_advice'], check_info['advice'] if advice['info']['type'] in ('demo', 'quiz') else ''
        ]) + (
            'So long as possible, '
            + advice['task'] + advice['knowledge']
        ) + (
            "Enclose the usage of the knowledge in your reply with ⟪⟫. GIVE PURE TEXT WITHOUT ANY FORMATS. "
            if advice['knowledge'] else ""
        )


        print("[INFO] advice prompt for main agent:", prompt)


        # 只要有一次sentence_type为hooray，就要设为已完成
        #print("[DEBUG] advice:")
        #pprint(advice)
        if advice['info']['type'] == 'hooray':
            print("[INFO] (handle_user_input) current scene completed")
            progress['scene_completed'] = True


        # NOTE: system型消息仅支持在context开头提供。
        # 用于引导agent进行针对性回应的prompt采用[*]标记，消息类型也为user
        update_context(context, "user", prompt)

        print("[INFO] main agent thinking ...", end="", flush=True)
        _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
        output = get_response(client, model, context)
        clen = len_context(context)
        logger.info(f"[PROFILE] main.get_response: {(perf_counter()-_t0)*1000:.1f} ms, clen: {clen}")
        print(f"\n{output}\n")

        update_context(context, "assistant", output, info={**advice['info'], 'audio_path': None})  
        
        if data_manager.save_history(context):
            print(f"[INFO] context saved: {len(context)} messages")
    

        # generate audio async (将此放在即将更新显示之时，以保证音频与显示效果同步)
        generate_and_play_audio(state, output, partial(update_audio_path, context_item=context[-1]))
    
    else: # 子会话agent
        
        current_dialogue = state.dialogue_chain[level]
        keyword = current_dialogue['keyword']
        
        # responser 构建一下
        responser = {
            'client': state.client,
            'model': state.model
        }
        
        if sub_interpreter == 'instance':
            print("[INFO] instance interpreter")
            _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
            case_result = instance_interpreter(
                context=context,
                responser=responser,
                scenarios=data_manager.scenarios,
                vocab=data_manager.vocab,
                grammar=data_manager.grammar,
                vocab_emb=data_manager.vocab_emb,
                world=data_manager.world,
                skill=data_manager.skill,
                progress=data_manager.progress,
                #sd_pipeline = state.get('sd_pipeline', None),
                flux_cfg=state['flux_cfg'],
                subject=keyword
            )
            logger.info(f"[PROFILE] instance_interpreter: {(perf_counter()-_t0)/60:.2f} min")
            pprint(case_result)
            new_context = case_result['content']
            update_context(context, "assistant", new_context, info=case_result['info'])
            generate_and_play_audio(state, case_result['info']['example'], partial(update_audio_path, context_item=context[-1]))
            
        else: # 'interactive'
            print("[INFO] interactive interpreter")
            _t0 = perf_counter()  # ==== CASCADE PATCH: profiling ====
            case_result = interactive_interpreter(
                context=context,
                responser=responser,
            )
            logger.info(f"[PROFILE] interactive_interpreter: {(perf_counter()-_t0)*1000:.1f} ms")
            new_context = case_result['content']
            update_context(context, "assistant", new_context)

            # generate audio async
            generate_and_play_audio(state, new_context, partial(update_audio_path, context_item=context[-1]))
    

def switch_dialogue(state, level):
    """
    切换当前活跃的对话
    """
    if 0 < level < len(state.dialogue_chain):
        state.active_dialogue_level = level
    else:
        print(f"[WARN] Invalid level {level} for switching.")


def close_dialogue(state, level):
    """
    关闭指定层级的子对话
    不显示Main按钮，因此level=0的对话不能被关闭，默认level>0
    """
    if 0 < level < len(state.dialogue_chain):
        del state.dialogue_chain[level]
        # 更新被关闭对话之后的所有对话的level(index)
        for i in range(level, len(state.dialogue_chain)):
            state.dialogue_chain[i]['level'] = i

        # 如果关闭的是当前活跃对话，则将活跃对话切换到上一级
        if state.active_dialogue_level == level:
            state.active_dialogue_level = max(0, level - 1) # 此值若=0，表示没有子会话
        # 如果关闭的对话级别在当前活跃对话之前，也需要更新
        elif state.active_dialogue_level > level:
            state.active_dialogue_level -= 1

        print(f"[INFO] Dialogue level {level} closed. New active level: {state.active_dialogue_level}")
    else:
        print(f"[WARN] Invalid level {level} for closing.")

# 无用接口已暂时删除