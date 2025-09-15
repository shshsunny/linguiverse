import os, json
from copy import deepcopy
import threading
from pprint import pprint
# global constants

# 每个知识点的最初和最终掌握度目标
INITIAL_PROFICIENCY_GOAL = 0.3 # 在回应提示器中会迭代地调高掌握度目标，直到达到较理想的值
FINAL_PROFICIENCY_GOAL = 0.75


# ------------------ configuration ------------------ #
def read_apikey(file_path):
    """
    Usage:
        从指定文件中读取API密钥
    Args:
        file_path : string
    Returns:
        api_key
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"System Error: Not Found ({file_path})")
            return None
        
        # 读取文件内容
        with open(file_path, "r") as file:
            api_key = file.read().strip()   # 移除首尾空白字符
        
        print(f"Successfully load apikey from {file_path}")
        return api_key
    
    except Exception as e:
        print(f"Read Error: {str(e)}")
        return None



# ------------------ global state management ------------------ #

class StateDict(dict):
    """
    一个支持点号(.)访问和修改键的字典类，其他行为与标准dict完全一致。
    """
    def __init__(self, *args, **kwargs):
        """初始化AttrDict，支持与dict相同的初始化方式"""
        super().__init__(*args, **kwargs)
        
        # 递归地将嵌套的字典转换为AttrDict
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, StateDict):
                self[key] = StateDict(value)
    def __getattr__(self, key):
        """通过点号访问获取键值"""
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'") from e

    def __setattr__(self, key, value):
        """通过点号访问设置键值"""
        # 如果值是字典，转换为AttrDict以支持嵌套点号访问
        if isinstance(value, dict) and not isinstance(value, StateDict):
            value = StateDict(value)
        self[key] = value

    def __delattr__(self, key):
        """通过点号访问删除键"""
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'") from e

    def __getstate__(self):
        """用于序列化支持"""
        return self.__dict__

    def __setstate__(self, state):
        """用于反序列化支持"""
        self.__dict__.update(state)

    def update(self, *args, **kwargs):
        """重写update方法，确保新增的字典项被转换为AttrDict"""
        # 处理位置参数
        if args:
            if len(args) > 1:
                raise TypeError(f"update expected at most 1 argument, got {len(args)}")
            other = args[0]
            if isinstance(other, dict):
                for key, value in other.items():
                    self[key] = StateDict(value) if isinstance(value, dict) and not isinstance(value, StateDict) else value
            else:
                for key, value in other:
                    self[key] = StateDict(value) if isinstance(value, dict) and not isinstance(value, StateDict) else value
        
        # 处理关键字参数
        for key, value in kwargs.items():
            self[key] = StateDict(value) if isinstance(value, dict) and not isinstance(value, StateDict) else value

    def copy(self):
        """重写copy方法，返回AttrDict实例而非普通dict"""
        return StateDict(super().copy())

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """重写fromkeys方法，返回AttrDict实例"""
        return cls(dict.fromkeys(iterable, value))
    

# ------------------ context management ------------------ #

"""
def save_context(context, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(context, file, ensure_ascii=False, indent=2)
        return True
    
    except Exception as e:
        print(f"History Saving Error: {str(e)}")
        print("context:", end=' ')
        pprint(context)
        return False
"""

context_lock = threading.Lock()

def update_context(context, role, message, info=dict()):
    """
    Usage:
        添加消息到对话历史
    Args:
        history : dict
        message : string
        info     : dict, optional, 仅对于系统角色的消息添加
    Returns:
        history : dict
    """
    assert not (role != 'assistant' and info), f"Only system output can have additional info: {role}, {info}, {message}. context: {context}"
    
    with context_lock: # NOTE: 主程序和audio_generator可能同时操作context，需确保线程安全
        if role == 'assistant':
            context.append({
                "role": role,
                "content": message,
                "info": info
            })
        else:
            context.append({
                "role": role,
                "content": message
            })
    return context


def update_context_info(context_item, info):
    with context_lock:
        if not context_item.get('info'):
            context_item['info'] = dict()
        context_item['info'].update(info)
    return context_item


PROMPT_PREFIX = "[*] "

def get_context_without_prompt(context):
    # 获取真实对话内容，滤除prompt
    # 为方便起见，假设消息为prompt <====> role为user且content以[*] 开头
    filtered_context = []
    for item in context:
        if item['role'] == 'assistant' or (item['role'] == 'user' and not item['content'].startswith(PROMPT_PREFIX)):
            filtered_context.append(deepcopy(item))
    return filtered_context

def get_context_assistant_only(context):
    # 获取仅包含系统回复的对话内容，不含prompt，并且滤除
    assistant_context = []
    for item in context:
        if item['role'] == 'assistant' and isinstance(item['content'], str) \
            and not item['content'].startswith(PROMPT_PREFIX):
            assistant_context.append(deepcopy(item))
    return assistant_context

def get_context_without_info(context):
    # 获取对话内容，滤除info字段
    filtered_context = []
    for item in context:
        if 'info' in item:
            item_copy = deepcopy(item)
            del item_copy['info']  # 删除info字段
            filtered_context.append(item_copy)
        else:
            filtered_context.append(deepcopy(item))
    return filtered_context

def get_context_without_media(context):
    # 获取对话内容，滤除所有媒体消息（媒体消息的content为媒体路径列表）
    filtered_context = []
    for item in context:
        if isinstance(item['content'], str) and (not item.get('info') or item['info'].get('type') != 'instance_explanation'):
            filtered_context.append(deepcopy(item))
    return filtered_context


def get_context_pure_dialogue(context):
    # 得到滤除prompt, media, info的对话内容，以便直接用于模型输入
    context = get_context_without_prompt(context)
    context = get_context_without_media(context)
    context = get_context_without_info(context)
    return context

def format_attended_context(context, world, att_len=6):
    # 将真实对话内容（滤除prompt和media消息）的最近最多att_len条格式化为字符串
    attended_context = get_context_without_media(get_context_without_prompt(context))
    if att_len is not None: attended_context = attended_context[-att_len:]
    role_name = {
        'assistant': world['meta']['system_role'],
        'user': world['meta']['user_role']
    }
    text = "\n".join([f"{role_name[turn['role']]}: {repr(turn['content'])}" for turn in attended_context])
    return text



def get_response(client, model, context, **kwargs):
    context = get_context_without_info(context)
    default = dict(
        temperature=1.0, # 控制回复的随机性
        top_p=1.0        # 控制多样性的核心采样
    )
    default.update(kwargs)
    return client.chat.completions.create(
                model=model, 
                messages=context,
                **default        
            ).choices[0].message.content

# ------------------ metadata management ------------------ #
def get_metadata_item(info, vocab, grammar):
    '''
    根据table和uid信息提取metadata表项
    '''
    meta = vocab if info['table'] == 'vocab' else grammar
    items = meta[meta['uid'] == info['uid']]
    if not items.empty:
        return items.iloc[0]
    else:
        return dict()
    
# ------------------ debug ------------------ #

def len_context(context):
    return sum(len(item['content']) if isinstance(item['content'], str) else 0 for item in context)