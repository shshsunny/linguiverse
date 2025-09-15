from modules.world_processor import world_processor
from test import get_test
from utils import read_apikey
if __name__ == '__main__':
    testcase = get_test('school.txt')
    context = [ # 现编会话上下文
        {'role': 'user', 'content': 'I\'m staying behind to chat with Miss Blackwood. Dear Miss Blackwood, when did you start teaching? How many students have you taught so far?'},
        {'role': 'assistant', 'content': 'I started teaching five years ago. I have taught over a hundred students since then. Many are 16-year-old just like you."'},
    ]
    apikey = read_apikey("config/api.key")
    world = world_processor(context, api_key=apikey, **testcase)