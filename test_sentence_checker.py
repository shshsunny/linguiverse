from modules.sentence_checker import sentence_checker
from test import get_test
from utils import read_apikey

from pprint import pprint
import time

# NOTE:
# This file serves as an integration/demo runner for `sentence_checker()`.
# It calls a REAL LLM endpoint via `modules/sentence_checker.py`, so outputs are non-deterministic
# and depend on network, model, and prompts. Use it to manually inspect behavior rather than as a
# strict unit test. For deterministic unit tests, see `test_sentence_checker_unit.py` (pytest).
#
# What to look for in the console output:
# - "origin result" shows the skill table before update.
# - "final result" includes:
#   * issues/advice/improved: LLM textual feedback (non-deterministic)
#   * updates: list of (table, uid, content, score) for detected knowledge points
#   * quality_score: average score over ALL skills of current scene (updates only cover detected ones)
# - You can trim the dialogue length via `context = context[:N]` below to probe different stages.
#
# Expected effects when running:
# - The function writes the constructed prompt into `test_output.txt` for inspection.
# - The returned dict must include keys: issues/advice/improved/updates/quality_score.
# - The in-memory `skill` DataFrame gets updated by EMA: new = 0.3*old + 0.7*score (clamped to [0,1]).

def _clamp(x, lo=0.0, hi=1.0):
    return max(min(x, hi), lo)

def _get_prof(skill, table, uid):
    rows = skill[(skill['table'] == table) & (skill['uid'] == uid)]['proficiency']
    return None if rows.empty else float(rows.iloc[0])

def _set_prof_debug_string(table, uid, content):
    return f"[{table}:{uid}] {content}"

if __name__ == '__main__':
    apikey = read_apikey("config/api.key")
    
    testcase = get_test('school.txt', load_vocab_emb=False)
        
    scenarios = testcase['scenarios']
    vocab = testcase['vocab']
    grammar = testcase['grammar']
    
    world = testcase['world']
    skill = testcase['skill']
    progress = testcase['progress']

    # context = [
    #     {'role': 'assistant', 'content': 'Yes, do you know why I say a cup of coffee?', 'info': {'table': 'vocab', 'uid': 2, 'type': 'quiz'}},

    #     {'role': 'user', 'content': 'Because a means single, and a cup means a single cup, it\'s kind of modifier.'},
    #     {'role': 'assistant', 'content': 'Yes, do you want to try more? I have too cups of new coffee, the white one and the dark one. Which one do you want to try first?', 'info': {'table': 'grammar', 'uid': 1, 'type': 'demo'}},

    #     {'role': 'user', 'content': 'I want to try the white one.'},
    #     {'role': 'assistant', 'content': 'What can you say if you want both?', 'info': {'table': 'grammar', 'uid': 1, 'type': 'quiz'}},

    #     {'role': 'user', 'content': 'I love to eat the French and Italian dish.'},
    #     {'role': 'assistant', 'content': 'Yes, let\'s begin.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},

    # ]
    
    # 1,  2,  3,      4,      5,     6,      7,   8,   9        grammar1
    # Mr, a, café, coffee, station, begin, start, hi, hello       and
    context = [
        {'role': 'user', 'content': ''},
        {'role': 'assistant', 'content': 'Hello, student.', 'info': {'table': 'vocab', 'uid': 9, 'type': 'demo'}},

        {'role': 'user', 'content': 'Hello, Mr. Shelton!'},
        {'role': 'assistant', 'content': 'Hi, what a nice day!', 'info': {'table': 'vocab', 'uid': 8, 'type': 'demo'}},

        # hello +
        
        {'role': 'user', 'content': 'Yeah, it\'s sunny and I want to go outside.'},
        {'role': 'assistant', 'content': 'Good idea, what about going to the café', 'info': {'table': 'vocab', 'uid': 3, 'type': 'demo'}},

        {'role': 'user', 'content': 'café? What is café?'},
        {'role': 'assistant', 'content': 'Café is a place where you can get coffee.', 'info': {'table': 'vocab', 'uid': 3, 'type': 'demo'}},
        
        # café stay same

        {'role': 'user', 'content': 'Then, what is coffee?'},
        {'role': 'assistant', 'content': 'Coffee is a drink made from coffee beans.', 'info': {'table': 'vocab', 'uid': 4, 'type': 'demo'}},

        {'role': 'user', 'content': 'Ok, i see, so we can go to the café and have some cups of coffee together.'},
        {'role': 'assistant', 'content': 'Good choice, what about taking the bus at the station? ', 'info': {'table': 'vocab', 'uid': 5, 'type': 'demo'}},

        {'role': 'user', 'content': 'Ok, let\'s go!'},
        {'role': 'assistant', 'content': 'Ok, we\'ve arrived, where are we now?', 'info': {'table': 'vocab', 'uid': 5, 'type': 'quiz'}},

        {'role': 'user', 'content': 'We are at the station?'},
        {'role': 'assistant', 'content': 'Yes, and now we are going to?', 'info': {'table': 'vocab', 'uid': 3, 'type': 'quiz'}},

        {'role': 'user', 'content': 'The café, and we are going to have some cups of coffee.'},
        {'role': 'assistant', 'content': 'Yes, we can have some cups of coffee. Do you know what is coffee?', 'info': {'table': 'vocab', 'uid': 4, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Yes, coffee is a drink made from coffee beans.'},
        {'role': 'assistant', 'content': 'Brilliant, let\'s begin out journey in the café', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},

        {'role': 'user', 'content': 'What is begin?'},
        {'role': 'assistant', 'content': 'Begin is the first step in a journey.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},

        {'role': 'user', 'content': 'Ok, let\'s begin have some coffee.'},
        {'role': 'assistant', 'content': 'Hi, I\'m Mr. Brown, a waiter in the café', 'info': {'table': 'vocab', 'uid': 1, 'type': 'demo'}},

        {'role': 'user', 'content': 'Hi, Mr. Brown! I\'d like to have a cup of coffee.'},
        {'role': 'assistant', 'content': 'Hi kid. Do you know why I am a Mr?', 'info': {'table': 'vocab', 'uid': 1, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Because you are a man? So I shall call you a Mr?'},
        {'role': 'assistant', 'content': 'Yes, and do you know what Hi means?', 'info': {'table': 'vocab', 'uid': 8, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Hi is like hello, it\'s a kind of greeting words'},
        {'role': 'assistant', 'content': 'Good, let me serve you a cup of coffee', 'info': {'table': 'vocab', 'uid': 2, 'type': 'demo'}},

        {'role': 'user', 'content': 'Thank you, the coffee is good.'},
        {'role': 'assistant', 'content': 'Yes, do you know why I say a cup of coffee?', 'info': {'table': 'vocab', 'uid': 2, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Because a means single, and a cup means a single cup, it\'s kind of modifier.'},
        {'role': 'assistant', 'content': 'Yes, do you want to try more? I have too cups of new coffee, the white one and the dark one. Which one do you want to try first?', 'info': {'table': 'grammar', 'uid': 1, 'type': 'demo'}},

        {'role': 'user', 'content': 'I want to try the white one.'},
        {'role': 'assistant', 'content': 'What can you say if you want both?', 'info': {'table': 'grammar', 'uid': 1, 'type': 'quiz'}},

        {'role': 'user', 'content': 'I want to have the white and black.'},
        {'role': 'assistant', 'content': 'Yes, let\'s begin.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},

        {'role': 'user', 'content': 'They taste good.'},
        {'role': 'assistant', 'content': 'Do you know what begin means?', 'info': {'table': 'vocab', 'uid': 6, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Yes, begin means completion of a task.'},
        {'role': 'assistant', 'content': 'Not exactly right, begin means the first step in a journey.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},
        {'role': 'assistant', 'content': 'Do you know what start means?', 'info': {'table': 'vocab', 'uid': 7, 'type': 'quiz'}},

        {'role': 'user', 'content': 'I know start is almost the same as begin, which means the first step.'},
        {'role': 'assistant', 'content': 'Exactly, take this coffee, and say something using begin', 'info': {'table': 'vocab', 'uid': 6, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Let\'s begin enjoying our coffee!'},
        {'role': 'assistant', 'content': 'You\'ve learned all the key words and grammar points today. Well done!', 'info': {'table': None, 'uid': None, 'type': 'hooray'}},
    ]   
    
    LOG_PATH = "test_sentence_checker_run.txt"
    with open(LOG_PATH, 'w', encoding='utf-8') as log:
        context = context[:]  # use full conversation
        # 仅遍历从第一个 user 开始的用户轮（步长 +2），保证 LATEST REPLY 是 user
        user_indices = [idx for idx, turn in enumerate(context) if turn.get('role') == 'user']
        for user_idx in user_indices:
            i = user_idx + 1  # 用于日志标记的“长度”，与之前的 i 语义一致（context[:i]）
            _context = context[:i]

            # 打印一个简短窗口：上一个 assistant 与本次 user（如果存在）
            log.write("DIALOGUE:\n")
            start = max(0, user_idx - 1)
            pprint(context[start:user_idx+1], stream=log)

            # snapshot old proficiencies for all skills to validate EMA update per-turn
            old_profs = {}
            for _, row in skill.iterrows():
                old_profs[(row['table'], int(row['uid']))] = float(row['proficiency'])
            log.write(f"[CALL] i={i}\n")
            log.flush()
            try:
                result = sentence_checker(_context, scenarios, vocab, grammar, world, skill, progress, apikey)
                # dump internal debug messages to log
                dbg = result.get('debug_msgs', []) or []
                if dbg:
                    log.write("[DEBUG]\n")
                    for line in dbg:
                        log.write(line + "\n")
                    log.write("[DEBUG-END]\n")
                updates = result.get('updates', [])

                # Validate quality_score = sum(scores in updates)/total_skills
                total_skills = len(skill)
                sum_scores = sum(u[3] for u in updates) if updates else 0.0
                expected_qs = (sum_scores / total_skills) if total_skills else None
                if expected_qs is not None:
                    log.write(f"quality_score -> expected:{expected_qs:.6f}, actual:{result.get('quality_score')}\n")
                if result.get('quality_score') is not None:
                    assert abs(result['quality_score'] - expected_qs) < 1e-6, "quality_score mismatch"

                # Validate each updated skill's EMA and print delta direction
                for table, uid, content, score in updates:
                    uid = int(uid)
                    before = old_profs.get((table, uid), 0.0)
                    expected_after = _clamp(0.3 * before + 0.7 * float(score))
                    after = _get_prof(skill, table, uid)
                    tag = _set_prof_debug_string(table, uid, content)
                    log.write(f"update {tag}: before={before:.6f}, score={score:.6f}, after={after:.6f}, expected_after={expected_after:.6f}\n")
                    assert after is not None
                    assert abs(after - expected_after) < 1e-6, f"EMA update mismatch for {tag}"
                    # Direction check: sign of delta matches score-old
                    delta = after - before
                    trend = 'up' if delta > 1e-9 else ('down' if delta < -1e-9 else 'flat')
                    expected_trend = 'up' if score > before else ('down' if score < before else 'flat')
                    log.write(f"  trend: {trend} (expected {expected_trend})\n")
                    assert trend == expected_trend, f"Trend mismatch for {tag}"
                log.write("------------------------------------------------\n")
                log.write(f"[DONE] i={i}\n")
            except Exception as e:
                log.write(f"[ERROR] i={i}: {repr(e)}\n")
                log.flush()
                raise
            log.flush()
            time.sleep(0.7)

        # 所有轮次处理完毕后，输出最终的 skill 表（dict/表格）
        log.write("\n[FINAL SKILL]\n")
        try:
            log.write(skill.to_string(index=False) + "\n")
        except Exception:
            # 回退为打印 records
            pprint(skill.to_dict(orient='records'), stream=log)