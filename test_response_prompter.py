from modules.response_prompter import response_prompter
from test import get_test
from utils import read_apikey
import torch
if __name__ == '__main__':
    testcase = get_test('school.txt', load_vocab_emb=True)
    """context = [
        {'role': 'user', 'content': "I'm staying behind to chat with Miss Blackwood. Dear Miss Blackwood, when did you start teaching? How many students have you taught so far?"},
        {'role': 'assistant', 'content': "I started teaching five years ago. I have taught over a hundred students since then. Many are 16-year-old just like you."},
        {'role': 'user', 'content': "That is a lot. What do you like most about teaching English?"},
        {'role': 'assistant', 'content': "I like to see students speak with confidence. I also enjoy correcting grammar and helping with new words."},
        {'role': 'user', 'content': "Can you tell me more about these new words? How do you choose them?"},
        {'role': 'assistant', 'content': "I choose words that match your level and interests. Today, we learned words about school and daily routines."},
        {'role': 'user', 'content': "Yes, we learned 'classroom', 'library', and 'dormitory'. I like the word 'dormitory'."},
        {'role': 'assistant', 'content': "Good choice. 'Dormitory' is a noun. You use 'the dormitory' with the definite article."},
        {'role': 'user', 'content': "I understand. Can you give me a sentence with 'dormitory'?"},
        {'role': 'assistant', 'content': "Sure. 'The students return to the dormitory at six o'clock.'"},
        {'role': 'user', 'content': "Thank you. May I ask about the next lesson? What will we study tomorrow?"},
        {'role': 'assistant', 'content': "Tomorrow we will practice present continuous tense and talk about activities."},
        {'role': 'user', 'content': "I like that. I want to say 'I am studying English now.' Is that correct?"},
        {'role': 'assistant', 'content': "Very good. That is correct present continuous: 'I am studying English now.'"},
        {'role': 'user', 'content': "Great! Can we also talk about the big window in the classroom?"}
    ]"""

    context = [
        {'role': 'user', 'content': 'Hello, Miss Blackwood!'},
        {'role': 'assistant', 'content': 'Hello, Elinor!', 'info': {'table': 'vocab', 'uid': 9, 'type': 'demo'}},

        {'role': 'user', 'content': 'It is nice to see you again.'},
        {'role': 'assistant', 'content': 'Hi! You look happy today.', 'info': {'table': 'vocab', 'uid': 8, 'type': 'demo'}},

        {'role': 'user', 'content': 'Yes, I like this class.'},
        {'role': 'assistant', 'content': 'Hi, Mr Thompson is also in class today.', 'info': {'table': 'vocab', 'uid': 8, 'type': 'quiz'}},

        {'role': 'user', 'content': 'He sits near the door.'},
        {'role': 'assistant', 'content': 'Hello, Mr Thompson!', 'info': {'table': 'vocab', 'uid': 1, 'type': 'demo'}},

        {'role': 'user', 'content': 'He is reading a book.'},
        {'role': 'assistant', 'content': 'Mr Green is in the library.', 'info': {'table': 'vocab', 'uid': 1, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Can we start now?'},
        {'role': 'assistant', 'content': 'Yes, let’s start the lesson.', 'info': {'table': 'vocab', 'uid': 7, 'type': 'demo'}},

        {'role': 'user', 'content': 'We start with listening?'},
        {'role': 'assistant', 'content': 'Yes, we start with a short story.', 'info': {'table': 'vocab', 'uid': 7, 'type': 'quiz'}},

        {'role': 'user', 'content': 'OK, I’m ready.'},
        {'role': 'assistant', 'content': 'The lesson begins now.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'demo'}},

        {'role': 'user', 'content': 'We begin every day at 8.'},
        {'role': 'assistant', 'content': 'We begin with some reading.', 'info': {'table': 'vocab', 'uid': 6, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Miss, I see a new place outside.'},
        {'role': 'assistant', 'content': 'That is a café near the station.', 'info': {'table': 'vocab', 'uid': 3, 'type': 'demo'}},

        {'role': 'user', 'content': 'It is small and quiet.'},
        {'role': 'assistant', 'content': 'Yes, it is a small café.', 'info': {'table': 'vocab', 'uid': 3, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Do you go there?'},
        {'role': 'assistant', 'content': 'Yes, I drink coffee there every morning.', 'info': {'table': 'vocab', 'uid': 4, 'type': 'demo'}},

        {'role': 'user', 'content': 'Coffee is nice.'},
        {'role': 'assistant', 'content': 'My friend also likes coffee.', 'info': {'table': 'vocab', 'uid': 4, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Is it near the bus stop?'},
        {'role': 'assistant', 'content': 'No, it is near the train station.', 'info': {'table': 'vocab', 'uid': 5, 'type': 'demo'}},

        {'role': 'user', 'content': 'Do you go by train?'},
        {'role': 'assistant', 'content': 'Yes, I walk to the station every day.', 'info': {'table': 'vocab', 'uid': 5, 'type': 'quiz'}},

        {'role': 'user', 'content': 'This café is quiet and clean.'},
        {'role': 'assistant', 'content': 'Yes, it is small and nice.', 'info': {'table': 'grammar', 'uid': 1, 'type': 'demo'}},

        {'role': 'user', 'content': 'My home is big and quiet too.'},
        {'role': 'assistant', 'content': 'The teachers are kind and helpful.', 'info': {'table': 'grammar', 'uid': 1, 'type': 'quiz'}},

        {'role': 'user', 'content': 'The dog is small and friendly.'},
        {'role': 'assistant', 'content': 'Very good, Elinor. You used it well.', 'info': {'table': 'grammar', 'uid': 1, 'type': 'quiz'}},

        {'role': 'user', 'content': 'Thank you!'},
        {'role': 'assistant', 'content': 'You’ve learned all the key words today. Well done, Elinor!', 'info': {'table': None, 'uid': None, 'type': 'hooray'}},
    ]


    apikey = read_apikey("config/api.key")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, value in testcase.items():
        if isinstance(value, torch.Tensor):
            testcase[key] = value.to(device)
    world = response_prompter(context, api_key=apikey, **testcase)