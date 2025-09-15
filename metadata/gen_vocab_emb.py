# 生成词汇表中各词汇的语义embedding，用于response_prompter等模块中的语义相关度计算
# 保存到[language]/vocab_emb，可由DataManager加载，避免每次用到时重复计算
# embedding仅根据词汇本身计算，因为它的运用与系统给出的续写建议有关，而不依赖于guideword或某个具体义项。
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

def gen_vocab_emb(language='french'):
    print('starting...')
    vocab = pd.read_csv(f"{language}/vocab.csv")
    print('loaded vocab')
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    words = vocab['word'].tolist()
    print('starting embedding calculation')
    embeddings = embed_model.encode(words, convert_to_tensor=True)
    print('calculation complete')
    print(embeddings.shape)
    torch.save(embeddings, f"{language}/vocab_emb.pt")

if __name__ == '__main__':
    gen_vocab_emb()
