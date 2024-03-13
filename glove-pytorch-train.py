
# import gensim
from gensim.models.word2vec import PathLineSentences, LineSentence
# from multiprocessing import Pool
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from collections import defaultdict
import psutil
from time import perf_counter
import os
import lmdb
import pickle
import shutil
# import glob
import signal
import sys
import numpy as np

# Constants
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOW_TOKEN = "<bow>"
EOW_TOKEN = "<eow>"
WEIGHT_INIT_RANGE = 0.1

# linesentence_reader_process_number = 8

# embedding_dim = 64
# context_size = 2
# batch_size = 1024
# num_epoch = 10
embedding_dim = 300
context_size = 8
batch_size = 512
num_epoch = 30
learning_rate = 0.001

word_separated_txt_path = 'data/temp_training_data' # 1% of the data
results_path = 'data'
line_limit_per_document = None #for testing. None for no limit

lmdb_map_size = 30*1024*1024*1024 #30GB

# 用以控制样本权重的超参数
m_max = 100
alpha = 0.75


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

def remove_contents(directory):
    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # remove file or symlink
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # remove directory


# ver1
def load_linesentences(txt_path, limit=None):
    for sentence in PathLineSentences(source=txt_path, limit=limit):
        yield sentence

# ver2
# def process_path(path_limit):
#     path, limit = path_limit
#     return [sentence for sentence in LineSentence(source=path, limit=limit)]
# def load_linesentences(txt_path, limit=None):
#     all_paths = glob.glob(txt_path + '/*.gz', recursive=True)
#     with Pool(linesentence_reader_process_number) as pool:
#         for document in pool.imap_unordered(process_path, ((path, limit) for path in all_paths)):
#             for sentence in document:
#                 yield sentence


def save_pretrained(vocab, embeds, save_path):
    """
    Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line.
    """
    with open(save_path, "w") as writer:
        writer.write(f"{embeds.shape[0]} {embeds.shape[1]}\n")
        for idx, token in enumerate(vocab.idx_to_token):
            vec = " ".join(["{:.4f}".format(x) for x in embeds[idx]])
            writer.write(f"{token} {vec}\n")
    print(f"Pretrained embeddings saved to: {save_path}")


def get_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle
    )
    return data_loader

def init_weights(model):
    for name, param in model.named_parameters():
        if "embedding" not in name:
            torch.nn.init.uniform_(
                param, a=-WEIGHT_INIT_RANGE, b=WEIGHT_INIT_RANGE
            )


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in tqdm(text, desc="Building Vocabulary"):
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    
    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def save_vocab(vocab, path):
    with open(path, 'w') as writer:
        writer.write("\n".join(vocab.idx_to_token))


def read_vocab(path):
    with open(path, 'r') as f:
        tokens = f.read().split('\n')
    return Vocab(tokens)




# the original in-memory version
class GloveDataset(Dataset):
    def __init__(self, corpus, vocab, context_size=2):
        # 记录词与上下文在给定语料中的共现次数
        self.cooccur_counts = defaultdict(float)
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        for sentence in tqdm(corpus, desc="Dataset Construction"):
            # sentence = [self.bos] + sentence + [self.eos]
            sentence = [self.bos] + vocab.convert_tokens_to_ids(sentence) + [self.eos]
            for i in range(1, len(sentence)-1):
                w = sentence[i]
                left_contexts = sentence[max(0, i - context_size):i]
                right_contexts = sentence[i+1:min(len(sentence), i + context_size)+1]
                # 共现次数随距离衰减: 1/d(w, c)
                for k, c in enumerate(left_contexts[::-1]):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)
                for k, c in enumerate(right_contexts):
                    self.cooccur_counts[(w, c)] += 1 / (k + 1)
        self.data = [(w, c, count) for (w, c), count in self.cooccur_counts.items()]
        print(f'co-occurence matrix size: {len(self.data)}, memory required: {len(self.data)*3*8/1024/1024} MB')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        return (words, contexts, counts)




# LMDB version
class GloveDatasetLMDB(Dataset):
    def __init__(self, corpus_generator, vocab, context_size=2):
        self.filename = os.path.join(results_path, 'cooccur_counts_lmdb')
        self.filename_list = os.path.join(results_path, 'cooccur_counts_list_lmdb')
        if os.path.exists(self.filename):
            remove_contents(self.filename)
        if os.path.exists(self.filename_list):
            remove_contents(self.filename_list)
        self.env = lmdb.open(path=self.filename, map_size=lmdb_map_size, map_async=True, writemap=True, readonly=False, lock=False, readahead=False,  meminit=False, create=True)
        self.env_list = lmdb.open(path=self.filename_list, map_size=lmdb_map_size, map_async=True, writemap=True, readonly=False, lock=False, readahead=False, meminit=False, create=True)
        
        self.bos = vocab[BOS_TOKEN]
        self.eos = vocab[EOS_TOKEN]
        with self.env.begin(write=True, buffers=True) as txn:
            for sentence in tqdm(corpus_generator, desc="Dataset Construction"):
                sentence = [self.bos] + vocab.convert_tokens_to_ids(sentence) + [self.eos]
                for i in range(1, len(sentence)-1):
                    w = sentence[i]
                    left_contexts = sentence[max(0, i - context_size):i]
                    right_contexts = sentence[i+1:min(len(sentence), i + context_size)+1]
                    for k, c in enumerate(left_contexts[::-1]):
                        key = pickle.dumps((w, c))
                        old_value = txn.get(key)
                        if old_value is not None:
                            old_value = pickle.loads(old_value)
                        else:
                            old_value = 0
                        txn.put(key, pickle.dumps(old_value + 1 / (k + 1)))
                    for k, c in enumerate(right_contexts):
                        key = pickle.dumps((w, c))
                        old_value = txn.get(key)
                        if old_value is not None:
                            old_value = pickle.loads(old_value)
                        else:
                            old_value = 0
                        txn.put(key, pickle.dumps(old_value + 1 / (k + 1)))

        # dump to a new database with consequtive index as a key and (w, c, count) as value
        print('dumping to a list')
        with self.env.begin(write=False) as txn:
            with self.env_list.begin(write=True, buffers=True) as txn_list:
                for i, (k, v) in tqdm(enumerate(txn.cursor())):
                    k = pickle.loads(k)
                    v = (k[0], k[1], pickle.loads(v))
                    txn_list.put(pickle.dumps(i), pickle.dumps(v))


    def __len__(self):
        with self.env.begin(write=False, buffers=True) as txn:
            return txn.stat()['entries']

    def __getitem__(self, i):
        with self.env_list.begin(write=False, buffers=True) as txn_list:
            return pickle.loads(txn_list.get(pickle.dumps(i)))
        

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples])
        contexts = torch.tensor([ex[1] for ex in examples])
        counts = torch.tensor([ex[2] for ex in examples])
        return (words, contexts, counts)




class GloveModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloveModel, self).__init__()
        # 词嵌入及偏置向量
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_biases = nn.Embedding(vocab_size, 1)
        # 上下文嵌入及偏置向量
        self.c_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.c_biases = nn.Embedding(vocab_size, 1)

    def forward_w(self, words):
        w_embeds = self.w_embeddings(words)
        w_biases = self.w_biases(words)
        return w_embeds, w_biases

    def forward_c(self, contexts):
        c_embeds = self.c_embeddings(contexts)
        c_biases = self.c_biases(contexts)
        return c_embeds, c_biases


def main():

    start_time = perf_counter()
    print(f'memory used: {psutil.virtual_memory().percent}%')
    vocab = Vocab.build(load_linesentences(word_separated_txt_path, line_limit_per_document), reserved_tokens=[PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    print(f"Length of vocab: {len(vocab)}")

    end_time = perf_counter()
    m, s = divmod(end_time-start_time, 60)
    print(f'time {m} minutes {s} seconds')
    print(f'memory used: {psutil.virtual_memory().percent}%')

    corpus = load_linesentences(word_separated_txt_path, line_limit_per_document)


    dataset = GloveDataset(
            corpus,
            vocab,
            context_size=context_size
        )


    # dataset = GloveDatasetLMDB(
    #         corpus,
    #         vocab,
    #         context_size=context_size
    #     )
   

    end_time = perf_counter()
    m, s = divmod(end_time-start_time, 60)
    print(f'time {m} minutes {s} seconds')
    print(f'memory used: {psutil.virtual_memory().percent}%')

    data_loader = get_loader(dataset, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GloveModel(len(vocab), embedding_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            words, contexts, counts = [x.to(device) for x in batch]
            # 提取batch内词、上下文的向量表示及偏置
            word_embeds, word_biases = model.forward_w(words)
            context_embeds, context_biases = model.forward_c(contexts)
            # 回归目标值：必要时可以使用log(counts+1)进行平滑
            log_counts = torch.log(counts)
            # 样本权重
            weight_factor = torch.clamp(torch.pow(counts / m_max, alpha), max=1.0)
            optimizer.zero_grad()
            # 计算batch内每个样本的L2损失
            loss = (torch.sum(word_embeds * context_embeds, dim=1, keepdim=True) + word_biases + context_biases - log_counts) ** 2
            # 样本加权损失
            wavg_loss = (weight_factor * loss).mean()
            wavg_loss.backward()
            optimizer.step()
            total_loss += wavg_loss.item()
        print(f"Loss: {total_loss:.2f}")

    # 合并词嵌入矩阵与上下文嵌入矩阵，作为最终的预训练词向量
    combined_embeds = model.w_embeddings.weight + model.c_embeddings.weight
    save_pretrained(vocab, combined_embeds.data, os.path.join(results_path, "glove.vec"))

    end_time = perf_counter()
    m, s = divmod(end_time-start_time, 60)
    print(f'time {m} minutes {s} seconds')


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)
    main()
    print('done')