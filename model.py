from os.path import exists

from torch import torch, nn

from dataset import split_data
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

device = 'cpu'
dataset_file = 'cleaned_job_postings.csv'
model_file = 'model_weights.pth'
vocab_file = 'vocab.pyi'
tokenizer = get_tokenizer('basic_english')
dataset = split_data(dataset_file)


def text_pipeline(text: str, vocab):
    temp = vocab(tokenizer(text))
    return temp


def label_pipeline(label: str):
    return int(label)


def collate_batch(vocab):
    def cb(_batch):
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in _batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text, vocab), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    return lambda x: cb(x)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


def save_model(model):
    torch.save(model[0], model_file)
    torch.save(model[1], vocab_file)


def load_model():
    if not exists(model_file):
        return None, None

    model = torch.load(model_file)
    voc = torch.load(vocab_file)

    voc.set_default_index(voc["<unk>"])
    return model, voc
