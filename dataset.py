import pandas
import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

header_size = dict()
vocab_list = dict()
tokenizer = get_tokenizer('basic_english')


def get_header_size(df):
    for header in df.columns:
        size = 0
        for item in df[header]:
            size = max(size, len(str(item).split(" ")))
        header_size[header] = size


def text_pipeline(text: str, header):
    temp = vocab_list.get(header)(tokenizer(text))
    temp += [0] * (header_size.get(header) - len(temp))
    return temp


def get_tensor(df):
    for header in df.columns:
        vocab = build_vocab_from_iterator([y for y in [str(x).split(" ") for x in df[header]]])
        vocab.set_default_index(0)
        vocab_list[header] = vocab


class JobPostingDataSet(Dataset):
    def __init__(self, filename):
        self.job_postings = pandas.read_csv(filename)
        get_header_size(self.job_postings)
        # tokenise(self.job_postings)
        get_tensor(self.job_postings)

    def __len__(self):
        return len(self.job_postings)

    def __getitem__(self, idx):
        temp = self.job_postings.iloc[idx].to_dict()
        label = temp["fraudulent"]

        for header in self.job_postings.columns:
            temp[header] = torch.tensor(text_pipeline(str(temp.get(header)), header))

        return temp, label
