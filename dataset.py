import pandas
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

features_to_use = ['company_profile', 'description', 'requirements']


def build_vocab(df):
    vocab_list = dict()

    for header in df.columns:
        vocab = build_vocab_from_iterator([y for y in [str(x).split(" ") for x in df[header]]])
        vocab.set_default_index(0)
        vocab_list[header] = vocab

    return vocab_list


def combine_descriptions(df):
    df['combined_description'] = ''
    for feature in features_to_use:
        df['combined_description'] += ' ' + df[feature]


class JobPostingDataSet(Dataset):
    def __init__(self, filename):
        self.job_postings = pandas.read_csv(filename)
        combine_descriptions(self.job_postings)
        self.vocab_list = build_vocab(self.job_postings)

    def __len__(self):
        return len(self.job_postings)

    def __getitem__(self, idx):
        temp = self.job_postings.iloc[idx].to_dict()
        label = temp.get("fraudulent")

        text = ' '.join(list(map(lambda x: str(temp.get(x)), features_to_use)))
        return text, label
