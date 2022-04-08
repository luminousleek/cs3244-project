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
    df['description'] = df['description'].astype(str)
    df['company_profile'] = df['company_profile'].astype(str)
    df['combined_description'] = df[features_to_use].apply(lambda row: " ".join(row.values.astype(str)), axis=1)


class JobPostingDataSet(Dataset):
    def __init__(self, filename):
        self.job_postings = pandas.read_csv(filename)
        combine_descriptions(self.job_postings)
        self.vocab_list = build_vocab(self.job_postings)

    def __len__(self):
        return len(self.job_postings)

    def __getitem__(self, idx):
        temp = self.job_postings.iloc[idx].to_dict()
        label = temp.get('fraudulent')
        text = temp.get('description')
        return text, label
