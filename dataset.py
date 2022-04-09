import pandas
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


features_to_use = ['company_profile', 'description', 'requirements', 'benefits']
tokenizer = get_tokenizer('basic_english')


def combine_descriptions(df):
    df['combined_description'] = df[features_to_use].apply(lambda row: " ".join(row.values.astype(str)), axis=1)


class JobPostingDataSet(Dataset):
    def __init__(self, filename):
        self.job_postings = pandas.read_csv(filename)
        combine_descriptions(self.job_postings)

    def __len__(self):
        return len(self.job_postings)

    def __getitem__(self, idx):
        temp = self.job_postings.iloc[idx].to_dict()
        label = temp.get('fraudulent')
        text = temp.get('combined_description')
        return text, label


def build_vocab(dataset):
    temp = list(map(lambda x: tokenizer(x[0]), dataset))

    vocab = build_vocab_from_iterator(temp, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    return vocab
