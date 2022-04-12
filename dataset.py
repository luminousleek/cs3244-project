import pandas
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

boolean_features = {'has_company_logo': 0, 'has_questions': 1}
small_features = ['required_experience', 'required_education', 'employment_type', 'industry', 'function']
long_features = ['description']
tokenizer = get_tokenizer('basic_english')


def split_data(file_path, to_DS=False):
    df = pandas.read_csv(file_path)
    fjp = df.copy(True).drop(df[df['fraudulent'] == 0].index)
    tjp = df.copy(True).drop(df[df['fraudulent'] == 1].index)

    if to_DS:
        fjp, tjp = JobPostingDataSet(dataset=fjp), JobPostingDataSet(dataset=tjp)

    return fjp, tjp


def format_boolean(value, dft):
    try:
        return str(int(value))
    except ValueError:
        return str(dft)


def combine_descriptions(df):
    for feature, dft in boolean_features.items():
        df[feature] = df[feature].apply(lambda x: format_boolean(x, dft))
    df['temp1'] = df[boolean_features.keys()].apply(lambda row: "__".join(row.values.astype(str)), axis=1)
    df['temp2'] = ''
    for feature in long_features:
        df['temp2'] = df['temp2'] + " " + df[feature].apply(lambda x: " ".join(str(x).split(" ")[:100]))
    df['combined_description'] = df[['temp1', 'temp2'] + small_features].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1)
    df.drop(['temp1', 'temp2'], axis=1, inplace=True)

    return df


class JobPostingDataSet(Dataset):
    def __init__(self, filename=None, dataset=None):
        if filename is None and dataset is None:
            raise RuntimeError("provide file name or pandas dataset")
        if dataset is not None:
            self.job_postings = dataset
        else:
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
