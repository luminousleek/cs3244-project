import pandas
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

boolean_features = ['salary_range', 'has_company_logo', 'has_questions']
small_features = ['title', 'location', 'required_experience', 'required_education', 'employment_type', 'industry',
                  'function']
long_features = ['description', 'company_profile', 'requirements', 'benefits']
tokenizer = get_tokenizer('basic_english')


def split_data(file_path, to_DS=False):
    df = pandas.read_csv(file_path)
    fjp = df.copy(True).drop(df[df['fraudulent'] == 0].index)
    tjp = df.copy(True).drop(df[df['fraudulent'] == 1].index)

    if to_DS:
        fjp, tjp = JobPostingDataSet(dataset=fjp), JobPostingDataSet(dataset=tjp)

    return fjp, tjp


def format_boolean(value, dft: str):
    try:
        int(value)
        return ":".join([dft, str(value)])
    except ValueError:
        return ":".join([dft, "nan"])


def format_string(string: str, dft: str):
    if string == "nan" or string == "":
        return ":".join([dft, "nan"])
    return string


def combine_descriptions(df):
    for feature in boolean_features:
        df[feature] = df[feature].apply(lambda x: format_boolean(x, feature))

    for feature in small_features + long_features:
        df[feature] = df[feature].apply(lambda x: format_string(x, feature))

    df['temp'] = ''
    for feature in long_features:
        df['temp'] = df['temp'] + " " + df[feature].apply(lambda x: " ".join(str(x).split(" ")[:300]))

    df['combined_description'] = df[boolean_features + small_features + ['temp']].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1)
    df.drop(['temp'], axis=1, inplace=True)

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
