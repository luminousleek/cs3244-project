import pandas as pd
from string import punctuation
from nltk import corpus, WordNetLemmatizer

job_posting = pd.read_csv('fake_job_postings.csv')
stopword = corpus.stopwords.words('english')
wn = WordNetLemmatizer()
month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
numerical_cols = ['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'fraudulent']


def remove_punctuation_and_captalisation(text):
    stripped = str(text).replace(u'\xa0', u'')
    no_punct = [char.lower() for char in stripped if char not in punctuation]
    result = ''.join(no_punct)
    return result


def tokenise(text):
    split = text.split(" ")
    result = [word for word in split if not word.isspace() and word != '']
    return result


def remove_stop_words(text):
    result = [word for word in text if word not in stopword]
    return result


def lemmatise(text):
    return list(map(lambda word: wn.lemmatize(word), text))


def clean_text(t):
    t = remove_punctuation_and_captalisation(t)
    t = tokenise(t)
    t = remove_stop_words(t)
    t = lemmatise(t)
    return t


def process_location(t):
    # no removing stop words and lemmatise because might mess up data
    # e.g. "US" gets shortened to just "u"
    t = remove_punctuation_and_captalisation(t)
    t = tokenise(t)
    return t


def process_salary_value(t):
    # things like 9-12 get saved as 9-Dec in the csv file :/
    if t in month_dict:
        t = month_dict.get(t)

    t = int(t)
    # some salary ranges omit the thousands, e.g. 40000-50000 just becomes 40-50
    if t < 500:
        t = t * 1000
    return t


def process_salary(df):
    df['start_salary'] = ''
    df['end_salary'] = ''
    for i in range(len(df)):
        sal_range = df.at[i, 'salary_range']
        if pd.isna(sal_range):
            continue

        sal_range = sal_range.split('-')
        start_sal = process_salary_value(sal_range[0])
        df.at[i, 'start_salary'] = start_sal

        if len(sal_range) > 1:
            end_sal = process_salary_value(sal_range[1])
            df.at[i, 'end_salary'] = end_sal


def clean_df(df):
    for col in df:
        # print(col)
        if col == 'location':
            df[col] = df[col].apply(lambda x: process_location(x))
        elif col == 'salary_range' or col in numerical_cols:
            continue
        else:
            df[col] = df[col].apply(lambda x: clean_text(x))
    process_salary(df)


clean_df(job_posting)
job_posting.to_csv('cleaned_job_postings.csv')