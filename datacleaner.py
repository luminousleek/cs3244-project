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


def remove_stop_words(text):
    split = text.split(" ")
    result = [word for word in split if word not in stopword]
    result = ' '.join(result)
    return result


def lemmatise(text):
    split = text.split(" ")
    result = list(map(lambda word: wn.lemmatize(word), split))
    result = ' '.join(result)
    return result


def clean_text(t):
    t = remove_punctuation_and_captalisation(t)
    t = remove_stop_words(t)
    t = lemmatise(t)
    return t


def process_location(t):
    # no removing stop words and lemmatise because might mess up data
    # e.g. "US" gets shortened to just "u"
    t = remove_punctuation_and_captalisation(t)
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
    df['salary'] = df['salary_range'].apply(lambda x: get_salary(x))
    df['start_salary'] = df['salary'].apply(lambda x: x.split("-")[0])
    df['end_salary'] = df['salary'].apply(lambda x: x.split("-")[1])
    df.drop(['salary', 'salary_range'], axis=1, inplace=True)


def get_salary(salary) -> str:
    if pd.isna(salary):
        return "0-0"

    salary = salary.split('-')
    start_sal = process_salary_value(salary[0])

    if len(salary) > 1:
        end_sal = process_salary_value(salary[1])
        return str(start_sal) + "-" + str(end_sal)

    return str(start_sal) + "-" + str(start_sal)


def clean_df(df):
    for col in df:
        if col == 'location':
            df[col] = df[col].apply(lambda x: process_location(x))
        elif col == 'salary_range' or col in numerical_cols:
            continue
        else:
            df[col] = df[col].apply(lambda x: clean_text(x))
    process_salary(df)


clean_df(job_posting)
job_posting.to_csv('cleaned_job_postings.csv', index=False)
