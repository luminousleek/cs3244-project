import pandas as pd
from string import punctuation
from nltk import corpus, WordNetLemmatizer

job_posting = pd.read_csv("./no_punctuation_stopwords_with_lemma_data.csv")  # open csv file
stopword = corpus.stopwords.words('english')
wn = WordNetLemmatizer()


def clean_text(t):
    temp = []

    # remove stopwords and lemmatize
    for word in str(t).split(" "):
        if word in stopword:
            continue
        temp.append(wn.lemmatize(word) + " ")

    if temp[-1].isspace():
        temp.pop()

    result = []
    # remove punctuations
    for c in "".join(temp):
        if c not in punctuation:
            result.append(c.lower())

    return "".join(result)


def clean_texts(texts):
    for col in job_posting:
        texts[col] = texts[col].apply(lambda x: clean_text(x))


clean_texts(job_posting)
job_posting.to_csv('no_punctuation_stopwords_with_lemma_data.csv')  # save back csv file
