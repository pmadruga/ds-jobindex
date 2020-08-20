from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
import numpy as np
import os
import pandas as pd
import re


class FeatureEngineering():
    """ initiating the cleanup pipeline """

    def __init__(self):
        self.load_dataset()
        # removes unwanted features (ie, ratings_link)
        self.filter_features()
        # vectorize
        self.tfidf_vectorize()

    def load_dataset(self):
        dataset_path = '/data/interim/jobindex_cropped_bigger.csv'
        # load raw data set
        self.df_init = pd.read_csv(os.path.abspath(os.getcwd()) + dataset_path)
        print('dataset loaded successfuly.')

        # removes unwanted features (ie, ratings_link)
        self.filter_features()

    def filter_features(self):
        # word bagging: merge desired features into one
        self.df_init['merged'] = (
            self.df_init['title'].fillna('') + ' '
            + self.df_init['company'].fillna('') + ' '
            + self.df_init['location'].fillna('') + ' '
            + self.df_init['link'].astype(str).fillna('') + ' '
            + self.df_init['ratings_link'].fillna('') + ' '
            + self.df_init['source'].fillna('') + ' '
            + self.df_init['description'].fillna('') + ' '
            + self.df_init['date'].astype(str).fillna('')
        )
        self.df = pd.DataFrame(self.df_init[['merged', 'title']])

    def preprocess_text(self, text):
        da_stop_words = stopwords.words('danish')

        # detect text language

        # if language is 'da' then translate to 'en'

        # remove punctuation
        # TODO: use Textblob
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        # remove stop words (in 'en', since 'da' is gone)
        removed_stop_words = [
            x for x in text.split() if x not in da_stop_words]
        text = " ".join(removed_stop_words)

        # TODO: lemmatization (= root value of word)

        # TODO: stemming (= removes -ing)

        return text

    def tfidf_vectorize(self):
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(
            self.df['merged'].apply(
                lambda x: self.preprocess_text(
                    text=x)))

    def vectorized_bag_of_words(self):
        return (self.X, self.df, self.df_init)


def load():
    nltk.download('stopwords')
