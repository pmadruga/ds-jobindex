from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import nltk
import numpy as np
import os
import pandas as pd
import re
import time
from tensorflow.keras.metrics import CosineSimilarity

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class MatrixGenerator():
    """ initiating the cleanup pipeline """

    def init(self, dataset_path):
        self.load_dataset(dataset_path)
        # removes unwanted features (ie, ratings_link)
        self.filter_features()
        # vectorize
        self.tfidf_vectorize()
        # create similarity matrix
        self.generate_similarity_matrix()

    def load_dataset(self, dataset_path):
        # load raw data set
        self.df_init = pd.read_csv(os.path.abspath(os.getcwd()) + dataset_path)
        print('\n Dataset loaded successfuly.')

    def filter_features(self):
        print('\n Selecting features')
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
        # self.df = pd.DataFrame(self.df_init[['merged', 'title']])

        self.df = self.df_init

    def preprocess_text(self, text):
        print('\n Processing text')
        # caveat: this might conflict with the english text
        da_stop_words = stopwords.words('danish')

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
        print('\n Performing TFIDF')
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(
            self.df['merged'].apply(
                lambda x: self.preprocess_text(
                    text=x)))

    def generate_similarity_matrix(self):
        print("\n Generating similarity matrix")
        similarity_matrix = cosine_similarity(self.X)

        self.similarity_matrix_output(similarity_matrix=similarity_matrix)
        return similarity_matrix

    def similarity_matrix_output(self, similarity_matrix):
        print("\n Creating output of similarity matrix")

        outname = 'similarity_matrix.csv'

        outdir = os.path.abspath(os.getcwd()) + '/data/processed/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname = os.path.join(outdir, outname)
        df_matrix = pd.DataFrame(data=similarity_matrix)
        df_matrix.to_csv(fullname)

    def vectorized_bag_of_words(self):
        return (self.X, self.df, self.df_init)


def load():
    print('\n load stop words:')
    nltk.download('stopwords')
