from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.metrics import CosineSimilarity
from tqdm.notebook import tqdm
from features.process_text import preprocess_text

import csv
import nltk
import numpy as np
import os
import pandas as pd
import re
import swifter
import time
import torch


class Preprocess():
    def init(self, dataset_path):
        self.load_dataset(dataset_path)
        # removes unwanted features (ie, ratings_link)
        self.filter_features()

        self.preprocess()

        self.create_embeddings()

        self.calculate_distances()

        self.export_distances_matrix(dataset_path)

    def load_dataset(self, dataset_path):
        # load raw data set
        self.df_init = pd.read_csv(os.path.abspath(os.getcwd()) + dataset_path)
        print('\n Dataset loaded successfuly.')

    """
    During testing, it was discovered that the 'description' feature
    was adding more signal than noise. Hence being excluded. Other features were
    also removed due to its noise.

    With more time, it would be interesting to investigate which features have more signal,
    perhaps using a PCA approach or similar.
    """

    def filter_features(self):
        print('\n Selecting features')
        # word bagging
        self.df_init['merged'] = (
            self.df_init['title'].fillna('') + ' '
            + self.df_init['company'].fillna('') + ' '
            + self.df_init['location'].fillna('') + ' '
            # + self.df_init['link'].astype(str).fillna('') + ' '
            # + self.df_init['ratings_link'].fillna('') + ' '
            # + self.df_init['source'].fillna('') + ' '
            # + self.df_init['description'].fillna('') + ' '
            # + self.df_init['date'].astype(str).fillna('')
        )

        self.df = self.df_init

    """
    There are two types of features created after processing text:
    1 - corpus, which includes several original features;
    2 - title_processed, that helps with direct search results.
    """

    def preprocess(self):
        print('\n Preprocessing text')
        self.df['corpus'] = self.df['merged'].swifter.apply(
            preprocess_text)
        self.df['title_processed'] = self.df['title'].swifter.apply(
            preprocess_text)

    def create_embeddings(self):
        # Load multilingual BERT
        embedder = SentenceTransformer(
            'distilbert-multilingual-nli-stsb-quora-ranking')


        
        # Corpus is the bag of words
        corpus = self.df['corpus']
        title = self.df['title']

        # TFIDF embeddings
        print('\n Creating embeddings - TFIDF')
        vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_embeddings = vectorizer.fit_transform(corpus)

        # BERT embeddings
        print('\n Creating embeddings - BERT')
        self.bert_embeddings = embedder.encode(
            self.df['title'], convert_to_tensor=True)

        # Creating an index row for the distance matrix
        x, y = self.bert_embeddings.shape
        self.index_cols = np.arange(0, x, 1).tolist()

    def calculate_distances(self):
        print('\n Calculating distances - TFIDF')
        self.tfidf_distances = pairwise_distances_chunked(
            self.tfidf_embeddings, metric='cosine', n_jobs=-1)

        """
        Below is deprecated distance calculation for BERT,
        Since embeddings are created using setence-transfomers (as a tensor)
        and the distances are calculated afterwards (in the jupyter notebook).
        """
        # print('\n Calculating distances - BERT')
        # self.bert_distances = pairwise_distances_chunked(
        #     self.bert_embeddings, metric='cosine', n_jobs=-1)

    def export_distances_matrix(self, dataset_path):
        outname_tfidf = os.path.basename(dataset_path).split('.')[
            0] + '_distances_tfidf.csv'
        outname_bert = os.path.basename(dataset_path).split('.')[
            0] + '_encodings_bert.pt'
        outname_df = os.path.basename(dataset_path).split('.')[
            0] + '_preprocessed_df.csv'

        outdir = os.path.abspath(os.getcwd()) + '/data/processed/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        full_path_tfidf = os.path.join(outdir, outname_tfidf)
        full_path_bert = os.path.join(outdir, outname_bert)
        full_path_df = os.path.join(outdir, outname_df)

        # tfidf
        self.write_file(full_path_tfidf, self.tfidf_distances)

        # bert
        # self.write_file(full_path_bert, self.bert_distances)

        print('\n Storing embeddings - BERT')
        torch.save(self.bert_embeddings, full_path_bert)

        # dataframe - preprocessed
        self.df.to_csv(full_path_df)

    def write_file(self, file_path, data):
        print(f'\n Writing distance matrix')
        with open(file_path, "w") as fp:
            wr = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_ALL)
            # writing the first row as the index
            wr.writerow(self.index_cols)

            # iterating the generated distance matrix
            # and writing to file.
            for chunk in tqdm(data):
                for item in chunk:
                    wr.writerow(item)


def load():
    print('\n load stop words:')
    nltk.download('stopwords')
