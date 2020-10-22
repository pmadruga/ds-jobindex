# from nltk.corpus import stopwords
# from nltk.stem.snowball import DanishStemmer
from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.metrics import CosineSimilarity
# from textblob import TextBlob
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

        # self.export_preprocessed(dataset_path)

        # vectorize
        # self.tfidf_vectorize()
        # create similarity matrix
        # self.generate_similarity_matrix()

    def load_config(self):
        tqdm.pandas()
        nltk.download('stopwords')

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
            # + self.df_init['link'].astype(str).fillna('') + ' '
            # + self.df_init['ratings_link'].fillna('') + ' '
            # + self.df_init['source'].fillna('') + ' '
            # + self.df_init['description'].fillna('') + ' '
            + self.df_init['date'].astype(str).fillna('')
        )

        self.df = self.df_init

    # def preprocess_text(self, text):
    #     # text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    #     text = str(text).lower().strip()

    #     # caveat: this might conflict with the english text
    #     da_stop_words = stopwords.words('danish')
    #     stemmer = DanishStemmer()
    #     lemmatizer = lemmy.load("da")

    #     # remove plurals
    #     textblob = TextBlob(text)
    #     singles = [stemmer.stem(word) for word in textblob.words]

    #     # remove danish stopwords
    #     no_stop_words = [word for word in singles if word not in da_stop_words]

    #     # join text so it can be lemmatized
    #     joined_text = " ".join(no_stop_words)

    #     # lemmatization
    #     final_text = lemmatizer.lemmatize("", joined_text)

    #     return final_text[0]

    def preprocess(self):
        print('\n Preprocessing text')
        self.df['corpus'] = self.df['merged'].swifter.apply(
            preprocess_text)
        self.df['title_processed'] = self.df['title'].swifter.apply(
            preprocess_text)

    # def export_preprocessed(self, dataset_path):
    #     print('\n Exporting preprocessed dataset')
    #     outname = os.path.basename(dataset_path)

    #     outdir = os.path.abspath(os.getcwd()) + '/data/processed/'
    #     if not os.path.exists(outdir):
    #         os.mkdir(outdir)

    #     full_path = os.path.join(outdir, outname)
    #     self.df[['bow', 'merged', 'title','description']].to_csv(full_path)
    #     print(f'\n preprocessed dataset exported to: \n {full_path}')

    def create_embeddings(self):
        # Load multilingual BERT
        embedder = SentenceTransformer(
            'distilbert-multilingual-nli-stsb-quora-ranking')
        print('\n Creating embeddings')
        # Corpus is the bag of words
        corpus = self.df['corpus']
        title = self.df['title']

        # TFIDF embeddings
        vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_embeddings = vectorizer.fit_transform(corpus)

        # BERT embeddings
        self.bert_embeddings = embedder.encode(self.df['title'], convert_to_tensor=True)

        # Creating an index row for the distance matrix
        x, y = self.bert_embeddings.shape
        self.index_cols = np.arange(0, x, 1).tolist()

    def calculate_distances(self):
        print('\n Calculating distances - TFIDF')
        self.tfidf_distances = pairwise_distances_chunked(
            self.tfidf_embeddings, metric='cosine', n_jobs=-1)

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

    # def tfidf_vectorize(self):
    #     print('\n Preprocessing')
    #     self.df['merged'].swifter.apply(self.preprocess_text)
    #     print('\n Vectorizing')
    #     vectorizer = TfidfVectorizer()
    #     self.X = vectorizer.fit_transform(self.df['merged'])

    # def generate_similarity_matrix(self):
    #     print("\n Generating similarity matrix")
    #     similarity_matrix = cosine_similarity(self.X)

    #     self.similarity_matrix_output(similarity_matrix=similarity_matrix)
    #     return similarity_matrix

    # def similarity_matrix_output(self, similarity_matrix):
    #     print("\n Creating output of similarity matrix")

    #     outname = 'similarity_matrix.csv'

    #     outdir = os.path.abspath(os.getcwd()) + '/data/processed/'
    #     if not os.path.exists(outdir):
    #         os.mkdir(outdir)

    #     fullname = os.path.join(outdir, outname)
    #     df_matrix = pd.DataFrame(data=similarity_matrix)
    #     df_matrix.to_csv(fullname)

    # def vectorized_bag_of_words(self):
    #     return (self.X, self.df, self.df_init)


def load():
    print('\n load stop words:')
    nltk.download('stopwords')
