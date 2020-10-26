"""
This is DEPRECATED. The most up-to-date logic lives in the notebooks folder,
under the search_engine_results.ipynb.

This will be reinstated in case there's a need to get the search engine results via a python script (which there'is not at the moment).
"""


# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
# from features.process_text import preprocess_text
# import pandas as pd
# import time
# import os
# import numpy as np
# import torch


# def search(job_title, df):
#     matching_entries = [df.index[df['merged'].str.contains(
#         word, case=False)].values for word in job_title.split()]
#     return list(set(matching_entries[0]).intersection(*matching_entries))


# class Recommender():
#     def init(self):
#         self.agg_results_indexes = []
#         self.agg_results = []
#         self.agg_indexes = []

#     def load_csv(self, dataset_path):
#         """
#         These would be more performant if it were stored & read
#         from a database, instead of memory.
#         """

#         outname_tfidf = os.path.basename(dataset_path).split('.')[
#             0] + '_distances_tfidf.csv'

#         outname_bert = os.path.basename(dataset_path).split('.')[
#             0] + '_distances_bert.csv'

#         outname_df = os.path.basename(dataset_path).split('.')[
#             0] + '_preprocessed_df.csv'

#         print('\n Loading TFIDF distances')
#         self.df = pd.read_csv(
#             f'{os.path.abspath(os.getcwd())}/data/processed/{outname_df}',
#             index_col=0)

#         print('\n Loading BERT distances')
#         self.tfidf = pd.read_csv(
#             f'{os.path.abspath(os.getcwd())}/data/processed/{outname_tfidf}',
#             index_col=0)

#         print('\n Loading Processed Dataframe')
#         self.bert = pd.read_csv(
#             f'{os.path.abspath(os.getcwd())}/data/processed/{outname_bert}',
#             index_col=0)

#     def search_results(self, job_title):
#         matching_entries = [self.df['title_processed']
#                             .index[self.df['title_processed'].str.contains(word, case=False)]
#                             .values for word in job_title.split()]
#         return list(set(matching_entries[0]).intersection(*matching_entries))

#     def print_direct_search_results(self):
#         print('\n DIRECT RESULTS')
#         for result in self.search_results:
#             print('- ', self.df['title'][result])
#             self.agg_results_indexes.append(result)

#     def determine_similarity(self, job_title, dataset_path):
#         self.load_csv(dataset_path)

#         # preprocessed text improves direct search results
#         self.direct_search_results = self.search_results(
#             preprocess_text(text=job_title))
#         self.print_direct_search_results()

# df['merged'] = (
#     df['title'].fillna('') + ' '
#     + df['company'].fillna('') + ' '
#     + df['location'].fillna('') + ' '
#     + df['link'].astype(str).fillna('') + ' '
#     + df['ratings_link'].fillna('') + ' '
#     + df['source'].fillna('') + ' '
#     + df['description'].fillna('') + ' '
#     + df['date'].astype(str).fillna('')
# )

# len, features = similarity_matrix.shape
# similarities = [None] * len

# search_matching_indeces = search(job_title, df)

# for matching_index in search_matching_indeces:
#     for idx, similarity_value in enumerate(
#             (similarity_matrix.to_numpy())[matching_index]):

#         similarities[idx] = ({
#             "title": df['title'][idx],
#             "similarity": similarity_value,
#             "description": df['description'][idx],
#             "location": df['location'][idx],
#             "exact_match": True if matching_index == idx else False
#         })

# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(df['bow'])

# concat_similarities = []

# search_results = search(job_title, df)

# for index, result in enumerate(search_results):
#     distances = pairwise_kernels(X, X[result], n_jobs=-1)
#     # print(distance)

#     flat_distances = np.array(distances).flatten()

#     for index, distance in enumerate(flat_distances):
#         concat_similarities.append({
#             'distance': float(distance),
#             'index': index,
#             'result_title': df['title'][result],
#             'description': df['description'][result]
#         })

# sorted_similarities = sorted(
#     concat_similarities,
#     key=lambda k: k['distance'],
#     reverse=True)[
#     :50]

# for index,value in tqdm(enumerate(sorted_similarities)):
# print(value)

# just return the top X similarities
# return sorted_similarities[:20]
