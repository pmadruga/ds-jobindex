from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import time
import os


def search(job_title, df):
    matching_entries = [df.index[df['merged'].str.contains(
        word, case=False)].values for word in job_title.split()]
    return list(set(matching_entries[0]).intersection(*matching_entries))


class Recommender():
    def determine_similarity(self, job_title, dataset_path):
        similarity_matrix = pd.read_csv(
            os.path.abspath(
                os.getcwd()) +
            '/data/processed/similarity_matrix.csv', index_col=0)
        df = pd.read_csv(os.path.abspath(os.getcwd()) + dataset_path)

        df['merged'] = (
            df['title'].fillna('') + ' '
            + df['company'].fillna('') + ' '
            + df['location'].fillna('') + ' '
            + df['link'].astype(str).fillna('') + ' '
            + df['ratings_link'].fillna('') + ' '
            + df['source'].fillna('') + ' '
            + df['description'].fillna('') + ' '
            + df['date'].astype(str).fillna('')
        )

        len, features = similarity_matrix.shape
        similarities = [None] * len

        search_matching_indeces = search(job_title, df)

        for matching_index in search_matching_indeces:
            for idx, similarity_value in enumerate(
                    (similarity_matrix.to_numpy())[matching_index]):

                similarities[idx] = ({
                    "title": df['title'][idx],
                    "similarity": similarity_value,
                    "description": df['description'][idx],
                    "location": df['location'][idx],
                    "exact_match": True if matching_index == idx else False
                })

        # sort by similarity
        sorted_similarities = sorted(
            similarities,
            key=lambda x: x['similarity'],
            reverse=True)

        # just return the top X similarities
        return sorted_similarities[:20]
