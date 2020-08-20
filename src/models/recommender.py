from sklearn.metrics.pairwise import cosine_similarity


class Similarity():
    # def __init__(self):
    # self.cosine_similarity(X)

    def search(self, job_title, df):
        matching_entries = [df.index[df['merged'].str.contains(
            word, case=False)].values for word in job_title.split()]
        return list(set(matching_entries[0]).intersection(*matching_entries))

    def cosine_similarity(self, job_title, X, df, df_init):
        similarity_matrix = cosine_similarity(X)

        similarities = []
        search_matching_indeces = self.search(job_title, df)

        # print(search_matching_indeces)
        for matching_index in search_matching_indeces:
            for idx, similarity_value in enumerate(
                    similarity_matrix[matching_index]):
                # go fetch info from each similarity value
                # by using its index and compare with the data frame

                similarities.append({
                    "title": df['title'][idx],
                    "similarity": similarity_value,
                    "description": df_init['description'][idx],
                    "location": df_init['location'][idx],
                    "exact_match": True if matching_index == idx else False
                })

        # sort by similarity
        sorted_similarities = sorted(
            similarities,
            key=lambda x: x['similarity'],
            reverse=True)

        # just return the top X similarities
        return sorted_similarities[:20]
