import sys
from features.build_features import load, FeatureEngineering
from models import Similarity
from tqdm import tqdm


class JobRecommender():
    def __init__(self, job_title):
        load()
        self.recommend_jobs(job_title)

    def recommend_jobs(self, job_title):
        (X, df, df_init) = FeatureEngineering().vectorized_bag_of_words()
        recommendations = Similarity().cosine_similarity(job_title, X, df, df_init)
        self.prettify_recommendations(recommendations)

    def prettify_recommendations(self, recommendations):
        for (index, recommendation) in enumerate(recommendations):

            print(
                f'''
                    #{index}
                    Title: {recommendation["title"]}
                    similarity: {recommendation["similarity"]}
                    exact match: {recommendation["exact_match"]}
                    description: {recommendation["description"]} \n\n
                    ''')


if __name__ == "__main__":
    JobRecommender(job_title=sys.argv[1])
