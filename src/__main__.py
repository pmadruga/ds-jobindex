import sys
from features.build_features import load, MatrixGenerator
from models import Recommender
from tqdm import tqdm
import argparse


class JobRecommender():
    def __init__(self):
        self.init_parser()
        # load()
        # self.recommend_jobs(job_title)

        # if no job title is proved, just generate similarity matrix.
        if(self.args.recommend is None):
            MatrixGenerator().init(dataset_path=self.args.path)
        # otherwise, specify an existing generated model to get job
        # recommendations
        else:
            self.recommend_jobs(self.args.recommend, self.args.path)

    def init_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--recommend',
            help="returns recommendations for the job title provided. ex: --recommend 'risk manager'.")
        parser.add_argument("--path", help="path to the raw dataset. without using --recommend, this will generate similarity matrix.")
        self.args = parser.parse_args()

    def recommend_jobs(self, job_title, dataset_path):
        recommendations = Recommender().determine_similarity(job_title, dataset_path)
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
    JobRecommender()
