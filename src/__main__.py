import sys
from features.build_features import load, Preprocess
# from models import Recommender
import argparse


class JobRecommender():
    def __init__(self):
        # helper text for when running the app
        self.init_parser()

        # if no job title is proved, just generate a preprocessed dataset.
        # if(self.args.recommend is None):
        Preprocess().init(dataset_path=self.args.process)
        # otherwise, specify an existing generated model to get job
       
        # recommendations
        # else:
            # self.recommend_jobs(self.args.recommend, self.args.path)

    def init_parser(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument(
        #     '--recommend',
        #     help="returns recommendations for the job title provided. ex: --recommend 'risk manager'.")

        parser.add_argument(
            "--process",
            help="path to the raw dataset. without using --recommend, this will generate a preprocessed dataset.")

        # make app arguments available in the class
        self.args = parser.parse_args()

    # def recommend_jobs(self, job_title, dataset_path):
    #     recommendations = Recommender().determine_similarity(job_title, dataset_path)
    #     self.prettify_recommendations(recommendations)


if __name__ == "__main__":
    JobRecommender()
