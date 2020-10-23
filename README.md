# Deep Learning based search engine - Jobindex.DK case

## Abstract
After scrapping the data from jobindex.dk - Denmark's biggest job portal - of 4.2 million jobs, a set of different Natural Language Processing techniques and Machine Learning models were applied to the data. Specifically, [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and [BERT](https://github.com/google-research/bert) models were applied on top of a direct search (no AI).

The goal was to improve search results on a non-English data set, which was achieved, especially with [BERT](https://github.com/google-research/bert).

## Structure

The search engine is structured into two parts:

1. Python script: preprocessing of text, generation of embeddings, distance calculation and file exporting.
2. Jupyter notebook: Determine BERT distances and present results.

## Run

To generate a preprocessed dataset:

```
python src --process '/data/interim/jobindex_cropped_bigger.csv'
```

To get recommendations (a preprocessed dataset has to be generated beforehand), run the `search_engine_results.ipynb` notebook.

## Scripts

To lint code, run: 
```
./scripts/lint-code.sh
```

To start notebooks, run:
```
./scripts/start_notebooks.sh
```
