#ds-jobindex
==============================
Machine learning techniques applied to the jobindex.dk dataset

## Run

To generate a preprocessed dataset:

```
python src --path '/data/interim/jobindex_cropped_bigger.csv'
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
