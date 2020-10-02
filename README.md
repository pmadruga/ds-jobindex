#ds-jobindex
==============================

##Data science and machine learning applied on the jobindex.dk dataset

### Run

To generate a preprocessed dataset:

```python src --path '/data/interim/jobindex_cropped_bigger.csv'```

To get recommendations (a preprocessed dataset has to be generated beforehand):

```python src --path '/data/processed/jobindex_cropped_bigger.csv' --recommend 'risk manager'```


### Results

It should return a dictionary of direct matches and subsequent recommendations.
