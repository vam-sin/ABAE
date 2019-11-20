# Attention Based Aspect Extraction

Implementation of the paper titled "An Unsupervised Neural Attention Model for Aspect Extraction".
[Link to the paper](https://www.aclweb.org/anthology/P17-1036/)

Aims to predict more coherent aspects. 
Tasks:
Aspect Term Extraction.
Clustering the aspect terms into categories.

# Steps

## Installing the required libraries and dependencies

```python3
pip3 install -r requirements.txt
```
## Generation of dependent files
Run these commands in the /src folder
```python3
python3 train_word2vec.py
python3 preprocess.py
python3 word2vec.py
python3 aspect_embeddings.py
```
Excecute the following the command to train the model. (Set the hyperaparameters accordingly)

## Model training

```python3
python3 train.py
```

## Results
Run these commands in the /src folder
```python3
python3 aspect_retrieval.py
```


### Author

[Vamsi Nallapareddy](https://github.com/vam-sin)
