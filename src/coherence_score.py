# Calculates coherence score given the aspect_embeddings pickle file.

import numpy
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

def document_frequency(word, reviews):
    
