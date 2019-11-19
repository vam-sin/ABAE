import csv
import numpy as np
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))
import pickle

def generate_dataset():
    print("Dataset Generation")
    reviews = []
    with open("../dataset/train_data_laptop.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            rev = line[1]
            # Punctuation removal
            for c in string.punctuation:
                rev = rev.replace(c,"")
            # Stopword removal
            reviews.append(rev)

    outfile = open('reviews.pickle','wb')
    pickle.dump(reviews ,outfile)
    outfile.close()

    return reviews

# List of unique words
def generate_unique_words(reviews):
    print("Unique Words Generation")
    unique_words = []
    words = []
    for i in range(1, len(reviews)):
        rev = word_tokenize(reviews[i])
        for j in rev:
            if j not in stop_words:
                words.append(j)
                if j not in unique_words:
                    unique_words.append(j)

    outfile = open('unique_words.pickle','wb')
    pickle.dump(unique_words ,outfile)
    outfile.close()

    return unique_words

if __name__ == '__main__':
    reviews = generate_dataset()
    unique_words = generate_unique_words(reviews)
    print(unique_words)
