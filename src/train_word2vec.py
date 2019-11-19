from gensim.models import Word2Vec
import pickle
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

# define training data
infile = open('reviews.pickle','rb')
reviews = pickle.load(infile)
infile.close()

sentences = []

for i in range(1, len(reviews)):
    sentences.append(word_tokenize(reviews[i]))

# train model
model = Word2Vec(sentences, size = 200, window = 10, min_count = 1, workers = 4)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['charge'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
