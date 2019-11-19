from gensim.models import Word2Vec
import pickle

def gen_word_embeddings_matrix_E(global_words):
    '''
    Parameters: Takes in all the words in the words in the reviews.
    Returns: The Global Embeddings matrix E
    '''
    print("Generating Global Embeddings Matrix E")
    E = []
    for i in global_words:
        print(i)
        vec = model[i]
        E.append(vec)

    outfile = open('E.pickle','wb')
    pickle.dump(E ,outfile)
    outfile.close()

    return E

def get_word_embedding(word, E, unique_words):
    '''
    Parameters: word for which we want the vector, The global word embeddings matrix M,
                list of unique words.
    Return: The word vector.
    '''
    return E[unique_words.index(word)]

if __name__ == '__main__':

    # Load pre-trained model
    model = Word2Vec.load('model.bin')

    infile = open('unique_words.pickle','rb')
    unique_words = pickle.load(infile)
    infile.close()

    E = gen_word_embeddings_matrix_E(unique_words)
