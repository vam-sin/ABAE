# Performs k means on the values present in E into 14 different centers and initializes the value of T
from sklearn.cluster import KMeans
import pickle

if __name__ == '__main__':

    infile = open('E.pickle','rb')
    E = pickle.load(infile)
    infile.close()

    kmeans = KMeans(n_clusters = 14, random_state = 0).fit(E)
    cluster_centres = kmeans.cluster_centers_

    T = []
    for i in cluster_centres:
        T.append(i)

    print(len(T))

    outfile = open('T.pickle','wb')
    pickle.dump(T ,outfile)
    outfile.close()

    print("Aspect Embeddings Matrix Generated")
