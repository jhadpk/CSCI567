import numpy as np


#################################
# DO NOT IMPORT OHTER LIBRARIES
#################################

def get_center_by_custom_logic(generator, distance_to_closest_centroid):
    # https://piazza.com/class/kdzgr2vpest74w?cid=232
    r = generator.rand()
    cumulative_prob = 0
    for index in range(len(distance_to_closest_centroid)):
        cumulative_prob += distance_to_closest_centroid[index]
        if cumulative_prob > r:
            return index

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    """
    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data - numpy array of points
    :param generator: random number generator. Use it in the same way as np.random.
            In grading, to obtain deterministic results, we will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being the *index* of a sample
             chosen as centroid.
    """
    first_center_index = generator.randint(0, n)
    centers_point = []
    centers = []
    centers_point.append(x[first_center_index])
    centers.append(first_center_index)

    for k in range(n_cluster - 1):
        distance_to_closest_centroid = []
        for i in range(len(x)):
            min_distance = float("inf")
            for j in range(len(centers_point)):
                center = centers_point[j]
                distance = np.sum((x[i] - center)**2)
                if distance < min_distance:
                    min_distance = distance
            distance_to_closest_centroid.append(min_distance)
        sum_distances = sum(distance_to_closest_centroid)
        for m in range(len(distance_to_closest_centroid)):
            distance_to_closest_centroid[m] = distance_to_closest_centroid[m] / sum_distances
        index = get_center_by_custom_logic(generator, distance_to_closest_centroid)
        centers.append(index)
        centers_point.append(x[index])
    # DO NOT CHANGE CODE BELOW THIS LINE
    return centers


# Vanilla initialization method for KMeans
def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)


class KMeans:
    """
        Class KMeans:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    """

    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        """
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple in the following order:
                  - final centroids, a n_cluster X D numpy array,
                  - a length (N,) numpy array where cell i is the ith sample's assigned cluster's index (start from 0),
                  - number of times you update the assignment, an Int (at most self.max_iter)
        """
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        self.generator.seed(42)
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        centroids = np.zeros([self.n_cluster, D])
        for i in range(self.n_cluster):
            centroids[i] = x[self.centers[i]]

        y = np.zeros(N)
        distortion = np.sum([np.sum((x[y == i] - centroids[i]) ** 2) for i in range(self.n_cluster)]) / N

        iter_count = 0
        while iter_count < self.max_iter:
            iter_count += 1
            y = np.argmin(np.sum(((x - np.expand_dims(centroids, axis=1)) ** 2), axis=2), axis=0)
            distortion_updated = np.sum([np.sum((x[y == i] - centroids[i]) ** 2) for i in range(self.n_cluster)]) / N
            if np.absolute(distortion - distortion_updated) <= self.e:
                break
            distortion = distortion_updated
            centroids_new = np.array([np.mean(x[y == i], axis=0) for i in range(self.n_cluster)])
            index = np.where(np.isnan(centroids_new))
            centroids_new[index] = centroids[index]
            centroids = centroids_new
        self.max_iter = iter_count
        return centroids, y, self.max_iter


class KMeansClassifier:
    """
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of clusters for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
    """

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        """
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Store following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (numpy array of length n_cluster)
        """

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        k_means = KMeans(self.n_cluster, self.max_iter, self.e, self.generator)
        centroids, assignment, num_iter = k_means.fit(x, centroid_func)
        centroids = np.array(centroids)
        assigned_label = [[] for i in range(self.n_cluster)]
        for i in range(N):
            assigned_label[assignment[i]].append(y[i])

        centroid_labels = np.zeros([self.n_cluster])
        for i in range(self.n_cluster):
            centroid_labels[i] = np.argmax(np.bincount(assigned_label[i]))

        # DO NOT CHANGE CODE BELOW THIS LINE
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape

        l2_norm = np.zeros([self.n_cluster, N])
        for k in range(self.n_cluster):
            l2_norm[k] = np.sqrt(np.sum(np.power((x - self.centroids[k]), 2), axis=1))
        nearest_centroid = np.argmin(l2_norm, axis=0)
        # labels = self.centroid_labels[nearest_centroid]
        labels = [[] for i in range(N)]
        for i in range(N):
            labels[i] = self.centroid_labels[nearest_centroid[i]]

        return np.array(labels)


def transform_image(image, code_vectors):
    """
        Quantize image using the code_vectors (aka centroids)

        Return a new image by replacing each RGB value in image with the nearest code vector
          (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    """

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    R, G, B = image.shape
    img_reshape = image.reshape(R * G, B)
    nearest_index = np.argmin(np.sum(((img_reshape - np.expand_dims(code_vectors, axis=1)) ** 2), axis=2), axis=0)
    transformed_img = code_vectors[nearest_index].reshape(R, G, B)
    return transformed_img

