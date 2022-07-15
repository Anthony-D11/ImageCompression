import cv2
from numpy import reshape, zeros, random, sum, min, multiply, where
import matplotlib.pyplot as plt

def KMeansInitCentroids(X, K):
    random_index = random.permutation(X.shape[0])
    centroids = X[random_index[1:K]][:]
    return centroids
def runKMeans(X, initial_centroids, max_iters, plot_progress=False):
    (m, n) = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    index = zeros((m, 1))
    for i in range(max_iters):
        print(f'KMeans iteration {i}/{max_iters}\n')
        index = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, index, K)
    return centroids, index

def findClosestCentroids(X, centroids):
    K = centroids.shape[0]
    index = zeros((X.shape[0], 1))
    sub = zeros((1, K))
    for i in range(X.shape[0]):
        for j in range(K):
            sub[j] = sum(multiply(X[i][:] - centroids[j][:], X[i][:] - centroids[j][:]))
        index[i] = where(sub == min(sub))
    return index

def computeCentroids(X, index, K):
    m, n = X.shape
    centroids = zeros((K, n))
    for i in range(K):
        sum = zeros((1, n))
        count = 0
        for j in range(m):
            if index[j] == i:
                sum += X[j][:]
                count+=1
        centroids[i][:] = sum/count
    return centroids

A = (cv2.imread('bird_small.png'))
A = A / 255

image_size = A.shape

X = reshape(A, (image_size[0] * image_size[1], image_size[2]))

K = 16
max_iters = 10

initial_centroids = KMeansInitCentroids(X, K)
centroids, index = runKMeans(X, initial_centroids, max_iters)
print('\nApplying K-Means to compress an image.\n\n')

index = findClosestCentroids(X, centroids)

X_recovered = centroids[index][:]

X_recovered = reshape(X_recovered, image_size[0], image_size[1], 3)

plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()