import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal
import turicreate as tc
import colorsys
import copy
import array

from PIL import Image
from io import BytesIO


# implementing the EM algorithm for Gaussian mixture models

# log likelihood for mixture of Gaussians
# log likelihood quantifies the probability of observing a given set of data under a particular setting of the
# parameters in our model

def log_sum_exp(Z):
    """ Compute log(sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])

    ll = 0
    for d in data:

        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            # compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1 / 2. * (num_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)

        # increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)

    return ll


# E Step: assign cluster responsibilities, given current parameters
def compute_responsibilities(data, weights, means, covariances):
    """E-step: compute responsibilities, given the current parameters"""
    num_data = len(data)
    num_clusters = len(means)
    resp = np.zeros((num_data, num_clusters))

    # update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
    # to compute likelihood of seeing data point i given cluster k, use multivariate_normal.pdf
    for i in range(num_data):
        for k in range(num_clusters):
            resp[i, k] = weights[k] * multivariate_normal.pdf(data[i], means[k], covariances[k])

    # add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums

    return resp


# M step: update parameters, given current cluster responsibilities

# before updating the parameters, we first compute what is known as "soft counts".
# the soft count of a cluster is the sum of all cluster responsibilities for that cluster:
def compute_soft_counts(resp):
    # compute the total responsibility assigned to each cluster, which will be useful when
    # implementing M-steps below. In the lectures this is called N^{soft}
    counts = np.sum(resp, axis=0)
    return counts


# The cluster weights show us how much each cluster is represented over all data points.
# The weight of cluster k is given by the ratio of the soft count $N^{\text{soft}}_{k}$ to the total number of
# data points N
def compute_weights(counts):
    num_clusters = len(counts)
    weights = [0.] * num_clusters

    for k in range(num_clusters):
        # update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        weights[k] = counts[k] / counts.sum()

    return weights


def compute_means(data, resp, counts):
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters

    for k in range(num_clusters):
        # update means for cluster k using the M-step update rule for the mean variables.
        # this will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        weighted_sum = 0.
        for i in range(num_data):
            weighted_sum += resp[i, k] * data[i]

        means[k] = weighted_sum / counts[k]

    return means


def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim, num_dim))] * num_clusters

    for k in range(num_clusters):
        # update covariances for cluster k using the M-step update rule for covariance variables.
        # this will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            weighted_sum += resp[i, k] * np.outer(data[i] - means[k], data[i] - means[k])
        covariances[k] = weighted_sum / counts[k]

    return covariances


def expectation_maximization(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    # make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]

    # infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)

    # initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]

    for it in range(maxiter):
        if it % 5 == 0:
            print("Iteration %s" % it)

        # E step: compute responsibilities
        resp = compute_responsibilities(data, weights, means, covariances)

        # M step
        # compute the total responsibility assigned to each cluster, which will be useful when
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = compute_soft_counts(resp)

        # update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        weights = compute_weights(counts)

        # update means for cluster k using the M-step update rule for the mean variables.
        # this will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        means = compute_means(data, resp, counts)

        # update covariances for cluster k using the M-step update rule for covariance variables.
        # this will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        covariances = compute_covariances(data, resp, counts, means)

        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest

    if it % 5 != 0:
        print("Iteration %s" % it)

    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out


# testing the implementation on the simulated data
def generate_MoG_data(num_data, means, covariances, weights):
    """ Creates a list of data points """
    num_clusters = len(weights)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and weights to pick a cluster id greater than or equal to 0 and less than num_clusters.
        k = np.random.choice(len(weights), 1, p=weights)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covariances[k])

        data.append(x)
    return data


init_means = [
    [5, 0],  # mean of cluster 1
    [1, 1],  # mean of cluster 2
    [0, 5]  # mean of cluster 3
]
init_covariances = [
    [[.5, 0.], [0, .5]],  # covariance of cluster 1
    [[.92, .38], [.38, .91]],  # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
init_weights = [1 / 4., 1 / 2., 1 / 4.]  # weights of each cluster

# generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

plt.figure()
d = np.vstack(data)
plt.plot(d[:, 0], d[:, 1], 'ko')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

np.random.seed(4)

# initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

# Run EM
results = expectation_maximization(data, initial_means, initial_covs, initial_weights)


# deprecated
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data], 'ko')  # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


plot_contours(data, initial_means, initial_covs, 'Initial clusters')

results = expectation_maximization(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters')

results = expectation_maximization(data, initial_means, initial_covs, initial_weights, maxiter=12)
plot_contours(data, results['means'], results['covs'], 'Clusters after 12 iterations')

results = expectation_maximization(data, initial_means, initial_covs, initial_weights)
loglikelihoods = results["loglik"]

plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

# fitting a gaussian mixture for image data

images = tc.SFrame('images.sf')
tc.canvas.set_target('ipynb')

images['rgb'] = images.pack_columns(['red', 'green', 'blue'])['X4']
images.show()


np.random.seed(1)

# Initalize parameters
init_means = [images['rgb'][x] for x in np.random.choice(len(images), 4, replace=False)]
cov = np.diag([images['red'].var(), images['green'].var(), images['blue'].var()])
init_covariances = [cov, cov, cov, cov]
init_weights = [1/4., 1/4., 1/4., 1/4.]

# Convert rgb data to numpy arrays
img_data = [np.array(i) for i in images['rgb']]

# Run our EM algorithm on the image data using the above initializations.
# This should converge in about 125 iterations
out = expectation_maximization(img_data, init_means, init_covariances, init_weights)

# evaluating convergence

ll = out['loglik']
plt.plot(range(len(ll)), ll, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()

plt.figure()
plt.plot(range(3, len(ll)), ll[3:], linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size': 16})
plt.tight_layout()


# evaluating uncertainity
# explore the evolution of cluster assignment and uncertainty. Remember that the EM algorithm represents uncertainty
# about the cluster assignment of each data point through the responsibility matrix. Rather than making a 'hard'
# assignment of each data point to a single cluster, the algorithm computes the responsibility of each cluster for each
# data point, where the responsibility corresponds to our certainty that the observation came from that cluster.

# to make things easier we will plot the data using only two dimensions, taking just the [R G], [G B] or [R B] values
# instead of the full [R G B] measurement for each observation.
def plot_responsibilities_in_RB(img, resp, title):
    N, K = resp.shape

    HSV_tuples = [(x * 1.0 / K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    R = img['red']
    B = img['blue']
    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [tuple(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n])
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()



N, K = out['resp'].shape
random_resp = np.random.dirichlet(np.ones(K), N)
plot_responsibilities_in_RB(images, random_resp, 'Random responsibilities')

out = expectation_maximization(img_data, init_means, init_covariances, init_weights, maxiter=1)
plot_responsibilities_in_RB(images, out['resp'], 'After 1 iteration')


out = expectation_maximization(img_data, init_means, init_covariances, init_weights, maxiter=20)
plot_responsibilities_in_RB(images, out['resp'], 'After 20 iterations')

# interpreting each cluster
# Let's dig into the clusters obtained from our EM implementation. Recall that our goal in this section is to cluster
# images based on their RGB values. We can evaluate the quality of our clustering by taking a look at a few images that
# 'belong' to each cluster. We hope to find that the clusters discovered by our EM algorithm correspond to different
# image categories - in this case, we know that our images came from four categories ('cloudy sky', 'rivers', 'sunsets',
# and 'trees and forests'), so we would expect to find that each component of our fitted mixture model roughly
# corresponds to one of these categories.
#
# If we want to examine some example images from each cluster, we first need to consider how we can determine cluster
# assignments of the images from our algorithm output. This was easy with k-means - every data point had a 'hard'
# assignment to a single cluster, and all we had to do was find the cluster center closest to the data point of
# interest. Here, our clusters are described by probability distributions (specifically, Gaussians) rather than single
# points, and our model maintains some uncertainty about the cluster assignment of each observation.
#
# One way to phrase the question of cluster assignment for mixture models is as follows: how do we calculate the
# distance of a point from a distribution? Note that simple Euclidean distance might not be appropriate since
# (non-scaled) Euclidean distance doesn't take direction into account. For example, if a Gaussian mixture component is
# very stretched in one direction but narrow in another, then a data point one unit away along the 'stretched' dimension
# has much higher probability (and so would be thought of as closer) than a data point one unit away along the 'narrow'
# dimension.
#
# In fact, the correct distance metric to use in this case is known as Mahalanobis distance. For a Gaussian
# distribution, this distance is proportional to the square root of the negative log likelihood. This makes sense
# intuitively - reducing the Mahalanobis distance of an observation from a cluster is equivalent to increasing that
# observation's probability according to the Gaussian that is used to represent the cluster. This also means that we
# can find the cluster assignment of an observation by taking the Gaussian component for which that observation scores
# highest. We'll use this fact to find the top examples that are 'closest' to each cluster.

# calculate cluster assignments for the entire image dataset using the result of running EM for 20 iterations above:
weights = out['weights']
means = out['means']
covariances = out['covs']
rgb = images['rgb']
N = len(images)  # number of images
K = len(means)  # number of clusters

assignments = [0] * N
probs = [0] * N

for i in range(N):
    # Compute the score of data point i under each Gaussian component:
    p = np.zeros(K)
    for k in range(K):
        p[k] = weights[k] * multivariate_normal.pdf(rgb[i], mean=means[k], cov=covariances[k])

    # Compute assignments of each data point to a given cluster based on the above scores:
    assignments[i] = np.argmax(p)

    # For data point i, store the corresponding score under this cluster assignment:
    probs[i] = np.max(p)

assignments = tc.SFrame({'assignments': assignments, 'probs': probs, 'image': images['image']})


def get_top_images(assignments, cluster, k=5):
    # YOUR CODE HERE
    images_in_cluster = ...
    top_images = images_in_cluster.topk('probs', k)
    return top_images['image']


for component_id in range(4):
    get_top_images(assignments, component_id).show()