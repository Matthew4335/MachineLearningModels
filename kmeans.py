"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""

import numpy as np


class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        return self.points[np.random.choice(self.points.shape[0], self.K, replace=False), :]

    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        raise NotImplementedError

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Donot use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        self.assignments = np.argmin(pairwise_dist(self.points, self.centers), axis = 1)
        return self.assignments

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        centers = []
        for index in range(self.K):
            points = self.points[self.assignments == index]
            if len(points) != 0:
                centers.append(np.mean(points, axis = 0))
        self.centers = np.array(centers)
        return self.centers

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        loss = 0.0
        for index in range(self.K):
            center = self.centers[index]
            points = self.points[self.assignments == index]
            distance =np.sum(np.sum(np.square(points - center), axis = 1))
            loss += distance
        self.loss = loss
        return self.loss


    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        for index in range(self.max_iters):
            self.update_assignment()
            self.update_centers()
            num_empty_cluster = self.K - self.centers.shape[0]
            if num_empty_cluster > 0:
                new_centers = self.points[np.random.choice(self.points.shape[0], num_empty_cluster, replace=False), :]
                self.centers = np.concatenate((self.centers, new_centers), axis=0)
                self.update_assignment()
            previous_loss = self.loss
            self.get_loss()
            difference = 1
            if previous_loss != 0:
                difference = (previous_loss - self.loss) / previous_loss
            if difference < self.rel_tol:
                break
        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    X_squared = np.sum(x**2, axis=1)
    Y_squared = np.sum(y**2, axis=1)
    XY = np.dot(x, y.T)
    dist = np.sqrt(X_squared[:, np.newaxis] + Y_squared - 2 * XY)
    return dist


def rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate rand statistic value
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(len(xPredicted)):
        for j in range(i + 1, len(xPredicted)):
            if xPredicted[i] == xPredicted[j]:
                if xGroundTruth[i] == xGroundTruth[j]:
                    TP += 1
                else:
                    FP += 1
            else:
                if xGroundTruth[i] != xGroundTruth[j]:
                    TN += 1
                else:
                    FN += 1
    rand_statistic = (TP + TN) / (TP + TN + FP + FN)
    return rand_statistic
