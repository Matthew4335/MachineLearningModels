import numpy as np
from pca import PCA
from regression import Regression


class Slope(object):

    def __init__(self):
        pass

    @staticmethod
    def pca_slope(X, y):
        """		
		Calculates the slope of the first principal component given by PCA
		
		Args:
		    x: N x 1 array of feature x
		    y: N x 1 array of feature y
		Return:
		    slope: (float) scalar slope of the first principal component
		"""
        data = np.hstack((X, y))
        pca = PCA()
        pca.fit(data)
        slope = pca.get_V()[0][1] / pca.get_V()[0][0]

        return slope
        

    @staticmethod
    def lr_slope(X, y):
        """		
		Calculates the slope of the best fit returned by linear_fit_closed()
		
		For this function don't use any regularization
		
		Args:
		    X: N x 1 array corresponding to a dataset
		    y: N x 1 array of labels y
		Return:
		    slope: (float) slope of the best fit
		"""
        reg = Regression()
        x_bias = np.ones((np.shape(X)[0], 2))
        x_bias[:, 1] = X.flatten()
        return reg.linear_fit_closed(xtrain=x_bias, ytrain=y)[1][0]

    @classmethod
    def addNoise(cls, c, x_noise=False, seed=1):
        """		
		Creates a dataset with noise and calculates the slope of the dataset
		using the pca_slope and lr_slope functions implemented in this class.
		
		Args:
		    c: (float) scalar, a given noise level to be used on Y and/or X
		    x_noise: (Boolean) When set to False, X should not have noise added
		            When set to True, X should have noise.
		            Note that the noise added to X should be different from the
		            noise added to Y. You should NOT use the same noise you add
		            to Y here.
		    seed: (int) Random seed
		Return:
		    pca_slope_value: (float) slope value of dataset created using pca_slope
		    lr_slope_value: (float) slope value of dataset created using lr_slope
		"""
        np.random.seed(seed)
        X = np.arange(0.001, 1.001, 0.001).reshape(-1, 1)
        if x_noise == False:
            X += np.random.normal(loc=[0], scale=c, size=X.shape)
        y = 5 * X + np.random.normal(loc=[0], scale=c, size=X.shape)
        pca_slope_value = cls.pca_slope(X, y)
        lr_slope_value = cls.lr_slope(X, y)
        return pca_slope_value, lr_slope_value
