import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        Xcentered = X - np.mean(X, axis = 0)
        self.U, self.S, self.V = np.linalg.svd(Xcentered, full_matrices=False)

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        datacentered = data - np.mean(data, axis=0)
        return np.dot(datacentered, self.V[:K].T)

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        datacentered = data - np.mean(data, axis=0)
        variance = np.cumsum(self.S**2) / np.sum(self.S**2)
        K = 0
        for i in range(len(variance)):
            if variance[i] >= retained_variance:
                K = i + 1
                break
        return np.dot(datacentered, self.V[:K].T)

    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) ->None:
        """		
		You have to plot three different scatterplots (2d and 3d for strongest 2 features and 2d for weakest 2 features) for this function. For plotting the 2d scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later weakest) features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
		Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
		Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
		Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.
		Hint: To extract weakest features, consider the order of components returned in PCA.
		
		Args:
		    xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
		    ytrain: (N,) numpy array, the true labels
		
		Return: None
		"""
        self.fit(X)
    
        strongestTwoFeatures = self.transform(X, K=2)
        
        df = pd.DataFrame(strongestTwoFeatures, columns=['Feature 1', 'Feature 2'])
        df['label'] = y
    
        fig1 = px.scatter(df, x='Feature 1', y='Feature 2', color='label', title=f"{fig_title} - Strongest 2")
        fig1.show()

        strongestThreeFeatures = self.transform(X, K=3)

        df_3d = pd.DataFrame(strongestThreeFeatures, columns=['Feature 1', 'Feature 2', 'Feature 3'])
        df_3d['label'] = y

        fig2 = px.scatter_3d(df_3d, x='Feature 1', y='Feature 2', z='Feature 3', color='label', title=f"{fig_title} - Strongest 3")
        fig2.show()
        
        weakestTwoFeatures = np.dot(X - np.mean(X, axis=0), self.V.T)[:, -2:]

        df_weakest = pd.DataFrame(weakestTwoFeatures, columns=['Feature 1', 'Feature 2'])
        df_weakest['label'] = y
        
        fig3 = px.scatter(df_weakest, x='Feature 1', y='Feature 2', color='label', title=f"{fig_title} - Weakest 2")
        fig3.show()
