#Qn.1 A prgram to demonstrate the workingo f Dimenstionality reduction using PCA\
from sklearn import datasets # Dataset from which we are to load the iris_dataset\
iris = datasets.load_iris() # iris_dataset \

from sklearn.decomposition import PCA # PCA used for dimensionality reduction\

pca = PCA(n_components=.99) # PCA\

features = iris.data #features\
target = iris.target # target \

# Transforming Features\
features_pca = pca.fit_transform(features) # Reduced Features\
print("Original number of features:", features.shape[1]) # Original\
print("Reduced number of features:", features_pca.shape[1]) # Reduced Features
