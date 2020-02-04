
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA as pca
from sklearn.decomposition import factor_analysis as fact
from factor_analyzer import FactorAnalyzer
from sklearn import preprocessing
import sklearn.metrics as metcs
from scipy.cluster import hierarchy as hier
from sklearn import cluster as cls
from sklearn.preprocessing import StandardScaler


#For the tree
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

os.getcwd()
os.chdir("//Users//nikhilviswanath//Documents//python_data")

reduce_data= pd.read_csv('calihospital.txt',sep='\t')
rows,columns=reduce_data.shape
reduce_data.columns
reduce_data.dtypes
reduce_data.head()
reduce_data['Teaching'] = reduce_data['Teaching'].astype('category')  
reduce_data['TypeControl'] = reduce_data['TypeControl'].astype('category')  
reduce_data['DonorType'] = reduce_data['DonorType'].astype('category')


reduce_data_pca=reduce_data[['NoFTE','NetPatRev','InOperExp','OutOperExp','OperRev','OperInc','AvlBeds',
'Compensation','MaxTerm']]
sc = StandardScaler()
reduce_data_std = sc.fit_transform(reduce_data_pca)


pca_result = pca(n_components=9).fit(reduce_data_std)
pca_result.explained_variance_
pca_result.components_.T * np.sqrt(pca_result.explained_variance_)

plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xlim(0.25,4.25) 
plt.ylim(0,1.05) 
plt.xticks([1,2,3,4,5,6,7,8,9])

###Factor Analysis###

reduce_data_fac=reduce_data[['NoFTE','NetPatRev','InOperExp','OutOperExp','OperRev','OperInc','AvlBeds',
'Compensation','MaxTerm']]

fa = FactorAnalyzer(rotation = 'varimax', n_factors = 3)
fa.fit(reduce_data_fac)
fa.loadings_
fa.get_communalities()


#pd.DataFrame(fa.components_, columns = reduce_data_fac.feature_names)
###Incomplete####

#K-means cluster analysis##

km = cls.KMeans(n_clusters=3).fit(reduce_data.loc[:,['NoFTE','NetPatRev','InOperExp','OutOperExp',
'OperRev','OperInc','AvlBeds',
'Compensation','MaxTerm']])

km.labels_

reduce_data.Teaching.unique()
reduce_data.DonorType.unique()
reduce_data.dtypes

reduce_data['Teaching'] = reduce_data['Teaching'].astype('object')
reduce_data.Teaching.replace(['Small/Rural','Teaching'],[1,2], inplace=True)
reduce_data['Teaching'] = reduce_data['Teaching'].astype('category')

cm = metcs.confusion_matrix(reduce_data.Teaching, km.labels_)
print(cm)

##cluster analysis using TypeControl###

reduce_data['TypeControl'] = reduce_data['TypeControl'].astype('object')
reduce_data.TypeControl.replace(['District','Non Profit','Investor','City/County'],[1,2,3,4], inplace=True)
reduce_data['TypeControl']= reduce_data['TypeControl'].astype('category')

km1 = cls.KMeans(n_clusters=3).fit(reduce_data.loc[:,['NoFTE','NetPatRev','InOperExp','OutOperExp',
'OperRev','OperInc','AvlBeds',
'Compensation','MaxTerm']])
km1.labels_

cm1 = metcs.confusion_matrix(reduce_data.TypeControl, km1.labels_)
print(cm1)

####Cluster analysis using DonorType###

reduce_data['DonorType']=reduce_data['DonorType'].astype('object')
reduce_data.DonorType.replace(['Charity','Alumni'],[1,2],inplace=True)
reduce_data['DonorType']=reduce_data['DonorType'].astype('category')

km2 = cls.KMeans(n_clusters=3).fit(reduce_data.loc[:,['NoFTE','NetPatRev','InOperExp','OutOperExp',
'OperRev','OperInc','AvlBeds',
'Compensation','MaxTerm']])
km2.labels_

cm2 = metcs.confusion_matrix(reduce_data.DonorType, km2.labels_)
print(cm2)

####Graphical confusion matrix #####

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([1,2], ['Small/Rural','Teaching'])





