import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
from sklearn import metrics

#For displaying the tree
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus

os.getcwd()
dir="//Users//nikhilviswanath//Documents//python_data"
os.chdir(dir)

FileExistsError("reduction_data_new.txt")

reduction_data=pd.read_csv("reduction_data_new.txt",sep='\t')
reduction_data.dtypes
reduction_data.head()
reduction_data.intent01.unique()
rows,cols=reduction_data.shape


##Imputation of missing values###   
reduction_data.fillna(reduction_data.mean(), inplace=True)

#Dropping dependent variable
reduction_data2 = reduction_data.drop( ['intent01']  , axis=1 )

col_names = list(reduction_data2.columns.values)

tre1 = tree.DecisionTreeRegressor().fit(reduction_data2,reduction_data.intent01)

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

### The above tree is too large and has too many branches where###
### the sample size is too small. 


tre1 = tree.DecisionTreeRegressor(min_samples_split=20,min_samples_leaf=20)

tre1.fit(reduction_data2,reduction_data.intent01)

dot_data = StringIO()
tree.export_graphviz(tre1, out_file=dot_data,
                     feature_names=col_names,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

####################################################

### Titanic data #######


# read data
titanic_data=pd.read_csv("titanic_data.txt",sep='\t')
titanic_data.head()
titanic_data.dtypes
rows,cols=titanic_data.shape

#Performing variable coding to use in decision tree####

titanic_data['Class'].replace(['1st','2nd','3rd','Crew'],[0,1,2,3], inplace=True)
titanic_data['Sex'].replace(['Female','Male'],[0,1], inplace=True)
titanic_data['Age'].replace(['Child','Adult'],[0,1], inplace=True)

titanic_data2 = titanic_data.drop( ['Survived']  , axis=1 )

### Creating the basic decision tree with categorical variable##

col_names = list(titanic_data2.columns.values)
classnames = ["Yes","No"]
tre3 = tree.DecisionTreeClassifier().fit(titanic_data2,titanic_data['Survived'])


dot_data = StringIO()
tree.export_graphviz(tre3, out_file=dot_data,
                     feature_names=col_names,
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))


#confusion matrix
predicted = tre3.predict(titanic_data2)
print(metrics.classification_report(titanic_data['Survived'], predicted))

cm = metrics.confusion_matrix(titanic_data['Survived'], predicted)
print(cm)

plt.matshow(cm)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['Yes','No'])

###Pruning the tree by adding conditions on the number ####
####of branch nodes and sample for splitting the nodes###

tre4 = tree.DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=20).fit(titanic_data2,titanic_data['Survived'])

dot_data = StringIO()
tree.export_graphviz(tre4, out_file=dot_data,
                     feature_names=col_names,
                     class_names=classnames,
                     filled=True,
                     rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))


#confusion matrix
predicted = tre4.predict(titanic_data2)
print(metrics.classification_report(titanic_data['Survived'], predicted))

cm1 = metrics.confusion_matrix(titanic_data['Survived'], predicted)
print(cm1)

plt.matshow(cm1)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.xticks([0,1], ['Yes','No'])

#####################################################################
######################################################################

