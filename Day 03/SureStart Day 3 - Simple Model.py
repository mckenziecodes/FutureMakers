#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.datasets import load_iris
from sklearn import tree

#Load in the dataset
iris_data = load_iris()

#Initialize our decision tree object
classification_tree = tree.DecisionTreeClassifier()

#Train our decision Tree (tree induction + pruning)
classification_tree = classification_tree.fit(iris_data.data, iris_data.target)

import graphviz 
#when I run this code, I recief an error that says there is no module named 'graphviz'
dot_data = tree.export_graphvis(classification_tree, out_file=None, 
                                feature_names=iris.feature_names,
                                class_names=iris.target_names,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")


# In[7]:





# In[ ]:





# In[ ]:




