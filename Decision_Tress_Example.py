import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree

df = fetch_california_housing()

#check out array shape
df_shape = df.data.shape

dtr = tree.DecisionTreeRegressor(max_depth=2,criterion= "pass" )
dtr.fit(df.data[:,[6,7]],df.target)

#visulazie
dot_data = tree.export_graphviz(dtr, out_file = None, feature_names =df.feature_names[6:8], filled = True,impurity = False, rounded = True)

import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")

import  os
os.environ["PATH"] += os.pathsep + 'd:/software/bin/'  # set environment path

from IPython.display import Image
Image(graph.create_png())
graph.write_png("dtr_white_background.png")


from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(df.data, df.target, test_size = 0.1, random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train, target_train)
dtr.score(data_test, target_test)


# To select the best parameter
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
tree_param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))}
grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5) # cv means how many pieces we divide our train dataset to cross validation
grid.fit(data_train, target_train)

print(grid.best_score_,grid.best_params_,grid.cv_results_)





























