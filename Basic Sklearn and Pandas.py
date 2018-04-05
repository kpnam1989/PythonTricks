# Source: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook
import os
import pandas as pd
import numpy as np

os.chdir(r"C:\Users\nkieu\Desktop\Python\Python on Machine Learning\Housing Price")
df_train = pd.read_csv('train.csv')

###########################
#
# Basic methods
#
###########################
df_train = pd.get_dummies(df_train) # convert to dummies
df_train.describe()
df_train['SalePrice'].min() # also, imin, imax
df_train['SalePrice'].max()
df_train.shape
df_train.dtype
df_train.columns

x = np.array(range(3))
(x > 3).any()
(x > 3).any().any()
np.any(x>3)
np.all(x<4)
np.zeros((2,3))
np.zeros_like(x)
df.idxmax(axis = 0) #axis = 0 => per column
df[df.Length > 7]
np.where() # broadcasting


###########################
#
# Plotting the distribution
#
###########################
from scipy.stats import norm
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()

df_train['SalePrice'].plot.hist()
df_train['SalePrice'].plot.kde()

import scipy.stats
res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)

xx = train_df[[var_name, 'y']].groupby(var_name).mean().sort_values('y')


###########################
#
# Plotting capacity
#
###########################
ax1.axvline(0.65) # vertical line
ax2.axhline(0.45) # horizontal line
ax3.scatter(np.linspace(0, 1, 5), np.linspace(0, 5, 5))
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
df_train['SalePrice'].plot.hist()
df_train['SalePrice'].plot.kde()

var = 'GrLivArea'
df_train.plot.scatter(x = var, y = 'SalePrice')

# Scatter plot matrix
axs = pd.scatter_matrix(alldata[scattercols],
                        figsize=(12, 12), c='red')

train_df[['y','X1']].groupby('X1').boxplot() # The simple version
train_df.boxplot(column = 'y', by = 'X1')    # Better version
df_train.plot.box(x = var, y = 'SalePrice')
df_train.groupby(var).plot.box(x = 'SalePrice')
df_train[['SalePrice', var]].groupby(var).boxplot()
df_train[['SalePrice', var]].groupby(var).plot.box(x = 'SalePrice')
df_train.boxplot(column = 'SalePrice', by = var)

sns.boxplot(x = var, y = 'SalePrice', data = df_train)
#sns.boxplot has a 'order' argument that we can use to change the order of the graph

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)

# Correlation and heatmap
# vmax = value to anchor the map
# square = change the size of each cell so that we have a Square heatmap in the end
corrmat = df_train.corr()
sns.heatmap(corrmat, vmax = .8, square=True)
sns.heatmap(corrmat)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
#### Where cm is defined below
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cols = corrmat.nlargest(10, 'SalePrice').index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


###########################
#
# Bar plot
#
###########################
plt.barh(range(numFeatures), width = importance[index_sort[:numFeatures]], align = 'center', tick_label = train_X.columns[index_sort[:numFeatures]])

importance = model.get_fscore()
ss = sorted(importance, key = importance.get, reverse = True) # sorted is a python function
top_names = ss[0:50]
plt.barh(range(50), width = [importance.get(i) for i in top_names], align = 'center', tick_label = top_names)

# left: give x-coordinate of the left sides of the bar
# I add some extra distance in between to see the impact of parameter left
p3 = plt.barh(ind, one_count_list, width, left = zero_count_list_1000, color="blue")
p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="blue")

# From Green Taxi
fig, ax = plt.subplots(1, 2, figsize = (15, 4))
h = ...
plt.bar(h.index, h.values, width = .4, color = 'b')
h = ..
ax[1].bar(h.index + 0.4, h.values, width = .4, color = 'g')

ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2]) # bar horizontal

col_order_y = list(xx.index)
sns.stripplot(x = var_name, y = 'y', data = train_df, order = col_order)
sns.boxplot(x = var_name, y = 'y', data = train_df, order = col_order)
sns.violinplot(x = var_name, y = 'y', data = train_df, order = col_order)


###########################
#
# Plottong and subplotting
#
###########################
fig = plt.figure(figsize = (3,3))
fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
ax = fig.add_subplot(8,8, i +1, xticks = [], yticks = [])
ax.imshow(digits.images[i], cmap =plt.cm.binary, interpolation = 'nearest')
ax.text(0, 7, str(digits.target[i]))

plt.xticks(rotation=90);
sns.set(font_scale=1.25)


###########################
#
# Handling missing data
#
###########################
total = df_train.isnull().sum() # Sum missing values by columns
total.sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count())
percent = percent.sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1)
missing_data.columns = ['Total', 'Percent']

dropVariables = missing_data[missing_data['Total'] > 1].index # Slicing with a boolean list
oneObservationToDrop = df_train.loc[df_train['Electrical'].isnull()]
oneObservationToDrop.index # to get the index of the row


###########################
#
# Handling existing datasets
#
###########################
import pandas as pd
from sklearn import datasets
dir(datasets)

# Some simple datasets
datasets.load_boston()
datasets.load_diabetes

digits_data = pd.read_csv(r"C:\Users\nkieu\Desktop\Python\Python on Machine Learning\Input\optdigits.tra",header = None)
                          
# digits is of type BUNDLE
# There is a lot more data than just the csv type
digits = datasets.load_digits()
dir(digits)
print digits.DESCR
digits.data.shape
digits.target
digits.data[:5,:3]
df = pd.DataFrame(digits.data)


###########################
#
# PCA
# PCA can be used to see how correlated the data are
#
###########################
from sklearn import decomposition

randomized_pca = decomposition.RandomizedPCA(n_components = 2)
pca = decomposition.PCA(n_components = 2)

reduced_data_pca = pca.fit_transform(digits.data)
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Trying to visualized this data based on 2 dimensions
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

# plot each colors separately
for i in range(len(colors)):
    select = digits.target == i
    x = reduced_data_rpca[:, 0][select]
    y = reduced_data_rpca[:, 1][select]
    
    # Simple scattor plot
    # but each data point has a circle
    # we can change the border and fill of the circle
    # Plot 1 set of data at a time
    plt.scatter(x, y, c=colors[i])
    
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


###########################
#
# Basic model
#
###########################
model.fit(input, output)
model.predict(new_input)
accuracy = metrics.accuracy_score(predictions, data_output)

# Uing kfold validation
kf = KFold(num_row, n_folds)
for train, test in kf:
        train_x = (data[prediction_input].iloc[train,:])
        train_y = data[output].iloc[train]
        
        model.fit(train_x, train_y)
        
        test_x = data[prediction_input].iloc[test,:]
        test_y = data[output].iloc[test]
        
        # Each model has a build in score function
        error.append(model.score(test_x,test_y))
        
        actual_vs_predicted_avg = actual_vs_predicted.groupby(["snapshotmonth"]).mean()
        #  or .aggregate('mean')
        
# Using and visualizing feature importance from RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 500, max_features = 100, max_depth = 30, random_state=0)

# Fitting takes quite some time
forest.fit(train_X, train_y)
importance = forest.feature_importances_

numFeatures = 50 
index_sort = np.flip(np.argsort(importance), axis = 0)

plt.figure(figsize = (15, 15))
plt.barh(range(numFeatures), width = importance[index_sort[:numFeatures]], align = 'center', tick_label = train_X.columns[index_sort[:numFeatures]])

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}

import xgboost as xgb
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

# Gradient Boosting Regressor
gbr = ensemble.GradientBoostingRegressor()
clf = GridSearchCV(gbr, cv=3, param_grid=tuned_parameters,
        scoring='median_absolute_error')
preds = clf.fit(X_train, y_train)
best = clf.best_estimator_

# plot error for each round of boosting
# Note: best_estimator_, staged_predict
test_score = np.zeros(n_est, dtype=np.float64)

train_score = best.train_score_
for i, y_pred in enumerate(best.staged_predict(X_test)):
    test_score[i] = best.loss_(y_test, y_pred)

### Grid search
from pyspark import SparkContext, SparkConf
from spark_sklearn import GridSearchCV

conf = SparkConf()
sc = SparkContext(conf=conf)
clf = GridSearchCV(sc, gbr, cv=3, param_grid=tuned_parameters, scoring='median_absolute_error')
