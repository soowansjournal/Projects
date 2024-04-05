import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

# 1) Problem: Regression
# 2) Data: Explore + Clean + Xy Split + TestTrain Split + Standardize
# 3) Model: Cross-Validate (Fit + Predict + Parameter) + Test 

# to check if range of values or single average
def is_single(x):
    try:
        float(x)
    except:
        return False
    return True

# to convert range of values to single average
def range_avg(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, df_sub in df.groupby('location'):
        # returns Key = Location and df_sub = Location Dataframe
        m = np.mean(df_sub.price_per_sqft)
        st = np.std(df_sub.price_per_sqft)
        reduced_df = df_sub[(df_sub.price_per_sqft > (m-st))&(df_sub.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index = False)
    return df_out 

def scatter_plot(df, location):
    bhk_2 = df[(df.location == location) & (df.bhk == 2)]
    bhk_3 = df[(df.location == location) & (df.bhk == 3)]
    plot = matplotlib.rcParams['figure.figsize'] = (12,8)
    plt.scatter(bhk_2.total_sqft, bhk_2.price_per_sqft, color = 'blue', label = '2 BHK', s=50)
    plt.scatter(bhk_3.total_sqft, bhk_3.price_per_sqft, marker = '*', color = 'red', label = '3 BHK', s=50)
    plt.xlabel('Total Square Feet Area')
    plt.ylabel('Price/Square Feet Area')
    plt.title(location)
    plt.legend()
    plt.show()

def bhk_outlier(df):
    # outlier removal: properties at same location, for example price of 2 bhk < price of 1 bhk (with the same sqft area)
    # we will create a dictionary for each location, where key = # bhk and value = {mean, stdev, count}
    # and then remove the 2 bhk apartments where the price_per_sqft is less than mean price_per_sqft of 1 bhk

    exclude_indices = np.array([])
    # for each location
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        # for each bhk in that location
        for bhk, bhk_df in location_df.groupby('bhk'):
            # calculate the price statistics for specific bhk
            mean_pps = np.mean(bhk_df.price_per_sqft)
            stdev_pps = np.std(bhk_df.price_per_sqft)
            # dictionary values can be anything, in this case the values are also dictionary
            # {'1': {'mean': $$$, 'stdev': $$$, 'count': $$$}, '2': {'mean': $$$, 'stdev': $$$, 'count': $$$}}
            bhk_stats[bhk] = {'mean': mean_pps, 'stdev': stdev_pps, 'count': bhk_df.shape[0]}
        # for each bhk in that location
        for bhk, bhk_df in location_df.groupby('bhk'):
            # from the bhk_stats dictionary, get the stats from one less bhk key
            # .get(1) would return {'mean': $$$, 'stdev': $$$, 'count': $$$}
            stats = bhk_stats.get(bhk-1)
            # if stats exist for that bhk and the number of bhk properties is greater than 5
            if stats and stats['count'] > 5:
                # merge two arrays (original placeholder array, array of indices where price_per_sqft of bhk < mean of bhk-1)
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    
    # drop the outlier property values from original dataframe
    return df.drop(index = exclude_indices, axis = 'index')


# 1) Explore the Data

df1 = pd.read_csv("/Users/soowan/documents/vscode/projects/regression/bengaluru_house_data.csv")
print(df1.head(3),'\n')

# Dataset Size?
print("\n\n\n EXPLORE Data Size")
print(f'{df1.shape[0]} Rows, {df1.shape[1]} Columns\n')

# What Column?
print("\n\n\n EXPLORE Area Type")
print(f'Area Type: {df1.iloc[:,0].unique()}')
print(df1.groupby(by='area_type')['area_type'].agg('count'))


# 2) Clean the Data

# Drop Features (Columns)? 
df2 = df1.drop(['area_type','availability','society', 'balcony'], axis = 1)

print("\n\n\n EXPLORE Dataset with Less Features")
print(df2.head(3),'\n')

# Drop Nulls (Rows)?
print("\n\n\n EXPLORE Null Values in Each Column")
print(f"Null Values (Out of {df2.shape[0]}):\n{df2.isnull().sum()} '\n'")
print(f"Null Values (Percentage %):\n{df2.isnull().sum()/df2.shape[0]*100} '\n'")
df3 = df2.dropna()
print(df3.head(3),'\n')

df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

print("\n\n\n EXPLORE New Column for Number of Beds 'bhk'")
print(df3.head(3),'\n')

print("\n\n\n EXPLORE Number of Beds 'bhk'")
print(df3.bhk.unique())

print("\n\n\n EXPLORE Bedroom Size > 20")
print(df3.where(df3.bhk > 20).dropna())

print("\n\n\n EXPLORE Total Square Feet")
print(df3.total_sqft.unique())
df3['tsf'] = df3['total_sqft'].apply(lambda x: x.split('-'))

print("\n\n\n EXPLORE Total Square Feet with Range")
print(df3[~df3.total_sqft.apply(is_single)].head(10))

print("\n\n\n EXPLORE Total Square Feet Fixed Range to Average")
print(df3.loc[30].total_sqft)
print(range_avg(df3.loc[30].total_sqft))

df4 = df3.copy()

print("\n\n\n EXPLORE Total Square Feet Fixed")
print(df4[~df4.total_sqft.apply(is_single)].head(3))
df4['total_sqft'] = df4['total_sqft'].apply(range_avg)
print(df4.loc[[30,122,137]])

# 2.1) Feature Engineering

df5 = df4.copy()

print("\n\n\n EXPLORE price per sqft")
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
print(df5.head())

print("\n\n\n EXPLORE how many locations")
print(df5.location.unique())
print(f"number of locations: {len(df5.location.unique())}")

# too many locations for one hot encoding, group by location
df5['location'] = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)

print("\n\n\n EXPLORE how many location groups have less than 10 locs")
location_stats_less_than_10 = location_stats[location_stats <= 10]
print(len(location_stats_less_than_10))
print(len(df5.location.unique()))

print("\n\n\n EXPLORE new number of location groups")
# change the location name to 'other' if it is less than 10 in the dataset
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
print(len(df5.location.unique()))

# 2.2) Outlier Removal 

# standard deviation vs simple domain knowledge
print("\n\n\n", df5.head(5))

# 300sqft per bedroom is a common threshold 
print("\n\n\n EXPLORE how many bedrooms don't actually exist")
print("out of", df5.shape[0], len(df5[df5.total_sqft/df5.bhk < 300]), "bedrooms")

print("\n\n\n EXPLORE existing bedrooms")
df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(len(df6.iloc[:,0]))

# check which price per square feet is too high
print("\n\n\n EXPLORE price per sqft")
print(df6.price_per_sqft.describe())

print("\n\n\n EXPLORE size of dataframe after removing outliers in price per sqft")
df7 = remove_pps_outliers(df6)
print(df7.shape)

# check if price is associated with size of bedroom
# print(df7.location.unique())
# scatter_plot(df7, 'Hebbal')

print("\n\n\n EXPLORE size of dataframe after removing outliers bhk")
df8 = bhk_outlier(df7)
print(df8.shape[0])
# scatter_plot(df8, 'Hebbal')

# See how many properties are within the average price per square feet
# matplotlib.rcParams["figure.figsize"] = (12,8)
# plt.hist(df8.price_per_sqft, rwidth = 0.8)
# plt.xlabel('Price Per Square Feet (Rupees)')
# plt.ylabel('# of Properties')
# plt.show()
# # from the histogram, most properties are within 0 to 10,000 rupee range and follows a normal distribution

# now let's look at the number of bathrooms
print("\n\n\n EXPLORE # of bathrooms")
print(df8.bath.unique())
print(df8[df8.bath>10])
# matplotlib.rcParams["figure.figsize"] = (12,8)
# plt.hist(df8.bath, rwidth= 0.8)
# plt.xlabel('# of Bathrooms')
# plt.ylabel('# of properties')
# plt.show()
# # from the histogram, it seems like most of the properties have 2 bathrooms

# concluded that any property with # bathrooms > # bedrooms + 2 is an outlier
print("\n\n\n EXPLORE new dataframe after removing bathroom outlier")
df9 = df8[~(df8.bath>df8.bhk +2)]
print(df9.shape[0])

# we can remove size column since identical bhk column exists
# we can remove price_per_sqft column since price column exists
print("\n\n\n EXPLORE Final Dataframe")
df10 = df9.drop(columns = ['size','price_per_sqft','tsf'], axis='columns')
print(df10.head(3))

# 3) Solve the Data

# 3.1) One-Hot Encoding (Categorical --> Numerical)

# one-hot encoding for the categorical data 'location'
print("\n\n\n EXPLORE One-Hot Encoding")
dummies = pd.get_dummies(df10.location, dtype=int)
print(dummies.head(3))

# combine both dataframes
# dummy variable trap: 
# occurs when one or more dummy variables are redundant, 
# meaning they can be predicted from the other variables.
# avoid dummy variable trap, have one less dummy column
df11 = pd.concat([df10, dummies.drop('other', axis = 'columns')], axis = 'columns')
# we can drop 'locations' column since it is now one-hot encoded
df12 = df11.drop('location', axis = 'columns')

# 3.2) Split: Xy, TestTrain

X = df12.drop('price', axis='columns')
y = df12.price

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# 3.3) Model: Classifier, Train, Test
# 3.3) Model: Test Models & Tune Hyperparameter using GridSearchCV
# 3.3) Model: Create Predictor

# Classifier, Train, Test
print("\n\n\n EXPLORE Model Prediction Score")
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
# use values not the entire dataframe to train and test
print(lr_clf.fit(X_train.values,y_train.values))
print(lr_clf.score(X_test.values,y_test.values))

# Cross Validation
# The dataset is divided into k subsets or folds. 
# The model is trained and evaluated k times, using a different fold as the validation set each time. 
# Performance metrics from each fold are averaged to estimate the model's generalization performance.
print("\n\n\n EXPLORE Model Cross_Validation Score")
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
print(cross_val_score(LinearRegression(),X,y,cv=cv))
# Thus, the average score across all 5 folds is 80% which is pretty good!

# Test Models: Lasso Regression, Decision Tree Regression etc.
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
# Tune Hyperparameter: For each model using GridSearchCV
def model_param_gridsearchcv(X,y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'copy_X': [True, False],
                # 'fit_intercept': [True, False],
                # 'n_jobs': [1,2,3],
                # 'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error','friedman_mse'],
                'splitter': ['best','random']
                }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'],config['params'], cv=cv, return_train_score = False)
        # train the model
        gs.fit(X,y)
        # test the model
        scores.append({
            'model': algo_name,
            # tell me the best score, parameter for each run
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    
    return pd.DataFrame(scores, columns = ['model','best_score', 'best_params'])


# x = model_param_gridsearchcv(X,y)
# print(x)
# # Thus, from the GridSearchCV
# # the best model is LinearRegression 
# # the best hyperparameter is copy_X: True

# create predictions using best model 
def predict_price(location,sqft,bath,bhk):
    # make sure the correct location is selected
    loc_index = X.columns.tolist().index(location)

    # create ZEROS for independent variables
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    # make sure the correct location is selected by setting one-hot encoding to 1
    if loc_index >= 0:
        x[loc_index] = 1
    
    # given a list of independent variables [sqft, bath, bhk, location]
    # predict dependent variable, housing price
    # return the one-hot encoding dataframe to see how it works
    return x, round(lr_clf.predict([x])[0],5)

one_hot_df, price_1 = predict_price('1st Phase JP Nagar', 1000,2,2)
# notice how the one-hot encoding for location works
print("\n\n\n EXPLORE how one-hot encoding works")
print(one_hot_df)
print("\n\n\n EXPLORE Predictions!")
print(price_1, 'Rupees')

one_hot_df, price_2 = predict_price('Indira Nagar', 1000,2,2)
print(price_2, 'Rupees')


# to create backend - python flask server!!!

# convert python file into a byte stream using pickle
import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
            






