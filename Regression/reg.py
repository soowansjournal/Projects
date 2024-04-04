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

# we can remove bhk column since identical size column exists
# we can remove price_per_sqft column since price column exists
print("\n\n\n EXPLORE Final Dataframe")
df10 = df9.drop(columns = ['bhk','price_per_sqft','tsf'])
print(df10.head(3))


            






