import json
import pickle
import numpy as np

# use Postman Desktop App to test http calls from backend server!

# ind_var_columns.json --> independent variable columns
# model_real_estate_price.pickle --> linear regression model

# global variables
__locations = None
__data_columns = None
__model = None

# load independent variable file: .json + serialized machine learning model file: .pickle 
def load_saved_artifacts():
    # store into global variables
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    # load the location names from columns.json
    with open("/Users/soowan/documents/vscode/projects/regression/real_estate_price/server/artifacts/ind_var_columns.json", 'r') as f:
        # since it's a json file we're loading, json.load(file)
        # whatever object that's loaded will be converted to dictionary since that's what the current .json file contains
        # so we can call the 'key' of that dictionary which is 'data_columns' in this case
        __data_columns = json.load(f)['data_columns']
        # since the 'value' of that dictionary is a list and locations start at index 3...
        __locations = __data_columns[3:]

    global __model
    
    # load the serialized machine learning model from banglore_home_prices_model.pickle
    with open("/Users/soowan/Documents/VSCODE/Projects/Regression/real_estate_price/server/artifacts/model_real_estate_price.pickle", 'rb') as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done")


# returns the independent variable (locations) from global variable
def get_location_names():
    return __locations


# predict the price
def get_estimated_price(location,sqft,bhk,bath):
    # try except to last location since if location isn't found it throws an error
    try:
        # make sure the correct location is selected from dictionary of list (stored in global object: __data_columns) from columns.json file
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    # create ZEROS for independent variables
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    # make sure the correct location is selected by setting one-hot encoding to 1
    if loc_index >= 0:
        x[loc_index] = 1

    # machine learning model that we created (linear regression) takes 2 dimensional array to make predictions
    return round(__model.predict([x])[0],2)



if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))