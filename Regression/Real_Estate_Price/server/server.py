from flask import Flask, request, jsonify
import util

# use postman to test http calls from backend server!

# ind_var_columns.json --> independent variable columns
# model_real_estate_price.pickle --> linear regression model

# create an app
app = Flask(__name__)

# # expose http endpoint: add '/hello' to run hello() at http web server
# @app.route('/hello')
# def hello():
#     return "hi"

#  expose http endpoint: add '/get_location' to run get_location() at http web server
@app.route('/get_location_names') # default method 'GET' request - fetches data to url
def get_location_names():
    # jsonify serializes data to JSON format and wraps it in a Response format
    # the util file contains all the functions/methods to load locations, model and function/method to predict prices
    # return a response that contains all the house locations 
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

# expose http endpoint: add '/predict_home_price' to run predict_home_price() at http web server
@app.route('/predict_home_price', methods = ['POST']) # POST - send data in HTML form to the server
def predict_home_price():
    # whenever we make a http call from our html application, we get the inputs in request.form
    total_sqft = float(request.form['total_sqft']) # request.form will have an element called 'total_sqft'
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    # jsonify serializes data to JSON format and wraps it in a Response format
    # the util file contains all the functions/methods to load locations, model and function/method to predict prices
    # return a response that takes independent variables and predicts corresponding price
    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server For Real Estate Price Prediction...")
    # run the application on this specific code
    util.load_saved_artifacts()
    app.run()