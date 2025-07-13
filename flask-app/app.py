# pip freeze > requirements.txt

# Before executing the app Verify This:

# Run below command for create environment
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

from flask import Flask, render_template, request,jsonify
import joblib
import numpy as np
from flask_cors import CORS
import random
import math

app = Flask(__name__, template_folder='web')
CORS(app)

# Load model and preprocessing tools for vehicle rating prediction
model = joblib.load("car_rating_model.pkl")
encoder = joblib.load("encoder.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")
vehicleRateModel = joblib.load("vehicle_rate_model.sav")

@app.route('/')
def student():
    return render_template("home.html")

# This function predict the estimated time with the Regression Model
def ValuePredictorDifferentialIssue(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorOilLeak(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('oil-leak-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorChangeClutch(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('change-clutch-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorGasketIssue(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('gasket-issue-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorSuspensionChange(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('suspension-change-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorLightVehicle(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('light-alignment-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]

def ValuePredictorHeavyVehicle(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('heavy-alignment-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*0.9

def ValuePredictorService1000(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('service-1000-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*5.5

def ValuePredictorService5000(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('service-5000-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*7

def ValuePredictorService10000(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('service-10000-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*8.8

def ValuePredictorService40000(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('service-40000-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*12.3

def ValuePredictorTuneUp(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('tune-up-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*2.5

def ValuePredictorEngineRepair(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('engine-repair-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*4

def ValuePredictorInteriorWash(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('interior-wash-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*0.9

def ValuePredictorBodyWash(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1, 1)
    loaded_model = joblib.load('body-wash-model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]*0.6


# //////////////////////////////////////////////////////////////////////////////////

def VehicleRatePredict(make: str, model_name: str):
    try:
        # Try to encode and predict using Make and Model
        make_encoded = label_encoders["Make"].transform([make])[0]
        model_encoded = label_encoders["Model"].transform([model_name])[0]
        input_data = scaler.transform([[make_encoded, model_encoded]])

        # Get the prediction from the model
        predicted_values = vehicleRateModel.predict(input_data)[0]

        # Extract all relevant values
        predicted_rate_value = float(predicted_values[0])  # Vehicle Repair Rate
        cylinders = int(predicted_values[1])  # Cylinders
        transmission_value = float(predicted_values[2])  # Transmission Value
        engine_displacement = float(predicted_values[3])  # Engine Displacement
        vehicle_size_class_value = float(predicted_values[4])  # Vehicle Size Class Value
        average_rate = float(predicted_values[5])  # Average Rate for given Make

        # Format values as required
        formatted_rate = f"{predicted_rate_value:.2f}"  # Format to xxx.xx
        formatted_transmission_value = f"{transmission_value:.2f}"  # Format to xxx.xx
        formatted_engine_displacement = f"{engine_displacement*1000:.0f} cc"  # Format to xxxx cc
        formatted_vehicle_size_class_value = f"{vehicle_size_class_value:.2f}"  # Format to xxx.xx
        formatted_avg_rate = f"{average_rate:.2f}"  # Format to xxx.xx

        # Return all the predicted values as a dictionary with formatted values
        return {
            "rate": formatted_rate,
            # "cylinders": cylinders,  # Cylinders as integer (x)
            # "transmission_value": formatted_transmission_value,
            # "engine_displacement": formatted_engine_displacement,
            # "vehicle_size_class_value": formatted_vehicle_size_class_value,
            "average_rate": formatted_avg_rate
        }

    except Exception as e:
        print(f"Error with Make and Model ({model_name}): {e}. Trying with {model_name.split()[0]} as Model.")

        try:
            # Fall back to just the first word in Model as Model
            base_model = model_name.split()[0]  # Use the first word of Model as fallback Model
            base_model_encoded = label_encoders["Model"].transform([base_model])[0]
            input_data = scaler.transform([[make_encoded, base_model_encoded]])

            predicted_values = vehicleRateModel.predict(input_data)[0]

            # Check if prediction is valid
            if np.any(np.isnan(predicted_values)):
                raise ValueError(f"Prediction with {base_model} resulted in NaN values.")

            # Extract all relevant values
            predicted_rate_value = float(predicted_values[0])  # Vehicle Repair Rate
            cylinders = int(predicted_values[1])  # Cylinders
            transmission_value = float(predicted_values[2])  # Transmission Value
            engine_displacement = float(predicted_values[3])  # Engine Displacement
            vehicle_size_class_value = float(predicted_values[4])  # Vehicle Size Class Value
            average_rate = float(predicted_values[5])  # Average Rate for given Make

            # Format values as required
            formatted_rate = f"{predicted_rate_value:.2f}"  # Format to xxx.xx
            formatted_transmission_value = f"{transmission_value:.2f}"  # Format to xxx.xx
            formatted_engine_displacement = f"{engine_displacement*1000:.0f} cc"  # Format to xxxx cc
            formatted_vehicle_size_class_value = f"{vehicle_size_class_value:.2f}"  # Format to xxx.xx
            formatted_avg_rate = f"{average_rate:.2f}"  # Format to xxx.xx

            # Return all the predicted values as a dictionary with formatted values
            return {
                "rate": formatted_rate,
                # "cylinders": cylinders,  # Cylinders as integer (x)
                # "transmission_value": formatted_transmission_value,
                # "engine_displacement": formatted_engine_displacement,
                # "vehicle_size_class_value": formatted_vehicle_size_class_value,
                "average_rate": formatted_avg_rate,
                # "warning": f"Considering First Word as Model '{base_model}'..."
            }

        except Exception as e:
            print(f"Error with {base_model} as Model: {e}. Trying BaseModel now.")
            try:
                # Fall back to BaseModel encoding
                base_model_encoded = label_encoders["BaseModel"].transform([base_model])[0]
                input_data = scaler.transform([[make_encoded, base_model_encoded]])

                predicted_values = vehicleRateModel.predict(input_data)[0]

                # Check if prediction is valid
                if np.any(np.isnan(predicted_values)):
                    raise ValueError(f"Prediction with BaseModel ({base_model}) resulted in NaN values.")

                # Extract all relevant values
                predicted_rate_value = float(predicted_values[0])  # Vehicle Repair Rate
                cylinders = int(predicted_values[1])  # Cylinders
                transmission_value = float(predicted_values[2])  # Transmission Value
                engine_displacement = float(predicted_values[3])  # Engine Displacement
                vehicle_size_class_value = float(predicted_values[4])  # Vehicle Size Class Value
                average_rate = float(predicted_values[5])  # Average Rate for given Make

                # Format values as required
                formatted_rate = f"{predicted_rate_value:.2f}"  # Format to xxx.xx
                formatted_transmission_value = f"{transmission_value:.2f}"  # Format to xxx.xx
                formatted_engine_displacement = f"{engine_displacement*1000:.0f} cc"  # Format to xxxx cc
                formatted_vehicle_size_class_value = f"{vehicle_size_class_value:.2f}"  # Format to xxx.xx
                formatted_avg_rate = f"{average_rate:.2f}"  # Format to xxx.xx

                # Return all the predicted values as a dictionary with formatted values
                return {
                    "rate": formatted_rate,
                    # "cylinders": cylinders,  # Cylinders as integer (x)
                    # "transmission_value": formatted_transmission_value,
                    # "engine_displacement": formatted_engine_displacement,
                    # "vehicle_size_class_value": formatted_vehicle_size_class_value,
                    "average_rate": formatted_avg_rate,
                    # "warning": f"Considering the Vehicle Base Model '{base_model}'..."
                }

            except Exception as e:
                print(f"Error with BaseModel ({base_model}): {e}")

                try:
                    input_data = scaler.transform([[make_encoded, make_encoded]])
                    predicted_values = vehicleRateModel.predict(input_data)[0]

                    # Check if prediction is valid
                    if np.any(np.isnan(predicted_values)):
                        raise ValueError(f"Prediction with BaseModel ({base_model}) resulted in NaN values.")

                    # Extract all relevant values
                    predicted_rate_value = float(predicted_values[0])  # Vehicle Repair Rate
                    cylinders = int(predicted_values[1])  # Cylinders
                    transmission_value = float(predicted_values[2])  # Transmission Value
                    engine_displacement = float(predicted_values[3])  # Engine Displacement
                    vehicle_size_class_value = float(predicted_values[4])  # Vehicle Size Class Value
                    average_rate = float(predicted_values[5])  # Average Rate for given Make

                    # Format values as required
                    formatted_rate = f"{predicted_rate_value:.2f}"  # Format to xxx.xx
                    formatted_transmission_value = f"{transmission_value:.2f}"  # Format to xxx.xx
                    formatted_engine_displacement = f"{engine_displacement*1000:.0f} cc"  # Format to xxxx cc
                    formatted_vehicle_size_class_value = f"{vehicle_size_class_value:.2f}"  # Format to xxx.xx
                    formatted_avg_rate = f"{average_rate:.2f}"  # Format to xxx.xx

                    # Return all the predicted values as a dictionary with formatted values
                    return {
                        "rate": formatted_rate,
                        # "cylinders": cylinders,  # Cylinders as integer (x)
                        # "transmission_value": formatted_transmission_value,
                        # "engine_displacement": formatted_engine_displacement,
                        # "vehicle_size_class_value": formatted_vehicle_size_class_value,
                        "average_rate": formatted_avg_rate,
                        # "warning": f"Undefined Vehicle Model '{model_name}'..."
                    }

                except Exception as e: 
                    return {
                        # "error": str(e), 
                        "warning": f"Undefined Vehicle Brand '{make}'...",
                        "rate": round(random.uniform(400.00, 600.00), 2)
                        }  # Return error if all attempts fail



@app.route('/predict_vehicle_rate', methods=['POST'])
def predict_vehicle_rate():
    data = request.json
    make = data.get('make')
    model_name = data.get('model_name')

    if not make or not model_name:
        return jsonify({"error": "Make and model_name are required", "default": 1}), 400

    try:
        # Call VehicleRatePredict method to get prediction
        rate_prediction = VehicleRatePredict(make, model_name)
        print(f"Received request for {make} {model_name}, predicted values: {rate_prediction}")
        return jsonify(rate_prediction)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


    
# //////////////////////////////////////////////////////////////////////////////////


def predict():
    data = request.json
    categorical_data = np.array([[data["Transmission"], data["Fuel Type"], data["Drive"], data["Vehicle Size Class"]]])
    numerical_data = np.array([[data["Annual Fuel Cost For Fuel Type1"], data["City Mpg For Fuel Type1"], data["Co2 Fuel Type1"]]])
    
    encoded_cat_data = encoder.transform(categorical_data)
    scaled_num_data = scaler.transform(numerical_data)
    final_input = np.hstack([encoded_cat_data, scaled_num_data])
    
    prediction = model.predict(final_input)[0]
    return jsonify({"Predicted Rating": round(prediction, 2)})

# This function listen to the front-end and output the value
@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(request.form)
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = round(float(ValuePredictorDifferentialIssue(to_predict_list)), 2)
        return render_template("home.html", result=result)

# Json service which enables to send the order value via json and output 
# estimated time in json format
# required input content-type: application/json value: {'order': 30}
@app.route('/app/',methods=['POST', 'GET'])
def jsonService():
    if request.method == 'POST':
        total = 0
        rate_value = 0.00
        data = request.json
        print(request.get_json())
        print(data['order'])
        order_value = data['order']
        sub_cat_list = data['serviceEntries']
        vehicleMake = data['make']
        vehicleModel = data['model']
        for cateagory in sub_cat_list:
            match cateagory:
                case "Oil Leak": 
                    total = total + round(float(ValuePredictorOilLeak(order_value)), 2)
                case "Differential Issue": 
                    total = total + round(float(ValuePredictorDifferentialIssue(order_value)), 2)
                case "Change Clutch": 
                    total = total + round(float(ValuePredictorChangeClutch(order_value)), 2)
                case "Gasket Issue": 
                    total = total + round(float(ValuePredictorGasketIssue(order_value)), 2)
                case "Light Vehicle": 
                    total = total + round(float(ValuePredictorLightVehicle(order_value)), 2)
                case "Heavy Vehicle": 
                     total = total + round(float(ValuePredictorHeavyVehicle(order_value)), 2)
                case "1000KM": 
                    total = total + round(float(ValuePredictorService1000(order_value)), 2)
                case "5000KM": 
                    total = total + round(float(ValuePredictorService5000(order_value)), 2)
                case "10000KM": 
                    total = total + round(float(ValuePredictorService10000(order_value)), 2)
                case "40000KM": 
                    total = total + round(float(ValuePredictorService40000(order_value)), 2)
                case "Engine Repair": 
                     total = total + round(float(ValuePredictorEngineRepair(order_value)), 2)
                case "Suspension Change": 
                    total = total + round(float(ValuePredictorSuspensionChange(order_value)), 2)
                case "Tune Up": 
                    total = total + round(float(ValuePredictorTuneUp(order_value)), 2)
                case "Body wash": 
                    total = total + round(float(ValuePredictorBodyWash(order_value)), 2)
                case "Interior Wash": 
                    total = total + round(float(ValuePredictorInteriorWash(order_value)), 2)

        
        # result = round(float(ValuePredictorOilLeak(order_value)), 2)
        rate_prediction = VehicleRatePredict(vehicleMake, vehicleModel)
        rate_value = float(rate_prediction.get("rate", 0))  # Ensure 'rate' is a float

        # if total == 0 or math.isnan(total):  # Check if total is 0 or NaN
        #     rate_value = total

        if total != 0 :
            response = jsonify({"time_estimated":total+(rate_value/100*2), "rate":rate_value})
        else:
            response = jsonify({"time_estimated":total, "rate":rate_value})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
        return response

if __name__ == '__main__':
    app.run(debug=True)