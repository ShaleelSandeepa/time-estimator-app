import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Load dataset
file_path = "all-vehicles-model-formatted-transformed_data.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns if exists
df = df.drop(columns=['Unnamed: 21'], errors='ignore')

# Handle missing values (fill with median for numerical columns)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Make', 'Model', 'BaseModel']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Avoid division by zero in Rate calculation
df['Co2 Tailpipe For Fuel Type1'].replace(0, np.nan, inplace=True)  # Replace 0 with NaN
df['Annual Fuel Cost For Fuel Type1'].replace(0, np.nan, inplace=True)
df['Engine displacement'].replace(0, np.nan, inplace=True)

# Fill missing values with median (only for numerical columns)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Define Rate formula with scaling for final output
alpha = 0.65  # Highest priority to Make Value
beta = 0.15
gamma = 0.1
delta = 0.05
epsilon = 0.2  # Higher priority for Cylinders

df['Rate'] = (
    delta * (df['Combined Mpg For Fuel Type1'] / (df['Co2 Tailpipe For Fuel Type1'] + 1e-5)) +
    beta * (1 - df['Engine displacement'] / (df['Engine displacement'].max() + 1e-5)) +
    gamma * (1 - df['Annual Fuel Cost For Fuel Type1'] / (df['Annual Fuel Cost For Fuel Type1'].max() + 1e-5)) +
    alpha * (df['Make Value'] / (df['Make Value'].max() + 1e-5)) +
    epsilon * (df['Cylinders']) +  # Now Cylinders has a much higher impact
    0.04 * df['Year Value'] / (df['Year Value'].max() + 1e-5) +
    0.03 * df['Transmission Value'] / (df['Transmission Value'].max() + 1e-5) +
    0.03 * df['Drive Type Value'] / (df['Drive Type Value'].max() + 1e-5) +
    0.03 * df['Fuel Type Value'] / (df['Fuel Type Value'].max() + 1e-5) +
    0.02 * df['Vehicle Size Class Value'] / (df['Vehicle Size Class Value'].max() + 1e-5) +
    0.01 * df['Start-Stop Value']
)

# Normalize Rate to 0 - 1000 scale by applying a factor
df['Rate'] = df['Rate'] * 1000 / df['Rate'].max()  # Rescale to 0-1000

# Ensure Rate contains no infinities
df['Rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Rate'], inplace=True)

# //////////////////////////////////////////////////////////////////////

# # Compute the average Rate for each Make
make_average_rates = df.groupby("Make")["Rate"].mean().to_dict()

# # Add the AvgRate column to the dataset
df["AvgRate"] = df["Make"].map(make_average_rates)

# List of other columns (excluding Make, Model, BaseModel, Rate, and AvgRate)
other_columns = [col for col in df.columns if col not in ["Make", "Model", "BaseModel", "Rate", "AvgRate"]]

# Create additional records for each Make
additional_records = []

for make, avg_rate in make_average_rates.items():
    new_record = {col: 0 for col in other_columns}  # Set all other columns to 0
    new_record["Make"] = make
    new_record["Model"] = make
    new_record["BaseModel"] = make
    new_record["Rate"] = avg_rate
    new_record["AvgRate"] = avg_rate
    additional_records.append(new_record)

# Convert list of new records to DataFrame
new_records_df = pd.DataFrame(additional_records)

# Append new records to the original dataset
df = pd.concat([df, new_records_df], ignore_index=True)

# //////////////////////////////////////////////////////////////////////

# Define features and target
X = df[['Make', 'Model']]
y = df[['Rate', 'Cylinders', 'Transmission Value', 'Engine displacement', 'Vehicle Size Class Value', 'AvgRate']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.4f}')

r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, 'vehicle_rate_model.sav')  # Save the model as a .sav file
# Save the trained model, label encoders, and scaler
joblib.dump(model, 'vehicle_rate_model.pkl')  # Save the model as a .pkl file
joblib.dump(label_encoders, 'label_encoders.pkl')  # Save the label encoders
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

def predict_rate(make: str, model_name: str):
    try:
        # Try to encode and predict using Make and Model
        make_encoded = label_encoders['Make'].transform([make])[0]
        model_encoded = label_encoders['Model'].transform([model_name])[0]
        # avg_rate = df[df['Make'] == make]['AvgRate'].mean()
        input_data = scaler.transform([[make_encoded, model_encoded]])
        
        predicted_values = model.predict(input_data)[0]
        
        # Check if prediction is valid
        if np.any(np.isnan(predicted_values)):
            raise ValueError("Prediction resulted in NaN values, trying with BaseModel.")
        
        return predicted_values
    except Exception as e:
        print(f"Error with Make and Model ({model_name}): {e}. Trying with {model_name.split()[0]} as Model.")
        
        try:
            # Fall back to just the first word in Model as Model
            base_model = model_name.split()[0]  # Use the first word of Model as fallback Model
            base_model_encoded = label_encoders['Model'].transform([base_model])[0]
            input_data = scaler.transform([[make_encoded, base_model_encoded]])
            
            predicted_values = model.predict(input_data)[0]
            
            # Check if prediction is valid
            if np.any(np.isnan(predicted_values)):
                raise ValueError(f"Prediction with {base_model} resulted in NaN values.")
            
            return predicted_values
        except Exception as e:
            print(f"Error with {base_model} as Model: {e}. Trying BaseModel now.")
            try:
                # Fall back to BaseModel encoding
                base_model_encoded = label_encoders['BaseModel'].transform([base_model])[0]
                input_data = scaler.transform([[make_encoded, base_model_encoded]])

                predicted_values = model.predict(input_data)[0]

                # Check if prediction is valid
                if np.any(np.isnan(predicted_values)):
                    raise ValueError(f"Prediction with BaseModel ({base_model}) resulted in NaN values.")
                
                return predicted_values
            except Exception as e:
                print(f"Error with BaseModel ({base_model}): {e}")
                
                # # Final fallback: return the average Rate for the Make
                # avg_rate = make_average_rates.get(make, None)
                # if avg_rate is not None:
                #     print(f"Returning average Rate for {make}: {avg_rate}")
                #     return avg_rate
                # else:
                #     print(f"No average Rate found for {make}. Returning None.")
                return None  # Return None if all attempts fail

# Example prediction
example_make = 'Suzuki'
example_model = 'Wagon R'
example_base_model = 'Prius'
# predicted_rate = predict_rate(example_make, example_model, example_base_model)
predicted_rate = predict_rate(example_make, example_model)
print(f'Predicted Rate for {example_make} {example_model}: {predicted_rate}')
