import joblib 
import pandas as pd
import logging 

# Load the saved model and the encoders 
model_path = r'C:\Users\ap422\OneDrive\Desktop\loan_default_deployment\app\model\loan_model.joblib'
encoder_path = r'C:\Users\ap422\OneDrive\Desktop\loan_default_deployment\app\model\loan_encoders.joblib'

model = joblib.load(model_path)
label_encoders = joblib.load(encoder_path)

# Set up logging 
logging.basicConfig(level=logging.INFO)

def predict_loan_default_from_file() -> list:
    # load the csv file 
    df = pd.read_csv(r'C:\Users\ap422\OneDrive\Desktop\loan_default_deployment\app\data\loan_data.csv')
    target_column = 'loan_status'
    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except ValueError as e:
                raise ValueError(f"Invalid value for column {col}")
        else: 
            raise ValueError(f"Missing Column: {col}")
        
    # predictions 
    predictions = model.predict(df)
    # Converting the predictions into class labels 
    return ["Approved" if pred == 0 else "Rejected" for pred in predictions]

print("Eveeything ran successfully!")
