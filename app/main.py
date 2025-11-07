from fastapi import FastAPI, HTTPException 
from app.schema.input_data import LoanInput
from app.service.predict import predict_loan_default_from_file

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Loan default prediction API is successfully running!"}

@app.post("/predict")
def predict():
    try:
        result = predict_loan_default_from_file()
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
