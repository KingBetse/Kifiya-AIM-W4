from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the serialized pipeline
pipeline = joblib.load("model_24-09-2024-11-26-35.pkl")  # Replace with your actual pipeline filename

app = FastAPI()

# Define input data structure
class SalesInput(BaseModel):
    previous_sales: list
    customers: int = 0
    open: int = 1
    weekday: int = 1
    weekend: int = 0
    days_to_holiday: int = 5
    days_after_holiday: int = 0
    beginning_of_month: int = 0
    mid_month: int = 0
    end_of_month: int = 0
    promo: int = 0
    state_holiday: int = 0
    school_holiday: int = 0
    month: int = 1
    store: int = 1
    is_holiday: int = 0
    day_of_week: int = 1

@app.post("/predict/")
def predict_sales(data: SalesInput):
    try:
        # Create a DataFrame with the input data
        input_data = pd.DataFrame(columns=['Store', 'DayOfWeek', 'Customers', 'Open', 'Promo',
                                           'StateHoliday', 'SchoolHoliday', 'Weekday', 'Weekend',
                                           'DaysToHoliday', 'DaysAfterHoliday', 'BeginningOfMonth',
                                           'MidMonth', 'EndOfMonth', 'Month', 'IsHoliday'])

        # Populate the DataFrame with the provided data
        input_data.loc[0] = [
            data.store,
            data.day_of_week,
            data.customers,
            data.open,
            data.promo,
            data.state_holiday,
            data.school_holiday,
            data.weekday,
            data.weekend,
            data.days_to_holiday,
            data.days_after_holiday,
            data.beginning_of_month,
            data.mid_month,
            data.end_of_month,
            data.month,
            data.is_holiday,
        ]

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_data)

        # Return the prediction
        return {"predicted_sales": prediction[0]}  # Adjust based on your model's output shape
    except Exception as e:
        return {"error": str(e)}