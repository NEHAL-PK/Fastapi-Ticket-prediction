
from fastapi import FastAPI
from pydantic import BaseModel
import pickle as pkl


import uvicorn
# Define your Pydantic model for the combined prediction
class CombinedPredictionRequest(BaseModel):
    Duration: int
    AirlineNumber: int
    Total_stops: int
    Source_PAR: int
    Average_Price: float
    Source_RUH: int
    Source_SVO: int
    Destination_PAR: int
    Destination_RUH: int
    Destination_SVO: int
  

app = FastAPI()

# Load your models
with open("decision_tree_model.pkl", "rb") as pickle_in:
    classifier = pkl.load(pickle_in)

with open("RandomForestRegressor_model.pkl", "rb") as pickle_in2:
    regression = pkl.load(pickle_in2)

@app.get('/')
def index():
    return {'Message': 'Welcome \n'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f"Hello, {name}"}

@app.post('/predict_combined')
async def predict_combined(data: CombinedPredictionRequest):
    # Convert input data to dictionary
    data = data.dict()
    
    # Flight prediction uses Duration and AirlineNumber
    flight_prediction = classifier.predict([[data['Duration'], data['AirlineNumber']]])
    
    # Price prediction uses Duration again along with other fields
    price_prediction = regression.predict([[
        data['Duration'],  # Same Duration used here
        data['Total_stops'],
        data['Source_PAR'],
        data['Average_Price'],
        data['Source_RUH'],
        data['Source_SVO'],
        data['Destination_PAR'],
        data['Destination_RUH'],
        data['Destination_SVO']
        
    ]])

    # Return both predictions
    return {
        "flight_prediction": flight_prediction.tolist(),
        "price_prediction": price_prediction.tolist()
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
