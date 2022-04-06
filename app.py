
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('app')

# Define predict function
@app.post('/predict')
def predict(house_rooms, house_living_area, house_tot_area, house_construction_year, house_energy_class, house_address_715, house_address_716, house_address_717, house_address_740, house_address_741, house_address_742, house_address_743, house_address_746, house_address_748, house_address_749, house_address_750, house_heating_Etagenheizung, house_heating_Ofenheizung, house_heating_Zentralheizung, house_summary_year):
    data = pd.DataFrame([[house_rooms, house_living_area, house_tot_area, house_construction_year, house_energy_class, house_address_715, house_address_716, house_address_717, house_address_740, house_address_741, house_address_742, house_address_743, house_address_746, house_address_748, house_address_749, house_address_750, house_heating_Etagenheizung, house_heating_Ofenheizung, house_heating_Zentralheizung, house_summary_year]])
    data.columns = ['house_rooms', 'house_living_area', 'house_tot_area', 'house_construction_year', 'house_energy_class', 'house_address_715', 'house_address_716', 'house_address_717', 'house_address_740', 'house_address_741', 'house_address_742', 'house_address_743', 'house_address_746', 'house_address_748', 'house_address_749', 'house_address_750', 'house_heating_Etagenheizung', 'house_heating_Ofenheizung', 'house_heating_Zentralheizung', 'house_summary_year']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
