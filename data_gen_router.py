from forecast import predict_diabetic_risk
from datetime import datetime, timedelta
from support import data_synthesizer
from datetime import datetime
from fastapi import FastAPI
import pandas as pd
import logging
import random
import json
import os

logger = logging.getLogger('data_gen')
app = FastAPI()

@app.post('/data-generator')
async def data_generator(input_data: dict, samples: int, 
                         disease: str) -> dict:
    """Generates synthetic data \n

    Args: \n
        input_data (dict): reference data as dictionary \n
        samples (int): length of the dataset \n
        disease (str): name of the disease

    Raises: \n
        e: generic errors/exceptions \n

    Returns:\n 
        dict: generated data \n
    """    
    try:
        logger.info('Received request to generate synthetic data')

        logger.info('Retrieving information from request and references...')

        # fetching primary data
        age = input_data.get("Age", random.randint(18, 65))
        stage = input_data.get("Health_Status", None)
        random_date = f'2024-{random.randint(1, 12)}-{random.randint(1, 28)}'
        end_date = input_data.get("Date", random_date)
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        logger.info('Patient age is %s and current health status is %s', 
                    age, stage)
                 
        # fetching dictionary containing confidence intervals
        path = f"reference\\{disease}.json"        
        with open(path) as f:
            disease_ref = json.load(f)

        # encoding the disease stage to 1/2/3/4...
        dictt = next(iter(disease_ref.values()))
        stage_code = [int(k.split('-')[-1]) for k, v in dictt.items()\
                       if v.get('Name') == stage]
        assert len(stage_code) != 0
        stage_code = stage_code[0]

        # adjusting sample size 
        adj_size = samples//stage_code      
        
        # fetch column names and create dataframe
        logger.info("Creating empty dataframe and filling it with data")
        columns = list(list(dictt.values())[0].keys())
        columns.remove('Name')
        data = pd.DataFrame(columns=columns) 
                  
        # regressing the dates with month-wise freqeuncy
        dates = [(end_date - timedelta(days=delta)).date() \
                 for delta in range(0, 30*(adj_size*stage_code+1), 30)]
        dates.reverse()
        data['Date'] = dates

        # regressing the age 
        birth_date = datetime.strptime(f'{end_date.year - age}-12-12', 
                                       '%Y-%m-%d').date()
        data['Age'] = data['Date'].apply(lambda x: (x - birth_date).days//365)

        # mapping the age to age groups
        age_groups = [1 if age in range(18, 31) else 2 \
                      if age in range(31, 46) else 3 \
                        for age in data["Age"].to_list()]
        
        # regressing health status
        health_status = [dictt.get(f"STAGE-{i+1}", None).get("Name") \
                         for i in range(stage_code) for j in range(adj_size)]
        health_status.append(stage)
        data['Health_Status'] = health_status

        logger.info('Generating %s samples per class', adj_size)    
        for col in columns:
            # for each parameter, using input data as threshold
            limit = input_data.get(col, None)
            assert limit is not None, f"There is no data for {col} parameter."
            data[col] = data_synthesizer(
                            reference=disease_ref, 
                            parameter=col, 
                            stage_code=stage_code, 
                            thresh=limit, 
                            adj_size=adj_size,
                            age_groups=age_groups
                        )
            data[col] = data[col].round(3)

        logger.info('Successfully generated data. Adding final touches')
        rearrange = ["Date", "Age", "Health_Status"] + columns
        return data[rearrange].to_dict(orient='index')
    except Exception as e:
        logger.exception(e)
        raise e

@app.post('/forecast')
async def forecast(data_dict: dict) -> dict:
    """Generates forecasts for the given historical data.\n

    Args:\n
        `data_dict` (dict): historical data\n

    Raises:\n
        `e`: generic errors/exceptions\n

    Returns:\n
        `dict`: forecasted data\n
    """    
    try:
        logger.info('Received request to extrapolate for the given data')

        df = pd.DataFrame(list(data_dict.values()))
        response = predict_diabetic_risk(data=df)
        
        logger.info('Successfully generated forecasts.')
        return response.to_dict(orient='index')
    except Exception as e:
        logger.exception(e)
        raise e
