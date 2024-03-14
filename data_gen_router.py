from forecast import predict_diabetic_risk
from datetime import datetime, timedelta
from support import generate_data
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

        # fetching primary data
        logger.info('Retrieving information from request and references...')

        age = input_data.get("Age", random.randint(18, 65))
        stage = input_data.get("Health_Status", None)
        random_date = f'2024-{random.randint(1, 12)}-{random.randint(1, 28)}'
        end_date = input_data.get("Date", random_date)
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        logger.info('Patient age is %s and current health status is %s', 
                    age, stage)
        
        # determine the age group for fetching correct confidence intervals
        path = os.path.join('reference',f'{disease}','young.json') \
                if age in range(18, 31) else \
                    os.path.join('reference',f'{disease}','middle.json') \
                        if age in range(31, 46) else \
                            os.path.join('reference',f'{disease}','old.json')
        
        # fetching confidence intervals of all parameters per stage
        with open(path) as f:
            dictt = json.load(f)

        # encoding the disease stage to 1/2/3/4...
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
                  
        # regressing the dates with monthly freqeuncy
        dates = [(end_date - timedelta(days=delta)).date() \
                 for delta in range(30, 30*(adj_size*stage_code+1), 30)]
        dates.reverse()
        data['Date'] = dates

        # regressing the age 
        birth_date = datetime.strptime(f'{end_date.year - age}-12-12', 
                                       '%Y-%m-%d').date()
        data['Age'] = data['Date'].apply(lambda x: (x - birth_date).days//365)

        # regressing health status
        health_status = [dictt.get(f"STAGE-{i+1}", None).get("Name") \
                         for i in range(stage_code) for j in range(adj_size)]
        data['Health_Status'] = health_status

        logger.info('Generating %s samples per class', adj_size)    
        for col in columns:
            limit = input_data.get(col, None)
            assert limit is not None
            data[col] = generate_data(
                            reference=dictt, 
                            parameter=col, 
                            stage=stage_code, 
                            thresh=limit, 
                            size=adj_size
                        )
            data[col] = data[col].round(2)

        df = pd.concat([data, pd.DataFrame(input_data, index=[data.shape[0]])])
        logger.info('Successfully generated data. Adding final touches')
        rearrange = ["Date", "Age", "Health_Status"] + columns
        return df[rearrange].to_dict(orient='index')
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
