from darts.models import XGBModel, NaiveEnsembleModel
from darts.metrics import mape, ope
from darts import TimeSeries
import data_gen_logger
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger('data_gen')

def categorize_risk(y_value):
    try:
        risk = {
            (4.5, 5.7): "Normal",
            (5.7, 6.6): "Pre-diabetic",
            (6.6, 7.1): "Diabetic",
            (7.1, 9.0): "Uncontrolled",
            (9.1, 100): "Critically High"
        }

        for value_range, risk_category in risk.items():
            if y_value >= value_range[0] and y_value < value_range[1]:
                return risk_category
    except Exception as e:
        logger.exception(e)
        raise e
            
def predict_diabetic_risk(data: pd.DataFrame, 
                          target: str, n: int) -> pd.DataFrame:
    """Generates forecast for the given `data`.

    Args:
        `data` (pd.DataFrame): historical data
        `target` (str): name of the column to forecast
        `n` (int): number of forecasts to generate

    Raises: 
        `e`: generic errors/exceptions

    Returns:
        `pd.DataFrame`: dataset with the forecasts
    """    
    try:
        logger.info("Received request to generate forecasts for Diabetes.")
        
        # performing some pre-processing and converting data to timeseries
        data[target] = data[target].apply(lambda x: np.sqrt(x))
        data.set_index(keys=["Date"], verify_integrity=True, inplace=True)
        ts = TimeSeries.from_series(data[target], 
                                    freq=pd.DateOffset(months=1))
        assert len(ts) > 0, "Timeseries has no data"

        if not f"Diabetes_forecaster.pkl" in os.listdir("models"):
            log = f"Requested disease, Diabetes_forecaster is not listed."
            logger.warning(log)
            return {"status": "Request rejected", "reason": log}
        
        # loading the Diabetes forecasting model and forecasting
        logger.info('Generating forecasts for the provided dataset...')
        path = os.path.join("models", "Diabetes_forecaster.pkl")
        model = NaiveEnsembleModel.load(path)
        preds = model.predict(n=len(ts), series=ts)

        # post-processing the forecasting data
        columns = ["Date", "Data"]
        df = pd.DataFrame(columns=columns)
        df["Date"], df["Data"] = preds.time_index, np.round(preds.values(), 3)
        df['Date'] = df['Date'].apply(lambda x: x.date())
        df["Data"] = df["Data"].transform(lambda x: x**2)
        df["Data"] = df["Data"].round(2)
        # df["Status"] = df["Data"].apply(lambda x: categorize_risk(x))

        logger.info(f"Successfully completed forecasting for {target} column")
        return df
    except Exception as e:
         logger.exception(e)
         raise e



