import pandas as pd
import math
from prophet import Prophet
import data_gen_logger
import logging

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
            
def predict_diabetic_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Generates forecast for the given `data`.

    Args:
        `data` (pd.DataFrame): historical data

    Returns:
        `pd.DataFrame`: dataset with the forecasts
    """    
    try:
        logger.info('Generating forecasts for the provided dataset...')
        data['Date'] = pd.to_datetime(data['Date'])
        regressor_columns = [col for col in data.columns if col not in ['Date', 'Health_Status', 'HbA1c']]
        data = data[['Date', 'HbA1c'] + regressor_columns].rename(columns={'Date': 'ds', 'HbA1c': 'y'})

        future_df = pd.DataFrame()  # DataFrame to store future predicted values

        for column in data.columns[1:]:  # Loop through all columns except 'ds'
            # Prepare data for Prophet
            df = data[['ds', column]].rename(columns={'ds': 'ds', column: 'y'})

            # Create and fit Prophet model
            model = Prophet()
            model.fit(df)

            # Make future predictions
            future = model.make_future_dataframe(periods=200,freq='M')  # Change periods as needed
            forecast = model.predict(future)

            # Extract future predicted values and append to future_df
            future_values = forecast[['ds', 'yhat']].rename(columns={'ds': 'ds', 'yhat': column})
            future_df = pd.concat([future_df, future_values[column]], axis=1)

            # Append the dates from the forecast
            future_df['ds'] = forecast['ds']

            # Print or use future_df as needed
            # print(future_df)
        future_df.drop('y',axis=1,inplace=True)
        model = Prophet()
        model.add_regressor('Age')
        model.add_regressor('Urea')
        model.add_regressor('Cr')
        model.add_regressor('HDL')
        model.add_regressor('BGL')
        model.add_regressor('Chol')
        model.add_regressor('TG')
        model.add_regressor('LDL')
        model.add_regressor('VLDL')
        model.add_regressor('BMI')


        model.fit(data)
        print(future_df)
        forecast = model.predict(future_df)
                    
        future_df['y'] = forecast['yhat']
        future_df['Risk_Category'] = future_df['y'].apply(categorize_risk)
        future_df['Age'] = future_df['Age'].apply(lambda x: math.floor(x))
        new_columns_order = ['ds'] + [col for col in future_df.columns if col != 'ds']
        future_df = future_df[new_columns_order]
        
        type_cast = ['Age', 'Urea', 'Cr', 'BGL', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'y']
        future_df[type_cast] = future_df[type_cast].round(2)
        future_df["ds"] = future_df["ds"].dt.date
        future_df.rename(columns={'ds':'Date', 'y':'HbA1c'}, inplace=True)
        logger.info('Successfully generated forecasts')
        return future_df
    except Exception as e:
         logger.exception(e)
         raise e



