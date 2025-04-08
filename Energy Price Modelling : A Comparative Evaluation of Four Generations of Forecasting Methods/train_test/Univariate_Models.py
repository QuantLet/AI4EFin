from pmdarima import auto_arima
import timesfm
from chronos import ChronosPipeline
import pandas as pd
import torch
import numpy as np


class Univariate_Forecasting():
    """
    A class implementing the univariate forecasting techniques Chronos, TimesFM and ARIMA:

    Parameters:
    ----------
    df : Pandas dataframe
        The dataframe containing the (electricity price-related) time series
    train_size : int
        The number of training time steps.
    test_size : list of int values
        The number of testing time steps.
    seq_len : int
        The size of the input sequence, i.e., the number of input time steps.
    pred_len : int
        The size of the output sequence, i.e., the number of time steps to be predicted.
    n_splits : int
        The number of walk-forward validation splits.
    univariate_forecasters:list
        The univariate forecasting techniques to compute predictions with.
    
    Attributes:
    (the same as the parameters)
    ----------

    Methods
    -------
    compute_predictions(current_univariate_forecaster):
        Generates the predictions using the supported univariate models.
    """

    def __init__(self,
                 df:pd.DataFrame,
                 train_size:int,
                 test_size:int,
                 seq_len:int,
                 pred_len:int,
                 n_splits:int,
                 univariate_forecasters:list):
        
        self.df=df
        self.train_size=train_size
        self.test_size=test_size
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.n_splits=n_splits
        self.univariate_forecasters=univariate_forecasters

        super(Univariate_Forecasting, self).__init__()

    def compute_predictions(self)->dict:
        """Computes predictions from univariate time series.
        
        Parameters:
        ----------
        current_univariate_forecaster: str
        The name of the univariate forecasting technique.
        
        Returns:
        ----------
        dict_results: dict
        The dictionary with the generated predictions.

        Raises:
        ----------
        ValueError: If specified current_univariate_forecaster is not in the list [ARIMA, Chronos, TimesFM].
        """
        all_univariate_forecasters_results={}
        
        for current_univariate_forecaster in self.univariate_forecasters:
            if current_univariate_forecaster not in ['ARIMA','Chronos','TimesFM']:
                raise ValueError(f'The currently specified forecasting technique {current_univariate_forecaster} is not in the list of supported methods.')
            
            #Create the empty dictionary, which will be used in order to store the predtions:
            dict_results={}

            if current_univariate_forecaster in ['Chronos','TimesFM']:
                #If the current_univariate_forecaster is in [TimesFM,Chronos], then first 
                # load the corresponding model:
                if current_univariate_forecaster=='TimesFM':
                    tfm = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        context_len=self.seq_len, 
                        horizon_len=self.pred_len,  
                        input_patch_len=32,
                        output_patch_len=128,
                        num_layers=20,
                        model_dims=1280,
                        backend="cpu"),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))
                    
                    test_data_timesfm=self.df.iloc[self.train_size:].copy()
                    test_data_timesfm['unique_id']=1.0#create a dummy unique id
                    test_data_timesfm['ds']=test_data_timesfm.index

                else:#current_univariate_forecaster=='Chronos':
                    chronos_pipeline = ChronosPipeline.from_pretrained(
                            "amazon/chronos-t5-small",
                            device_map="cpu",
                            torch_dtype=torch.bfloat16,
                        )
                    num_samples=1


                train_data = self.df.iloc[:self.train_size]
                test_data = self.df.iloc[self.train_size:]

                for country_col in train_data.columns[:-2]:
                    current_country=country_col.split(' ')[0]

                    predictions=[]
                    actuals=[]
                    
                    for start_idx in range(0,len(test_data) - self.seq_len - self.pred_len + 1):
                        # Define the end index for the input and prediction window
                        end_idx = start_idx + self.seq_len
                        pred_end_idx = end_idx + self.pred_len
                        actual_sequence = test_data[country_col].values[end_idx:pred_end_idx] #ok as of now


                        if current_univariate_forecaster=='TimesFM':
                            input_sequence = test_data_timesfm[[country_col,'unique_id','ds']].iloc[start_idx:end_idx]
                            forecast_df = tfm.forecast_on_df(
                                inputs=input_sequence,freq="H",  
                                value_name=country_col,num_jobs=-1)
                            
                            predictionsTFM=forecast_df['timesfm'].values.flatten()
                            predictions.append(predictionsTFM)
                        
                        else:
                            input_sequence = test_data[country_col].iloc[start_idx:end_idx]
                            input_tensor = torch.tensor(input_sequence, dtype=torch.float32)

                            chronos_forecast = chronos_pipeline.predict(
                                context=input_tensor,
                                prediction_length=self.pred_len,
                                limit_prediction_length=False,
                                num_samples=num_samples)
                            
                            median_predictions = np.quantile(chronos_forecast[0].numpy(), [0.5], axis=0)
                            median_predictions=median_predictions.flatten()
                            predictions.append(median_predictions)
                                
                        actuals.append(actual_sequence)
                        
                    predictions=np.array(predictions)
                    actuals=np.array(actuals)

                    dict_results[current_country]={'actuals':actuals,
                                                'predictions':predictions}
            else:#current_univariate_forecaster=='ARIMA':

                for col in self.df.columns[:-2]:
                    dict_results[col.split(' ')[0]]={}

                #Loop through the walk-forward validation splits:
                for split in range(0,self.n_splits):
                    
                    # Define train and test ranges for this split
                    train_start = split * self.test_size
                    train_end = train_start + self.train_size
                    test_start = train_end
                    test_end = test_start + self.test_size

                    train_data = self.df.iloc[train_start:train_end]
                    test_data = self.df.iloc[test_start:test_end]

                    #Loop through the countries in each split:
                    for country_col in train_data.columns[:-2]:
                        current_country=country_col.split(' ')[0]

                        predictions=[]
                        actuals=[]

                        #Train a separate ARIMA model for each country (univariate time series):
                        model_arima = auto_arima(
                            train_data[country_col],seasonal=False,
                            trace=False,error_action='ignore',suppress_warnings=True)
                        
                        forecasted_values = model_arima.predict(n_periods=self.test_size)
                        
                        for start_idx in range(0,len(test_data) - self.seq_len - self.pred_len + 1):
                            # Define rolling input and prediction window
                            end_idx = start_idx + self.seq_len
                            pred_end_idx = end_idx + self.pred_len

                            input_sequence = test_data.iloc[start_idx:end_idx][country_col].values.flatten()
                            actual_sequence = test_data.iloc[end_idx:pred_end_idx][country_col].values.flatten()

                            forecast = forecasted_values[end_idx:pred_end_idx].values

                            # Store predictions and actuals
                            predictions.append(forecast)
                            actuals.append(actual_sequence)

                        
                        dict_results[current_country]['split_'+str(split+1)] = {
                                                    "predictions": predictions,
                                                    "actuals": actuals}
            all_univariate_forecasters_results[current_univariate_forecaster]=dict_results

        return all_univariate_forecasters_results

