from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import pandas as pd
import numpy as np
import torch
import warnings
from functools import wraps


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

class Multivaraite_TimeSeries_Preprocesser():
    """
    A class implementing the time series prerocessing for multivariate forecasting techniques:

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
    batch_size : int
        The number of sequence samples in each mini-batch in the pytorch dataloaders.
    
    Attributes:
    (in addition to the parameters)
    ----------
    label_len : int
        Defined as the half of the prediction horizon. This parameter is only relevant for 
        transformer based techniques.
    

    Methods
    -------
    get_TrainTest_splits(timestamp_data,test_set_names):
        Generates all possible train and test subsets based on the test subset names provided
        as an argument to the function. 
    create_dataloader(current_subset,current_timestamps_subset,current_model):
        Creates a pytorch dataloader for a pre-defined train and test subsets.
    get_timestamp_data:
        Extracts time covariates from the original time series dates.
    walk_forward_validation(df_stamp,current_model):
        Splits the original time series into multiple chunks fo train and test subsets suitable for walk-forward
        validation, rescaled the subsets and saves them in a dictionary.
    preprocess_data(current_multivariate_forecasters):
        Generates the preprocessed data as pytroch dataloaders for all specified forecasters.
        The function essentially calls multiple times walk_forward_validation for all specified prediction techniques.
    
    """

    def __init__(self,
                 df:pd.DataFrame,
                 train_size:int,
                 test_size:int,
                 seq_len:int,
                 pred_len:int,
                 batch_size:int):
        
        self.df=df
        self.train_size=train_size
        self.test_size=test_size
        self.seq_len=seq_len
        self.pred_len=pred_len
        self.label_len=int(self.pred_len/2)
        self.batch_size=batch_size

        super(Multivaraite_TimeSeries_Preprocesser,self).__init__()

    def preprocess_data(self,current_multivariate_forecasters:list)->dict:
        """Preprocesses multivariate time series for the technical requirements of the specified current_multivariate_forecasters.
        
        Parameters:
        ----------
        current_multivariate_forecasters: list
        The list containing the names of the multivariate forecasting techniques.
        
        Returns:
        ----------
        models_dataloaders_dict: dict
        The dictionary containing the pytroch dataloaders assigned to each model key.

        Raises:
        ----------
        ValueError: If specified current_multivariate_forecasters are not in the list 
        ['Basisformer','Autoformer','Informer','PatchTST','Quatformer','DLinear','NLinear','TSMixer'].
        """

        diff=set(current_multivariate_forecasters).difference(['Basisformer','Autoformer','Informer','PatchTST','Quatformer','DLinear','NLinear','TSMixer'])
        if list(diff)!=[]:
            raise ValueError(f'The currently specified multivariate forecasting techniques {list(diff)} are not in the list of supported methods.')
        
        df_timestamp=self.get_timestamp_data()

        models_dataloaders_dict={}
        for current_model in current_multivariate_forecasters:
            dataloaders_dict=self.walk_forward_validation(df_stamp=df_timestamp,
                                                     current_model=current_model)
            models_dataloaders_dict[current_model]=dataloaders_dict.copy()

        return models_dataloaders_dict
    
    def get_TrainTest_splits(self,
                             timestamp_data:pd.DataFrame,
                             test_set_names:list)->list:
    
        """Generates multiple train-test splits based on the pre-defined test subset names.
        
        Parameters:
        ----------
        timestamp_data: pd.DataFrame
        The dataframe of time covariates.
        test_set_names: list
        The names of the test subsets.

        Returns:
        ----------
        train_test_splits: list
        A list containing the training, testing subsets and the corresponding time series covariate subsets.

        """

        data=self.df.copy()
        train_test_splits=[]
        for current_test_set_name in test_set_names:

            last_test_set_index=data[data['Subset']==current_test_set_name].index[-1]
            first_index_train_set=last_test_set_index-timedelta(hours=self.train_size+self.test_size-1)

            current_train_test_split=data.loc[first_index_train_set:last_test_set_index,:]
            current_train_subset=current_train_test_split[current_train_test_split['Subset']!=current_test_set_name]
            current_test_subset=current_train_test_split[current_train_test_split['Subset']==current_test_set_name]

            current_timestamp_split=timestamp_data.loc[first_index_train_set:last_test_set_index,:]
            current_train_timestamps=current_timestamp_split[current_timestamp_split['Subset']!=current_test_set_name]
            current_test_timestamps=current_timestamp_split[current_timestamp_split['Subset']==current_test_set_name]

            #Drop date & subset columns from current train test split & timestamps:
            current_train_subset=current_train_subset.drop(['date','Subset'],axis=1)
            current_test_subset=current_test_subset.drop(['date','Subset'],axis=1)

            current_train_timestamps=current_train_timestamps.drop(['Subset'],axis=1)
            current_test_timestamps=current_test_timestamps.drop(['Subset'],axis=1)


            train_test_splits.append([current_train_subset,current_test_subset,current_train_timestamps,current_test_timestamps])

        return train_test_splits

    def create_dataloader(self,current_subset:pd.DataFrame,
                          current_timestamps_subset:pd.DataFrame,
                          current_model:str)->torch.utils.data.DataLoader:
        """Creates a pytorch dataloader for the specified multivariate time series forecasting technique.
        
        Parameters:
        ----------
        current_subset: pd.DataFrame
        The current subset of multivariate time series.
        current_timestamps_subset: pd.DataFrame
        The current subset of time covariates.
        current_model: str
        The name of the current multivariate time series forecasting technique.

        Returns:
        ----------
        current_dataloader: torch.utils.data.DataLoader
        Pytorch dataloader for a single train-test / walk-forward validation split.

        """
        
        inputs_for_dataloader=[]
        
        for timestamp_idx in range(0,current_subset.shape[0]-self.seq_len-self.pred_len):
            s_begin = timestamp_idx
            s_end = s_begin + self.seq_len
            
            #used for building a feature sequence!
            seq_x=current_subset.iloc[s_begin:s_end].values
            seq_x_mark=current_timestamps_subset.iloc[s_begin:s_end,:].values
            
            if current_model!='Crossformer' and current_model!='Chronos':
                r_begin = s_end - self.label_len
                #POTENTIALLY ADJUST FOR DIFFERENT MODELS by passing argument current_transformer_model to the function!
                r_end = r_begin + self.label_len + self.pred_len
            else:
                r_begin=s_end
                r_end=r_begin+self.label_len
            
            #used for building targets!
            seq_y=current_subset.iloc[r_begin:r_end,:].values
            seq_y_mark=current_timestamps_subset.iloc[r_begin:r_end].values

            #If rescale:
            #rescale based on input sequence until start token:
            if current_model not in ['Basisformer','Crossformer','Chronos']:
                inputs_for_dataloader.append([seq_x, seq_y, seq_x_mark, seq_y_mark])
            else:
                if current_model=='Basisformer':
                    index_list = np.arange(timestamp_idx,timestamp_idx+self.seq_len+self.pred_len,1)
                    ## normalize the index
                    len_denominator=len(current_subset) - self.seq_len - self.pred_len + 1
                    norm_index = index_list / len_denominator
                    inputs_for_dataloader.append([seq_x, seq_y, seq_x_mark, seq_y_mark,norm_index])
                else:#current_model=Crossformer
                    inputs_for_dataloader.append([seq_x, seq_y])
                    
            
        #Save in dataloader:
        current_dataloader=torch.utils.data.DataLoader(inputs_for_dataloader,batch_size=self.batch_size,shuffle=False)
        return current_dataloader
    
    @ignore_warnings
    def get_timestamp_data(self)->pd.DataFrame:
        """Create a dataframe of the time features using the original time series dataset.
        
        
        Returns:
        ----------
        df_stamp: pd.DataFrame
        The dataframe containing the extracted time fgeatures, e.g., month, day, week etc.

        """
        df_stamp=self.df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(self.df.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)+1
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp = df_stamp.drop(['date'], axis=1)
        df_stamp['Subset']=np.array(self.df['Subset'])

        return df_stamp
        

    def walk_forward_validation(self,
                                df_stamp:pd.DataFrame,
                                current_model:str)->dict:
        """Preprocesses multivariate time series for walk-forward validation purposes.

        Parameters:
        ----------
        df_stamp: pd.DataFrame
        The dataframe containing the time covariates.
        current_model: str
        The name of the time series forecasting technique to generate walk-forward validation splits.

        Returns:
        ----------
        dataloaders_dict: dict
        The dictionary containing the pytroch dataloaders for a specific model suitable for walk-forward validation.

        """

        dataloaders_dict={}

        all_TrainTest_splits=self.get_TrainTest_splits(timestamp_data=df_stamp,
                                                  test_set_names=list(self.df['Subset'].value_counts().index[1:]))

        for split_index in range(0,len(all_TrainTest_splits)):
            nr_TrainTest_split='Train_Test'+str(split_index+1)
            current_train_subset=all_TrainTest_splits[split_index][0]
            current_test_subset=all_TrainTest_splits[split_index][1]
            current_train_timestamps=all_TrainTest_splits[split_index][2]
            current_test_timestamps=all_TrainTest_splits[split_index][3]

            #Rescale Train Set & based on statistics also Test Set:
            current_scaler= StandardScaler()
            current_train_subset=pd.DataFrame(current_scaler.fit_transform(current_train_subset),
                                            columns=current_train_subset.columns)
            current_test_subset=pd.DataFrame(current_scaler.transform(current_test_subset),
                                            columns=current_test_subset.columns)

            #Create data loaders:
            current_train_dataloader=self.create_dataloader(current_subset=current_train_subset,
                                                            current_timestamps_subset=current_train_timestamps,
                                                            current_model=current_model)
            
            current_test_dataloader=self.create_dataloader(current_subset=current_test_subset,
                                                           current_timestamps_subset=current_test_timestamps,
                                                           current_model=current_model)


            dataloaders_dict[nr_TrainTest_split]={'Train_Set_Loader':current_train_dataloader,
                                                'Test_Set_Loader':current_test_dataloader,
                                                'Scaler':current_scaler}
        return dataloaders_dict

