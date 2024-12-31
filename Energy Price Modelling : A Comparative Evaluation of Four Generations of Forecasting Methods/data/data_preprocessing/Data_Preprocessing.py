from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import pandas as pd
import numpy as np
import torch

def get_TrainTest_splits(data,timestamp_data,train_size,
                         test_size,test_set_names):
    
    train_test_splits=[]
    for current_test_set_name in test_set_names:

        last_test_set_index=data[data['Subset']==current_test_set_name].index[-1]
        first_index_train_set=last_test_set_index-timedelta(hours=train_size+test_size-1)

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

def create_dataloader(current_subset,current_timestamps_subset,
                      seq_len,pred_len,label_len,batch_size,current_model):
    
    inputs_for_dataloader=[]
    
    for timestamp_idx in range(0,current_subset.shape[0]-seq_len-pred_len):
        s_begin = timestamp_idx
        s_end = s_begin + seq_len
        
        #used for building a feature sequence!
        seq_x=current_subset.iloc[s_begin:s_end].values
        seq_x_mark=current_timestamps_subset.iloc[s_begin:s_end,:].values
        
        if current_model!='Crossformer' and current_model!='Chronos':
            r_begin = s_end - label_len
            #POTENTIALLY ADJUST FOR DIFFERENT MODELS by passing argument current_transformer_model to the function!
            r_end = r_begin + label_len + pred_len
        else:
            r_begin=s_end
            r_end=r_begin+label_len
        
        #used for building targets!
        seq_y=current_subset.iloc[r_begin:r_end,:].values
        seq_y_mark=current_timestamps_subset.iloc[r_begin:r_end].values

        #If rescale:
        #rescale based on input sequence until start token:
        if current_model not in ['Basisformer','Crossformer','Chronos']:
            inputs_for_dataloader.append([seq_x, seq_y, seq_x_mark, seq_y_mark])
        else:
            if current_model=='Basisformer':
                index_list = np.arange(timestamp_idx,timestamp_idx+seq_len+pred_len,1)
                ## normalize the index
                len_denominator=len(current_subset) - seq_len - pred_len + 1
                norm_index = index_list / len_denominator
                inputs_for_dataloader.append([seq_x, seq_y, seq_x_mark, seq_y_mark,norm_index])
            else:#current_model=Crossformer
                inputs_for_dataloader.append([seq_x, seq_y])
                
        
    #Save in dataloader:
    current_dataloader=torch.utils.data.DataLoader(inputs_for_dataloader,batch_size=batch_size,shuffle=False)
    return current_dataloader

def walk_forward_validation(df,df_stamp,train_size,test_size,seq_len,pred_len,label_len,batch_size,current_model):
    dataloaders_dict={}
    all_TrainTest_splits=get_TrainTest_splits(data=df,timestamp_data=df_stamp,
                                            train_size=train_size,test_size=test_size,
                                            test_set_names=df['Subset'].value_counts().index[1:])

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
        current_train_dataloader=create_dataloader(current_subset=current_train_subset,
                                                    current_timestamps_subset=current_train_timestamps,
                                                    seq_len=seq_len,pred_len=pred_len,label_len=label_len,
                                                    batch_size=batch_size,current_model=current_model)
        
        current_test_dataloader=create_dataloader(current_subset=current_test_subset,
                                                    current_timestamps_subset=current_test_timestamps,
                                                    seq_len=seq_len,pred_len=pred_len,label_len=label_len,
                                                    batch_size=batch_size,current_model=current_model)


        dataloaders_dict[nr_TrainTest_split]={'Train_Set_Loader':current_train_dataloader,
                                            'Test_Set_Loader':current_test_dataloader,
                                            'Scaler':current_scaler}
    return dataloaders_dict

def get_timestamp_data(df,subsets):
    df_stamp=df[['date']]
    df_stamp['date'] = pd.to_datetime(df.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)+1
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    df_stamp = df_stamp.drop(['date'], axis=1)
    df_stamp['Subset']=np.array(subsets)

    return df_stamp

