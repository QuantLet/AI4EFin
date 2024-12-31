import pickle
from permetrics import RegressionMetric
import pandas as pd
import numpy as np


def evaluate_results(sequence_predictions,EU_countries_dict):
    #Evaluation of forecasts produced by second and third generation: ##################################
    models=['PatchTST','DLinear','NLinear','TSMixer','Autoformer','Basisformer','Informer','Quatformer']

    model_names_array=[]
    subset_names_array=[]
    batch_sample_indices=[]
    country_names=[]

    rmse_values_array=[]
    mae_values_array=[]
    mse_values_array=[]
    smape_values_array=[]
    r2_values_array=[]
    
    pairwise_comparisons_dict={}

    for current_model in models:
        #print('Current Model: ',current_model)
        with open('../evaluation/Results/'+current_model+'_Results_per_Subset.pkl','rb') as f:
            current_results=pickle.load(f)

            all_subsets_idx=current_results.keys()
            pairwise_comparisons_dict[current_model]={}

            for subsets_idx in all_subsets_idx:
                #print('- Current Train & Test Split: ',subsets_idx)
                true_batches=current_results[subsets_idx][-3]
                predicted_batches=current_results[subsets_idx][-2]
                pairwise_comparisons_dict[current_model][subsets_idx]=[]
                
                #last_seq_idx=0
                index_to_save=0
                for batch_idx in range(0,len(true_batches)):
                    true_mini_batch=true_batches[batch_idx]
                    predicted_mini_batch=predicted_batches[batch_idx]

                    for seq_batch_idx in range(0,true_mini_batch.shape[0]):
                        current_true_sequence=true_mini_batch[seq_batch_idx,:,:]
                        current_predicted_sequence=predicted_mini_batch[seq_batch_idx,:,:]
                        #pairwise_comparisons_dict[current_model][subsets_idx].extend(current_predicted_sequence.flatten())
                        
                        
                        #Loop through each country and compute error per country:###########################
                        for country_idx in range(0,current_true_sequence.shape[1]):
                            current_true_sequence_country=current_true_sequence[:,country_idx]
                            current_predicted_sequence_country=current_predicted_sequence[:,country_idx]
                            
                            evaluator = RegressionMetric(y_true=current_true_sequence_country,
                                                        y_pred=current_predicted_sequence_country)
                            results = evaluator.get_metrics_by_list_names(["RMSE", "MAE", "MSE","SMAPE", "R2"])
                            rmse_values_array.append(results["RMSE"])
                            mae_values_array.append(results["MAE"])
                            mse_values_array.append(results["MSE"])
                            smape_values_array.append(results["SMAPE"])
                            r2_values_array.append(results["R2"])

                            #current_true_binary_seq=np.where(current_true_sequence_country<0.0,1.0,0.0)
                            #current_predicted_binary_seq=np.where(current_predicted_sequence_country<0.0,1.0,0.0)
                            #tn, fp, fn, tp = confusion_matrix(y_true=current_true_binary_seq,
                            #                                  y_pred=current_predicted_binary_seq).ravel()
                            #fnr=0.0#fn/(fn+tp)
                            #fpr=0.0#fp/(fp+tn)
                            #fnr_values_array.append(fnr)
                            #fpr_values_array.append(fpr)
                            
                            model_names_array.append(current_model)
                            subset_names_array.append(subsets_idx)
                            batch_sample_indices.append(index_to_save)#seq_batch_idx)
                            country_names.append(EU_countries_dict[country_idx])
                            #Save Predicitons vs True Values in a separate dictionary for visualization purposes with classical time series plots:
                            if EU_countries_dict[country_idx] in ["Germany","Latvia","France","Luxembourg"] and current_model=="PatchTST" and index_to_save in [0,1,2,154,307]:
                                sequence_predictions[EU_countries_dict[country_idx]][current_model][subsets_idx][index_to_save]={"True":current_true_sequence_country,
                                                                                                                                "Predicted":current_predicted_sequence_country}
                                
                        index_to_save=index_to_save+1
        ####################################################################################

    #Evaluation of forecasts produced by first and fourth generation: #################################################
    with open('../evaluation/Results/'+'results_TimesFM_GPU_27_countries.pkl','rb') as f:
        current_results_TimesFM=pickle.load(f)

    with open('../evaluation/Results/'+'results_Chronos_GPU_27_countries.pkl','rb') as f:
        current_results_Chronos=pickle.load(f)

    with open('../evaluation/Results/'+'results_ARIMA_GPU_27_countries.pkl','rb') as f:
            current_results_ARIMA=pickle.load(f)
        
    subset_mapping_dict={}
    for i in range(0,6):
        subset_mapping_dict[i]="Train_Test"+str(i+1)

    for country_key in current_results_Chronos.keys():
        country_name=country_key.split("_")[1]
        
        start_idx=0
        for i in range(0,6):
            end_idx=start_idx+308
            name_current_subset=subset_mapping_dict[i]
            current_testSubset_true_values=current_results_Chronos[country_key]["actuals"][start_idx:end_idx]
            current_testSubset_chronos_predictions=current_results_Chronos[country_key]["predictions"][start_idx:end_idx]
            current_testSubset_timesfm_predictions=current_results_TimesFM[country_key]["predictions"][start_idx:end_idx]
            
            for seq_sample_idx in range(0,current_testSubset_true_values.shape[0]):
                current_true_values=current_testSubset_true_values[seq_sample_idx,:]
                current_chronos_prediction_values=current_testSubset_chronos_predictions[seq_sample_idx,:]
                current_timesfm_prediction_values=current_testSubset_timesfm_predictions[seq_sample_idx,:]
                    
                
                #Save results for Chronos:
                evaluator_chronos = RegressionMetric(y_true=current_true_values,
                                                        y_pred=current_chronos_prediction_values)
                results_chronos = evaluator_chronos.get_metrics_by_list_names(["RMSE", "MAE", "MSE","SMAPE", "R2"])
                rmse_values_array.append(results_chronos["RMSE"])
                mae_values_array.append(results_chronos["MAE"])
                mse_values_array.append(results_chronos["MSE"])
                smape_values_array.append(results_chronos["SMAPE"])
                r2_values_array.append(results_chronos["R2"])
                model_names_array.append("Chronos")
                subset_names_array.append(name_current_subset)
                batch_sample_indices.append(seq_sample_idx)
                country_names.append(country_name)
                
                #Save results for TimesFM:
                evaluator_timesfm = RegressionMetric(y_true=current_true_values,
                                                        y_pred=current_timesfm_prediction_values)
                results_timesfm = evaluator_timesfm.get_metrics_by_list_names(["RMSE", "MAE", "MSE","SMAPE", "R2"])
                rmse_values_array.append(results_timesfm["RMSE"])
                mae_values_array.append(results_timesfm["MAE"])
                mse_values_array.append(results_timesfm["MSE"])
                smape_values_array.append(results_timesfm["SMAPE"])
                r2_values_array.append(results_timesfm["R2"])
                model_names_array.append("TimesFM")
                subset_names_array.append(name_current_subset)
                batch_sample_indices.append(seq_sample_idx)
                country_names.append(country_name) 
                
                #Save Predicitons vs True Values in a separate dictionary for visualization purposes:
                if country_name in ["Germany","Latvia","France","Luxembourg"] and seq_sample_idx in [0,1,2,154,307]:
                    sequence_predictions[country_name]["TimesFM"][name_current_subset][seq_sample_idx]={"True":current_true_values,
                                                                                    "Predicted":current_timesfm_prediction_values}
            
            start_idx=end_idx+96*2


    for country_name in current_results_ARIMA.keys():
        start_idx=0
        for i in range(1,7):
            name_current_subset=subset_mapping_dict[i-1]
            current_testSubset_true_values=current_results_ARIMA[country_name]['split_'+str(i)]["actuals"][:-1]
            current_testSubset_arima_predictions=current_results_ARIMA[country_name]['split_'+str(i)]["predictions"][:-1]
            
            for seq_sample_idx in range(0,len(current_testSubset_true_values)):
                current_true_values=current_testSubset_true_values[seq_sample_idx]
                current_arima_prediction_values=current_testSubset_arima_predictions[seq_sample_idx].values
                        
                #Save results for ARIMA:
                evaluator_arima = RegressionMetric(y_true=current_true_values,
                                                        y_pred=current_arima_prediction_values)
                results_arima = evaluator_arima.get_metrics_by_list_names(["RMSE", "MAE", "MSE","SMAPE", "R2"])
                rmse_values_array.append(results_arima["RMSE"])
                mae_values_array.append(results_arima["MAE"])
                mse_values_array.append(results_arima["MSE"])
                smape_values_array.append(results_arima["SMAPE"])
                r2_values_array.append(results_arima["R2"])
                model_names_array.append("ARIMA")
                subset_names_array.append(name_current_subset)
                batch_sample_indices.append(seq_sample_idx)
                country_names.append(country_name) 
    ##################################################################################################################




    timeSeries_benchmark_results=pd.DataFrame({'Time_Series_Model':np.array(model_names_array),
                                            'TrainTest_Subset':np.array(subset_names_array),
                                            'Sequence_MiniBatch_Index':np.array(batch_sample_indices),
                                            'EU_Country':np.array(country_names),
                                                "Achieved_RMSE":np.array(rmse_values_array),
                                                "Achieved_MAE":np.array(mae_values_array),
                                                "Achieved_MSE":np.array(mse_values_array),
                                                "Achieved_SMAPE":np.array(smape_values_array),
                                                "Achieved_R2":np.array(r2_values_array)})
    return timeSeries_benchmark_results