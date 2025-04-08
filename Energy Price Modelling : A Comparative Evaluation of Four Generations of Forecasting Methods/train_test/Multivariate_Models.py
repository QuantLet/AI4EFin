import sys
sys.path.append("..")
from time_series_models.Basisformer.Basisformer_Utils.tools import EarlyStopping as EarlyStopping_Basisformer
from train_test.utils import  _acquire_device, _select_optimizer, _select_criterion,create_args,EarlyStopping, adjust_learning_rate
from typing import Callable, Any

import pandas as pd
import os 
import torch
import numpy as np
from torch.optim import lr_scheduler
import argparse


class Multivariate_Forecasting():
    """
    A class implementing the multivariate forecasting techniques with walk-forward validation.

    Parameters:
    ----------
    df : Pandas dataframe
        The dataframe containing the (electricity price-related) time series
    models_dataloaders_dict : dict
        The dict containing the pytroch dataloaders with the preprocessed data according to each models
        requirements.
    time_series_models_dict : dict
        The dictionary containing the mapping between the string names of the models and their callable modules.
    reverse_rescale : bool
        Whether the predictions should be reverse rescaled to the original scale.
    
    Attributes:
    (the same as the parameters)
    ----------

    Methods
    -------
    perform_walk_forward_validation():
        Generates the predictions using the walk-forward validation approach for all models
        included in models_dataloaders_dict.
    train_model(model,args,current_train_dataloader,criterion,device,
        model_optim,current_model,EarlyStopping):
        Trains each model on the specified subset saved in the current_train_dataloader.
    test_model(model,args,current_test_dataloader,device,
        current_model,reverse_rescale,current_scaler)
        Tests the trained model on the current out-of-time subset.
    """
    def __init__(self,
        df:pd.DataFrame,
        models_dataloaders_dict:dict,
        time_series_models_dict:dict,
        reverse_rescale:bool):

        self.df=df
        self.models_dataloaders_dict=models_dataloaders_dict
        self.time_series_models_dict=time_series_models_dict
        self.reverse_rescale=reverse_rescale
        super(Multivariate_Forecasting,self).__init__()

    def perform_walk_forward_validation(self)->dict:
        """Computes predictions using walk-forward validation.
        
        
        Returns:
        ----------
        all_models_results: dict
        The dictionary with the generated predictions for all models.

        """
        
        sys.argv=['']
        
        all_models_results={}
        #Select the device:
        for current_model in self.models_dataloaders_dict.keys():
            dataloaders_dict=self.models_dataloaders_dict[current_model]

            #del sys
            args=create_args(current_model=current_model,df=self.df)
            args.patience=1
            
            device = _acquire_device(args=args)

            
            #Select and initialize the model:
            if current_model in ['Autoformer','PatchTST','Quatformer','DLinear','NLinear','TSMixer']:
                model=self.time_series_models_dict[current_model](args).float()
            elif current_model=='Basisformer':
                model=self.time_series_models_dict[current_model](args.seq_len,args.pred_len,args.d_model,
                                                            args.heads,args.N,args.block_nums,
                                                            args.bottleneck,args.map_bottleneck,device,
                                                            args.tau).float()

            elif current_model=='Informer':
                model=self.time_series_models_dict[current_model](args.enc_in,args.dec_in, args.c_out, 
                                                            args.seq_len,args.label_len,args.pred_len, 
                                                            args.factor,args.d_model,args.n_heads, 
                                                            args.e_layers,args.d_layers,args.d_ff,
                                                            args.dropout,args.attn,args.embed,
                                                            args.freq,args.activation,args.output_attention,
                                                            args.distil,args.mix,device).float()
            else:
                continue
                
            #Select the optimizer and the current loss:   
            model_optim = _select_optimizer(model=model,args=args,
                                            current_model=current_model)
            criterion = _select_criterion()

            #Import the early stopping callback:
            if current_model=='Basisformer':
                current_early_stropping=EarlyStopping_Basisformer
            else:
                current_early_stropping=EarlyStopping
                
            #Prepare empty dictionary file for generated predictions:
            error_results={}

            print('Current Model: ',current_model)
                
            #Given the currently chosen time series model: loop through each train-test split & collect predictions on test set.
            for current_subset in dataloaders_dict.keys():#['Train_Test1','Train_Test2','Train_Test3',
                                #'Train_Test4','Train_Test5','Train_Test6']:
                print('Current Subset: ',current_subset)
                current_loaders=dataloaders_dict[current_subset]
                
                current_train_loader=current_loaders['Train_Set_Loader']
                current_test_loader=current_loaders['Test_Set_Loader']
                current_scaler=current_loaders['Scaler']

                
                #Initialize current transformer model:
                args.model=current_model
                
                avg_train_loss,trained_model=self.train_model(model=(model if current_subset=='Train_Test1' else trained_model),
                                                    args=args,
                                                    current_train_dataloader=current_train_loader,
                                                    criterion=criterion,
                                                    device=device,
                                                    model_optim=model_optim,
                                                    current_model=current_model,
                                                    EarlyStopping=current_early_stropping)
                
                model_batch_test_true,model_batch_test_predictions=self.test_model(model=trained_model,
                                                                                        args=args,
                                                                                        current_test_dataloader=current_test_loader,
                                                                                        device=device,
                                                                                        current_model=current_model,
                                                                                        reverse_rescale=self.reverse_rescale,
                                                                                        current_scaler=(current_scaler if self.reverse_rescale==True else None))
                
                error_results[current_subset]=[avg_train_loss,model_batch_test_true,model_batch_test_predictions]
                
                print('\n')
            
            all_models_results[current_model]=error_results

        return all_models_results
    
    def train_model(self,
                    model:torch.nn.Module,
                    args:argparse.ArgumentParser,
                    current_train_dataloader:torch.utils.data.DataLoader,
                    criterion:Any,#any loss from torch.nn
                    device:torch.device,
                    model_optim:torch.optim,
                    current_model:str,
                    EarlyStopping: Callable)->list:
        
        """Trains a multivariate time series model on a single subset of training data.
        
        Parameters:
        ----------
        model : torch.nn.Module
            The initialized pytroch time series model, which will be trained in this function.
        args : argparse.ArgumentParser
            The parsed arguments relevant for the training of the model, i.e., the mapping
            between argument names and their values, e.g., learning_rate, train_epochs etc.
        current_train_dataloader : torch.utils.data.DataLoader
            The current pytorch dataloader.
        criterion : loss function from torch.nn
            The loss function used to compute gradients during the training process.
        device: torch.device
            The device to be used during the training process, cpu or gpu.
        model_optim: torch.optim
            The optimizer utilized for the training process, e.g., Adam.
        current_model: str
            The name of the current model
        EarlyStopping: Callable
            A callable function utilized for the early stopping.
        
        Returns: a list containing two objects
        ----------
        np.mean(train_loss): np.float32|np.float64
            The average loss achieved on the training set from the last training epoch before the
            callback is activated.
        model: torch.nn.Module
            The trained pytroch model, which will be used in the testing phase.
        """
    
        model.train()
        iter_count = 0
        train_loss = []

        path = os.path.join(args.checkpoints, 'TS_Benchmark_training')
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=args.patience,
                                    delta=0.01,
                                    verbose=True)
        
        train_steps = len(current_train_dataloader)
        
        if current_model=='PatchTST':
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = args.pct_start,
                                                epochs = args.train_epochs,
                                                max_lr = args.learning_rate)

        
        epochs=args.train_epochs
        if args.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(0,epochs):
            print('Current Epoch: ',epoch+1)
            for i, train_items in enumerate(current_train_dataloader):
                
                #Get train items based on current model: ######################################################
                if current_model!='Basisformer':
                    batch_x, batch_y, batch_x_mark, batch_y_mark=train_items
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark,index=train_items
                    
                #End Train Items: #############################################################################


                iter_count += 1
                model_optim.zero_grad()
                #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                if current_model=='Basisformer':
                    #For basisformer this is done before the model training:
                    f_dim = -1 if args.features == 'MS' else 0
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    index=index.float().to(device)
                #End Data and Model on same Device: ############################################################


                # Prepare Decoder Input: ######################################################################
                if current_model=='Autoformer':
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                elif current_model=='Quatformer':
                    dec_inp = batch_x.mean(dim=1, keepdim=True).repeat(1,args.pred_len,1).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)


                elif current_model=='Informer':
                    if args.padding==0:
                        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    elif args.padding==1:
                        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float().to(device)

                
                elif current_model=='TSMixer':
                    dec_inp=None
                else:
                    if current_model not in ['DLinear','NLinear','PatchTST','Basisformer']:
                        print('Specified Model not in supported options!')
                #End Decoder Input Preparation: #################################################################


                #print('Training Step: ',i)
                #Perform the Forward Pass: encoder - decoder in case of transformer-based model chosen: #########
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if current_model in ['DLinear','NLinear','PatchTST']:
                            outputs=model(batch_x)
                        elif current_model=='Basisformer':
                            outputs,loss_infonce,loss_smooth,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index,batch_y,y_mark=batch_y_mark)
                        elif current_model=='Quatformer':
                            outputs, regularization_loss, _, _ = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, is_training=True)
                        else:
                            if args.output_attention:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)   
                else:
                    if current_model in ['DLinear','NLinear','PatchTST']:
                        outputs=model(batch_x)
                    elif current_model=='Basisformer':
                            outputs,loss_infonce,loss_smooth,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index,batch_y,y_mark=batch_y_mark)
                    elif current_model=='Quatformer':
                        outputs, _, regularization_loss = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, is_training=True)
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # End forward Pass: ###############################################################################
                #print('Shape of Predictions: ',outputs.shape)
                
                #Perform backward Pass: ###########################################################################
                if current_model!='Basisformer':
                    f_dim = -1 if args.features == 'MS' else 0
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    loss = criterion(outputs, batch_y)
                    if current_model=='Quatformer':
                        loss=loss+ regularization_loss
                else:
                    loss_p = criterion(outputs, batch_y)
                    lam1 = args.loss_weight_prediction
                    lam2 = args.loss_weight_infonce
                    lam3 = args.loss_weight_smooth
                    loss = lam1 * loss_p + lam2 * loss_infonce  + lam3 * loss_smooth
                    
                #Save the computed loss metric in the corresponding list:
                train_loss.append(loss.item())
                
                #print(loss.item())
                #Compute gradients:
                if args.use_amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(model_optim)
                    grad_scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                #End backward Pass: ################################################################################

            
            #Apply Early Stopping Callback: ########################################################################
            early_stopping(np.mean(train_loss), model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            #End Applying Early Stopping Callback: #################################################################

            
            #Adjust Learning Rate: #################################################################################
            #Performed if specified model is not Basisformer as it does not make adjustments to the learning rate:
            if current_model!='Basisformer':
                if current_model=='PatchTST':
                    if args.lradj == 'TST':
                        adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=False)
                        scheduler.step()
                else:
                    adjust_learning_rate(model_optim, epoch + 1, args)  
            #End adjusting learning Rate: ##########################################################################

        best_model_path = path+'/'+'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))  
        
        return np.mean(train_loss),model
    
    
    
    def test_model(self,
                model:torch.nn.Module,
                args:argparse.ArgumentParser,
                current_test_dataloader:torch.utils.data.DataLoader,
                device:torch.device,
                current_model:str,
                reverse_rescale:bool,
                current_scaler=Any|None)->list:
        
        """Tests a multivariate time series model on a single subset of out-of-time tes data.
        
        Parameters:
        ----------
        model : torch.nn.Module
            The already trained pytroch time series model.
        args : argparse.ArgumentParser
            The parsed arguments relevant for the training of the model, i.e., the mapping
            between argument names and their values, e.g., learning_rate, train_epochs etc.
        current_est_dataloader : torch.utils.data.DataLoader
            The current pytorch dataloader containing the test subset.
        device: torch.device
            The device to be used during the testing process, cpu or gpu.
        current_model: str
            The name of the current model
        reverse_rescale: bool
            Whether to reverse rescale predictions to the original scale
        current_scaler: Any|None
            The rescaler object from sklearn.preprocessing, which can be StandardScaler or MinMaxScaler etc.
            Only relevant if reverse_rescale=True.    
        
        Returns: a (nested) list with two objects
        ----------
        test_subset_true_values: list
            A list containing the true sequence values in the test subset.
        test_subset_predictions: list
            A list containing the predicted sequences.
        """
    
        model.eval()
        test_subset_predictions=[]
        test_subset_true_values=[]
        

        with torch.no_grad():
            for i, test_items in enumerate(current_test_dataloader):
                
                #Get train items based on current model: ######################################################
                if current_model!='Basisformer':
                    batch_x, batch_y, batch_x_mark, batch_y_mark=test_items
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark,index=test_items
                    
                #End Train Items: #############################################################################


                #print('Testing Step: ',i)
                #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                if current_model=='Basisformer':
                    #For basisformer this is done before the model training:
                    f_dim = -1 if args.features == 'MS' else 0
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                    index=index.float().to(device)
                #End Data and Model on same Device: ############################################################

                # Prepare Decoder Input: ######################################################################
                if current_model=='Autoformer':
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                elif current_model=='Quatformer':
                    dec_inp = batch_x.mean(dim=1, keepdim=True).repeat(1,args.pred_len,1).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

                elif current_model=='Informer':
                    if args.padding==0:
                        dec_inp = torch.zeros([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    elif args.padding==1:
                        dec_inp = torch.ones([batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
                    dec_inp = torch.cat([batch_y[:,:args.label_len,:], dec_inp], dim=1).float().to(device)

                
                elif current_model=='TSMixer':
                    dec_inp=None
                else:
                    if current_model not in ['DLinear','NLinear','PatchTST','Basisformer']:
                        print('Specified Model not in supported options!')
                #End Decoder Input Preparation: #################################################################


                #Perform the Prediction (Forward Pass): encoder - decoder in case of transformer-based model chosen
                if current_model in ['DLinear','NLinear','PatchTST']:
                    outputs=model(batch_x)
                elif current_model=='Basisformer':
                        outputs,m,attn_x1,attn_x2,attn_y1,attn_y2 = model(batch_x,index,batch_y,train=False,y_mark=batch_y_mark)
                elif current_model=='Quatformer':
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # End Prediction (forward Pass): ##################################################################

                #Rescale true value and predicted values: #########################################################
                if current_model!='Basisformer':
                    f_dim = -1 if args.features == 'MS' else 0
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                
                #Reverse rescale the predictions: ##################################################################
                true_batch_values=batch_y.detach().cpu().numpy()
                batch_predictions=outputs.detach().cpu().numpy()
                
                if reverse_rescale==True:
                    #Reverse rescale predictions & true values:
                    true_batch_values=np.array([current_scaler.inverse_transform(tr_batch) for tr_batch in true_batch_values])
                    batch_predictions=np.array([current_scaler.inverse_transform(batch_pr) for batch_pr in batch_predictions])
                #print(true_batch_values.shape)
                #print(batch_predictions.shape)

                test_subset_true_values.append(true_batch_values)
                test_subset_predictions.append(batch_predictions)
                #Done reverse rescaling Predictions: ###############################################################
                    
        return test_subset_true_values,test_subset_predictions

