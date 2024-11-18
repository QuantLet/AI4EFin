import torch
import numpy as np

def test_model(model,
                args,
                current_test_dataloader,
                current_scaler,
                device,
                current_model):
    
    model.eval()
    test_subset_predictions=[]
    test_subset_true_values=[]
    

    with torch.no_grad():
        for i, test_items in enumerate(current_test_dataloader):
            
            #Get train items based on current model: ######################################################
            if current_model not in ['Basisformer','Crossformer','Chronos']:
                batch_x, batch_y, batch_x_mark, batch_y_mark=test_items
            else:
                if current_model=='Basisformer':
                    batch_x, batch_y, batch_x_mark, batch_y_mark,index=test_items
                else:#current_model=Corssformer
                    batch_x, batch_y=test_items
            #End Train Items: #############################################################################


            print('Testing Step: ',i)
            #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if current_model not in ['Crossformer','Chronos']:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
            if current_model=='Basisformer':
                #For basisformer this is done before the model training:
                f_dim = -1 if args.features == 'MS' else 0
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                index=index.float().to(device)
            #End Data and Model on same Device: ############################################################

            # Prepare Decoder Input: ######################################################################
            if current_model in ['FEDformer','Autoformer']:
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

            elif current_model=='TimeMixer':
                if args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                else:
                    dec_inp = None

            elif current_model=='TSMixer':
                dec_inp=None
            else:
                if current_model not in ['DLinear','NLinear','PatchTST','Basisformer','Crossformer','Chronos']:
                    print('Specified Transformer not in supported options!')
            #End Decoder Input Preparation: #################################################################


            #Perform the Prediction (Forward Pass): encoder - decoder in case of transformer-based model chosen
            if current_model in ['DLinear','NLinear','PatchTST','Crossformer','Chronos']:
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
            if current_model not in ['Basisformer','Crossformer','Chronos']:
                f_dim = -1 if args.features == 'MS' else 0
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            #Reverse rescale the predictions: ##################################################################
            true_batch_values=batch_y.detach().cpu().numpy()
            batch_predictions=outputs.detach().cpu().numpy()
            
            #Reverse rescale predictions & true values:
            true_batch_values=np.array([current_scaler.inverse_transform(tr_batch) for tr_batch in true_batch_values])
            batch_predictions=np.array([current_scaler.inverse_transform(batch_pr) for batch_pr in batch_predictions])
            print(true_batch_values.shape)
            print(batch_predictions.shape)

            test_subset_true_values.append(true_batch_values)
            test_subset_predictions.append(batch_predictions)
            #Done reverse rescaling Predictions: ###############################################################
                
    return test_subset_true_values,test_subset_predictions