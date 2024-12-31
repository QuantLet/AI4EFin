import os 
import torch
import numpy as np
from torch.optim import lr_scheduler


def train_model(model,
                args,
                current_train_dataloader,
                criterion,
                device,
                model_optim,
                current_model,
                EarlyStopping,
                adjust_learning_rate):
    
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
            if current_model not in ['Basisformer','Crossformer']:
                batch_x, batch_y, batch_x_mark, batch_y_mark=train_items
            else:
                if current_model=='Basisformer':
                    batch_x, batch_y, batch_x_mark, batch_y_mark,index=train_items
                else:#current_model=Corssformer
                    batch_x, batch_y=train_items
            #End Train Items: #############################################################################


            iter_count += 1
            model_optim.zero_grad()
            #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if current_model!='Crossformer':
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
                if current_model not in ['DLinear','NLinear','PatchTST','Basisformer','Crossformer']:
                    print('Specified Transformer not in supported options!')
            #End Decoder Input Preparation: #################################################################


            print('Training Step: ',i)
            #Perform the Forward Pass: encoder - decoder in case of transformer-based model chosen: #########
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if current_model=='DLinear' or current_model=='NLinear' or current_model=='PatchTST' or current_model=='Crossformer':
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
                if current_model=='DLinear' or current_model=='NLinear' or current_model=='PatchTST' or current_model=='Crossformer':
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
            if current_model not in ['Basisformer','Crossformer']:
                f_dim = -1 if args.features == 'MS' else 0
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
                loss = criterion(outputs, batch_y)
                if current_model=='Quatformer':
                    loss=loss+ regularization_loss
            else:
                if current_model=='Basisformer':
                    loss_p = criterion(outputs, batch_y)
                    lam1 = args.loss_weight_prediction
                    lam2 = args.loss_weight_infonce
                    lam3 = args.loss_weight_smooth
                    loss = lam1 * loss_p + lam2 * loss_infonce  + lam3 * loss_smooth
                else:#current_model=Crossformer
                    loss = criterion(outputs, batch_y)
            
            #Save the computed loss metric in the corresponding list:
            train_loss.append(loss.item())
            
            print(loss.item())
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