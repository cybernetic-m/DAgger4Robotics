import os
import sys

import torch
import time
import json
from tqdm import tqdm

# Get the absolute paths of the directories containing the utils functions 
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from calculate_metrics import calculate_metrics

def test(model, test_dataloader, loss_fn, device):

    model.eval()
    
    # Definition of the metrics dictionary
    test_metrics = {
        'rmse': [],
        'mae': [],
        'r2': [],
        'loss': [],
        'inference_time_avg': [],
    }

    test_loss = 0
    a_pred_cat = None   # Initialize as None because at first iteration it will save the first batch of predictions
    a_hat_cat = None
    inference_time_list = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing..."):
            inputs = batch['observations'][:, :-1].float().to(device)   # Inputs to the model, excluding the last time step
            start_time = time.time()    # Start the timer for inference
            a_pred = model(inputs)      #  Predict actions using the model
            inference_time_list.append(time.time() - start_time)    # Record the inference time

            a_hat = batch["actions"].to(device) # Ground truth actions from the batch

            # Concatenate the tensors(the predicted and ground truth actions)
            # At first iteration, a_pred_cat and a_hat_cat are None, so they will be initialized with the first batch predictions and ground truth
            # At subsequent iterations, they will concatenate the new batch predictions and ground truth actions
            if a_pred_cat is None:
                a_pred_cat = a_pred
            else:
                a_pred_cat = torch.cat((a_pred_cat, a_pred), dim=0)

            if a_hat_cat is None:
                a_hat_cat = a_hat
            else:
                a_hat_cat = torch.cat((a_hat_cat, a_hat), dim=0)


            loss = loss_fn(a_pred, a_hat)   # Calculate the loss between predicted and ground truth actions
            test_loss += loss.item() # Accumulate the loss

    # Calculate the average loss over all batches
    test_loss_avg = test_loss / len(test_dataloader)

    # Convert the concatenated tensors to numpy arrays for metric calculation
    a_pred_numpy = a_pred_cat.view(-1, 2).cpu().detach().numpy()
    a_hat_numpy = a_hat_cat.view(-1, 2).cpu().detach().numpy()

    # Calculate the metrics using the utility function
    rmse, mae, r2 = calculate_metrics(y_true_list=a_hat_numpy, y_pred_list=a_pred_numpy, metrics=test_metrics)
    inference_time_avg = sum(inference_time_list) / len(inference_time_list)
    test_metrics['loss'].append(test_loss_avg)
    test_metrics['inference_time_avg'].append(inference_time_avg)

    print(f"Test Loss: {test_loss_avg:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    print(f"Average Inference Time per Batch: {inference_time_avg:.4f} seconds")


    


