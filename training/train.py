import json
import torch
from IL4ReacherEnv.training.one_epoch import one_epoch
from IL4ReacherEnv.utils.calculate_metrics import calculate_metrics

def train(train_loader, val_loader, model, optimizer, loss_fn, num_epochs, device):

  epoch_loss = 0
  epoch_val_loss = 0
  best_vloss = 10000000 # Starting value of the best validation loss to save the model

  train_metrics = {
      'rmse': [],
      'mae': [],
      'r2': [],
      'loss': [],
  }
  val_metrics = {
      'rmse': [],
      'mae': [],
      'r2': [],
      'loss': [],
  }

  for epoch in range(num_epochs):
    print(f"EPOCH: {epoch+1}/{num_epochs}")

    # At this stage train_a_hat_list is a list of tensors of different shapes (all (256,50,2) but the last (88,50,2) because is the last batch
    #[
    #  [ [[1, 2], [3,4], ..., [10,11]], ..., [[1.1, 2.2], [3.3,4.4], ..., [10.1,11.1]] ],  list element 1: tensor (256, 50, 2)
    #  [ [[1, 2], [3,4], ..., [10,11]], ..., [[1.1, 2.2], [3.3,4.4], ..., [10.1,11.1]] ],  list element 2: tensor (256, 50, 2)
    #   ...,
    #  [ [[1, 2], [3,4], ..., [10,11]], ..., [[1.1, 2.2], [3.3,4.4], ..., [10.1,11.1]] ],  list element 28: tensor (88, 50, 2)
    # ]
    epoch_loss, train_a_hat_list, train_a_pred_list = one_epoch(train_loader, model, optimizer, loss_fn, device)
    epoch_vloss, val_a_hat_list, val_a_pred_list = one_epoch(val_loader, model, optimizer, loss_fn, device, validation = True)

    # Concatenate all the tensors in the list through the batch dimension (256 * 27 + 88 = 7000) => (7000, 50, 2) becoming a big tensor
    #[
    #  [[1, 2], [3,4], ... [10,11]],  sample 1: 50 steps of 2D features
    #  [[1.6, 2.2], [3.1,4.2], ... [10.1,11.4]],  sample 2: 50 steps of 2D features
    #   ...,
    #  [[1.9, 2.8], [3.6,4.9], ... [10.8,11.7]],  sample 7000: 50 steps of 2D features
    # ]
    train_a_hat_tensor = torch.cat(train_a_hat_list, dim=0)
    train_a_pred_tensor = torch.cat(train_a_pred_list, dim=0)
    val_a_hat_tensor = torch.cat(val_a_hat_list, dim=0)
    val_a_pred_tensor = torch.cat(val_a_pred_list, dim=0)

    # Reshape (flattening) the tensor that (7000, 50, 2) => (7000*50, 2) => (350000, 2), it means 350000 pairs of torques and transform in numpy array
    #[
    #  [1, 2],    sample 1
    #  [3,4],     sample 2
    #   ...,
    #  [9, 10]    sample 35000
    # ]
    train_a_hat_numpy = train_a_hat_tensor.view(-1,2).cpu().detach().numpy()
    train_a_pred_numpy = train_a_pred_tensor.view(-1,2).cpu().detach().numpy()
    val_a_hat_numpy = val_a_hat_tensor.view(-1,2).cpu().detach().numpy()
    val_a_pred_numpy = val_a_pred_tensor.view(-1,2).cpu().detach().numpy()

    print(f"Training Loss: {epoch_loss}")
    print(f"Validation Loss: {epoch_vloss}")

    if epoch_vloss < best_vloss:
      best_vloss = epoch_vloss
      torch.save(model.state_dict(), "expert_policy.pt")
      print(f"Model saved as expert_policy.pt")

    # Compute the metrics and append in the dictionary
    calculate_metrics(y_true_list = train_a_hat_numpy, y_pred_list= train_a_pred_numpy, metrics = train_metrics)
    calculate_metrics(y_true_list = val_a_hat_numpy, y_pred_list= val_a_pred_numpy, metrics = val_metrics)
    train_metrics['loss'].append(epoch_loss)
    val_metrics['loss'].append(epoch_vloss)

  # Save the metrics in json
  with open("train_metrics.json", "w") as f:
    json.dump(train_metrics, f, indent=4)
  with open("val_metrics.json", "w") as f:
    json.dump(val_metrics, f, indent=4)
  print("Metrics of training and validation saved into colab Files!")