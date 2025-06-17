from tqdm import tqdm
import torch

def one_epoch(dataloader, model, optimizer, loss_fn, device, validation = False):

    # Set the modality of model depending if it is in tranining or validation mode
    if validation:
        model.eval()
    else:
      model.train()

    epoch_loss = 0
    a_hat_list = []
    a_pred_list = []

    # Set the gradient modality (if in validation mode, it do not compute the gradient)
    grad_modality = torch.no_grad() if validation else torch.enable_grad()

    with grad_modality:
      for batch in tqdm(dataloader, desc="Validation Batches" if validation else "Training Batches"):

          inputs = batch['observations'][:, :-1].float().to(device)
          a_pred = model(inputs)
          a_pred_list.append(a_pred)

          a_hat = batch["actions"].to(device)
          a_hat_list.append(a_hat)

          loss = loss_fn(a_pred, a_hat)

          if not validation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          epoch_loss += loss.item()
      epoch_loss_avg = epoch_loss / len(dataloader)
    return epoch_loss_avg, a_hat_list, a_pred_list