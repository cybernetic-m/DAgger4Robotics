import os
import sys

# Get the absolute paths of the directories containing the utils functions 
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))

# Add these directories to sys.path
sys.path.append(utils_path)

# Import section
from calculate_metrics import calculate_metrics


import torch
import random
import math
import numpy as np
import json
from tqdm import tqdm

class DAgger():
    def __init__(self, env, validationDataset, studentPolicy, expertPolicy, optimizer, loss_fn, batch_size, num_epochs, betaMode, device, rollouts_per_iteration, exponential_beta_k = 0):

        self.env = env
        self.validationDataset = validationDataset # IT SHOULD BE A MINARI DATASET (it is use to validate the students)
        self.studentPolicy = studentPolicy
        self.expertPolicy = expertPolicy
        self.device = device
        self.rollouts_per_iteration=rollouts_per_iteration
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # This is the modality of beta coefficient used to randomly choose between Expert or Student Policy
        # [linear, inverse, exponential]
        self.betaMode = betaMode
        if self.betaMode not in ['linear', 'inverse', 'exponential']:
          raise ValueError("betaMode must be one of ['linear', 'inverse', 'exponential']")
        if self.betaMode == 'exponential':
          self.k = exponential_beta_k
        # If True, initialize the dataset as void, if False as the Training Minari Dataset

        # Lists of x (states) and a (actions) collected in Dataset aggregation used for training
        self.x = []
        self.a = []

        # Lists of x (states) and a (actions) from the validation Dataset used for validation of students
        self.x_val = []
        self.a_val = []

        # Initialize a dict to save the beta values and the counter of students
        self.other_param_dict = {
            'beta_list': [],
            'student_Nchoice': [],
            'expert_Nchoice': []
        }

        # Initialize the validation dataset with Minari episodes
        self.initDataset()

    def initDataset(self):

        try:
            val_episodes = self.validationDataset.sample_episodes(n_episodes=len(self.validationDataset))
            # Iterate over all the episodes and divide in (state,actions) to save into the lists
            for ep in val_episodes:
              self.x_val.extend(ep.observations[:-1])
              self.a_val.extend(ep.actions[:])
            print(f"Validation Dataset correctly loaded.\nObs Space Dim: {len(self.x_val)}, Action Space Dim: {len(self.a_val)}")
        except Exception as e:
            print(f"Error: {e}")

    def beta_fn(self, n_iterations):
        if self.betaMode == 'linear':
          return lambda i: max(0, 1-(1/n_iterations)*i)
        elif self.betaMode == 'inverse':
          return lambda i: 1.0/(i+1)
        elif self.betaMode == 'exponential':
          return lambda i: math.exp(-self.k * i)

    def rollout_and_aggregate(self, beta, previousStudentPolicyPath):
        print("Rollout... ")
        exp_count = 0   # Initialize a counter to see how much time the expert is selected
        stud_count = 0  # Initialize a counter to see how much time the student is selected
        observation, _ = self.env.reset()
        done = False
        if os.path.exists(previousStudentPolicyPath):
          self.studentPolicy.load_state_dict(torch.load(previousStudentPolicyPath,map_location=self.device))
        with torch.no_grad():
          for ep in range(self.rollouts_per_iteration):
            observation, _ = self.env.reset()
            done = False
            while not done:
                # Convert observation to a Tensor
                observationTensor = torch.tensor(observation, dtype=torch.float32, device=self.device)

                # Get action from either student or expert policy depending on beta
                # random.random() give a number between [0,1]
                # iter 0: beta = 1 => then with 100% of probability it choose the Expert
                # iter 1: beta = 0.8 => then with 80% of probability it choose the Expert
                if random.random() < beta:
                    exp_count += 1
                    action = self.expertPolicy(observationTensor)
                else:
                    stud_count += 1
                    action = self.studentPolicy(observationTensor)

                # Needed because we always save the expert actions in the dataset
                expertAction = self.expertPolicy(observationTensor)

                # Collecting both observation and action appending to respective lists
                self.x.append(observationTensor.detach().cpu().numpy())
                self.a.append(expertAction.detach().cpu().numpy())

                observation, reward, terminated, truncated, _ = self.env.step(action.detach().cpu().numpy())
                done = terminated or truncated

        self.other_param_dict['student_Nchoice'].append(stud_count)
        self.other_param_dict['expert_Nchoice'].append(exp_count)
        print(f"The Expert was selected {exp_count} times, while the student was selected {stud_count} times")
        print("Collected data - Observations:", len(self.x), "Actions:", len(self.a))


    def trainStudentPolicy(self, iter):
        print("Training Student Policy...")
        dataset_size = len(self.x) # Take the size of the augmented dataset
        indices = np.arange(dataset_size) # Create a list of indices

        # List of predictions and ground truth accumulating during the single epoch, used to compute metrics per epoch
        # We accumulate also the losses for all the batches to do the average for epoch loss
        a_pred_list = []
        a_list = []
        epoch_loss_list = []

        train_metrics = {
              'rmse': [],
              'mae': [],
              'r2': [],
              'loss': [],
        }

        for epoch_idx in range(self.num_epochs):
          epoch_avg_loss = 0.0

          print(f"EPOCH {epoch_idx}:")
          np.random.shuffle(indices) # Firstly randomly shuffle the entire indices numpy array

          # Iterating over batches to updates the Student Policy Network
          for batch_idx in tqdm(range(0, dataset_size, self.batch_size), desc='Elements...') :
            batch_indices = indices[batch_idx:batch_idx + self.batch_size] # Take the indices from i to i+batch_size (Ex. at iter 0 => 0 - 32)
            # Create two tensors for observation and action, taking the respective batch_indices
            x_batchTensor = torch.tensor(np.array([self.x[idx] for idx in batch_indices ]), dtype=torch.float32).to(self.device)
            a_batchTensor = torch.tensor(np.array([self.a[idx] for idx in batch_indices ]), dtype=torch.float32).to(self.device)

            # Make the predictions using the Student Policy
            a_pred_batchTensor = self.studentPolicy(x_batchTensor)
            # Computing the loss for batch
            batch_loss = self.loss_fn(a_pred_batchTensor, a_batchTensor)

            # Transform the tensors of action ground truth and prediction into numpy and append into the lists
            a_pred_batch = a_pred_batchTensor.detach().cpu().numpy()
            a_batch = a_batchTensor.detach().cpu().numpy()

            # Append the values to the respective lists and compute metrics
            # At this moment the lists are formed by different np.array => [np.array([[a1,a2], [a3,a4], ...]), np.array([[a1,a2], [a3,a4], ...])]
            a_pred_list.append(a_pred_batch)
            a_list.append(a_batch)
            epoch_loss_list.append(batch_loss.item())

            # Zeroing the gradient, compute gradient for step and update weights
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

          # Compute the average loss
          epoch_avg_loss = sum(epoch_loss_list) / len(epoch_loss_list)

          # Flatten the list of arrays into a single 2D array
          # From this stage the list of np.array becoming only one np.array concatenating rows
          # [ [a1,a2],
          #   [a3,a4],
          # ]
          y_true = np.concatenate(a_list, axis=0)
          y_pred = np.concatenate(a_pred_list, axis=0)

          # Compute the metrics
          train_rmse, train_mae, train_r2 = calculate_metrics(y_true_list = y_true, y_pred_list= y_pred, metrics = train_metrics)
          train_metrics['loss'].append(epoch_avg_loss)
          print(f"TRAIN\t Loss: {epoch_avg_loss:.8f}, RMSE: {train_rmse:.8f}, MAE: {train_mae:.8f}, R2: {train_r2:.8f}")

          # Save the last epoch model
          if epoch_idx == self.num_epochs - 1:
            torch.save(self.studentPolicy.state_dict(), f"student_policy_{iter}.pt")
            print(f"Model saved as student_policy_{iter}.pt")

        # Save the metrics in json of the i student
        with open(f"train_metrics_student_{iter}.json", "w") as f:
          json.dump(train_metrics, f, indent=4)

    def evaluateStudents(self, model_path, iter):
        dataset_size = len(self.x_val) # Take the size of the augmented dataset
        indices = np.arange(dataset_size) # Create a list of indices

        # Load the Student Policy weights and put it in evaluation mode
        self.studentPolicy.load_state_dict(torch.load(model_path,map_location=self.device))
        self.studentPolicy.eval()

        # List of predictions and ground truth accumulating during the single epoch, used to compute metrics per epoch
        # We accumulate also the losses for all the batches to do the average for epoch loss
        a_pred_list = []
        a_list = []
        epoch_loss_list = []

        val_metrics = {
              'rmse': [],
              'mae': [],
              'r2': [],
              'loss': [],
        }

        np.random.shuffle(indices) # Firstly randomly shuffle the entire indices numpy array

        # Iterating over batches to updates the Student Policy Network
        for batch_idx in range(0, dataset_size, self.batch_size):
          batch_indices = indices[batch_idx:batch_idx + self.batch_size] # Take the indices from i to i+batch_size (Ex. at iter 0 => 0 - 32)
          # Create two tensors for observation and action, taking the respective batch_indices
          x_batchTensor = torch.tensor(np.array([self.x_val[idx] for idx in batch_indices ]), dtype=torch.float32).to(self.device)
          a_batchTensor = torch.tensor(np.array([self.a_val[idx] for idx in batch_indices ]), dtype=torch.float32).to(self.device)

          # Make the predictions using the Student Policy
          a_pred_batchTensor = self.studentPolicy(x_batchTensor)
          # Computing the loss for batch
          batch_loss = self.loss_fn(a_pred_batchTensor, a_batchTensor)

          # Transform the tensors of action ground truth and prediction into numpy and append into the lists
          a_pred_batch = a_pred_batchTensor.detach().cpu().numpy()
          a_batch = a_batchTensor.detach().cpu().numpy()

          # Append the values to the respective lists and compute metrics
          # At this moment the lists are formed by different np.array => [np.array([[a1,a2], [a3,a4], ...]), np.array([[a1,a2], [a3,a4], ...])]
          a_pred_list.append(a_pred_batch)
          a_list.append(a_batch)
          epoch_loss_list.append(batch_loss.item())

        # Compute the average loss
        val_avg_loss = sum(epoch_loss_list) / len(epoch_loss_list)

        # Flatten the list of arrays into a single 2D array
        # From this stage the list of np.array becoming only one np.array concatenating rows
        # [ [a1,a2],
        #   [a3,a4],
        # ]
        y_true = np.concatenate(a_list, axis=0)
        y_pred = np.concatenate(a_pred_list, axis=0)

        # Compute the metrics
        val_rmse, val_mae, val_r2 = calculate_metrics(y_true_list = y_true, y_pred_list= y_pred, metrics = val_metrics)
        val_metrics['loss'].append(val_avg_loss)
        print(f"{model_path}\t Loss: {val_avg_loss:.8f}, RMSE: {val_rmse:.8f}, MAE: {val_mae:.8f}, R2: {val_r2:.8f}")

        # Save the metrics in json of the i student
        with open(f"val_metrics_student_{iter}.json", "w") as f:
          json.dump(val_metrics, f, indent=4)

        return val_avg_loss


    def run(self, n_iterations):
        print(f"Run DAgger algorithm...")
        # This return a function beta (depending on self.betaMode) that should take the iter as input
        beta = self.beta_fn(n_iterations = n_iterations)

        # Training Loop
        for iter in range(n_iterations):
          beta_i = beta(iter)
          self.other_param_dict['beta_list'].append(beta_i)
          print(f"\n--- ITERATION {iter+1}/{n_iterations} | beta = {beta_i:.3f} ---")
          previousIter = iter - 1 if iter > 0 else None
          self.rollout_and_aggregate(beta_i, f"student_policy_{previousIter}.pt")
          self.trainStudentPolicy(iter)

        # Evaluation Loop
        vloss_list = []
        print("Evaluating the best student...")
        for iter in range(n_iterations):
          vloss_i = self.evaluateStudents(f"student_policy_{iter}.pt", iter)
          vloss_list.append(vloss_i)


        with open(f"others_param.json", "w") as f:
          json.dump(self.other_param_dict, f, indent=4)

        print(f"The better student Policy is the student_policy_{np.argmin(vloss_list)}.pt")

















