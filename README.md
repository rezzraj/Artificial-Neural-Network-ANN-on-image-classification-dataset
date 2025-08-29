# Fashion MNIST Classifier with PyTorch

This document provides a comprehensive overview of a neural network implementation for classifying images from the Fashion MNIST dataset using the PyTorch library. The project serves as a foundational example of the complete machine learning workflow, from data preprocessing to model evaluation.

-----

##  Table of Contents

  * [Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

  * Project Structure

  * Results

  * Code Highlights
-----

## About The Project ✨

This project demonstrates the end-to-end process of a machine learning classification task.

The key stages of the project are as follows:

1.  **Data Loading and Preprocessing**: The `fmnist_small.csv` dataset is loaded, partitioned, and normalized to prepare it for model training.
2.  **Custom Dataset and DataLoader**: A custom `Dataset` class is implemented, and `DataLoader` instances are used to efficiently supply data to the model in batches.
3.  **Neural Network Construction**: A feed-forward neural network is defined with two hidden layers utilizing the ReLU activation function.
4.  **Model Training**: A training loop is executed for 100 epochs. The model's parameters are optimized using Stochastic Gradient Descent (SGD), with Cross-Entropy Loss as the objective function.
5.  **Performance Evaluation**: Following the training phase, the model's performance is assessed on the test dataset to determine its classification accuracy.

**Technology Stack:**

  * PyTorch
  * Pandas
  * Scikit-learn
  * Matplotlib

-----

## Dataset 👗

The project utilizes the **Fashion MNIST** dataset, which comprises 70,000 grayscale images of clothing items, categorized into 10 distinct classes. Each image is 28x28 pixels.

The class labels are as follows:

| Label | Description  |
| :---: | :----------- |
|   0   | T-shirt/top  |
|   1   | Trouser      |
|   2   | Pullover     |
|   3   | Dress        |
|   4   | Coat         |
|   5   | Sandal       |
|   6   | Shirt        |
|   7   | Sneaker      |
|   8   | Bag          |
|   9   | Ankle boot   |

-----

## Project Structure 🏗️

The script is organized into a logical sequence of operations:

1.  **Initialization**: Necessary libraries are imported, and a manual seed is set for computational reproducibility.
2.  **Data Loading**: The dataset is read from `fmnist_small.csv` using the Pandas library.
3.  **Data Preparation**:
      * Features (`x`) and labels (`y`) are separated.
      * The data is split into training (80%) and testing (20%) sets.
      * Feature values are scaled to a range of [0, 1] by dividing by 255.0.
4.  **PyTorch Data Handling**:
      * A `customDataset` class is defined to convert NumPy arrays into PyTorch tensors.
      * `DataLoader` instances are created for both training and test sets to handle batching and shuffling.
5.  **Model Definition**:
      * A `myNN` class, which inherits from `nn.Module`, defines the neural network architecture:
          * **Input Layer**: 784 features
          * **Hidden Layer 1**: 128 neurons with ReLU activation
          * **Hidden Layer 2**: 64 neurons with ReLU activation
          * **Output Layer**: 10 neurons, corresponding to the number of classes
6.  **Training Configuration**:
      * **Hyperparameters**: The number of epochs is set to 100, and the learning rate is set to `0.01`.
      * **Loss Function**: `nn.CrossEntropyLoss` is selected, which combines a Softmax activation and a negative log-likelihood loss.
      * **Optimizer**: `optim.SGD` is used for updating the model's weight parameters.
7.  **Training Loop**:
      * The model iterates through the specified number of epochs.
      * Within each epoch, the model processes all batches from the `train_dataloader`.
      * The standard training steps are performed: a forward pass, loss calculation, gradient zeroing, backpropagation, and a parameter update.
8.  **Evaluation**:
      * The model is switched to evaluation mode using `model.eval()`.
      * Gradient calculations are disabled via `torch.no_grad()` to improve inference speed.
      * The model's accuracy is calculated by comparing its predictions on the test set against the true labels.
9.  **Visualization**:
      * A plot of the training loss per epoch is generated using Matplotlib to visualize the model's learning progress.

-----

## Results 🏆

After 100 epochs, the model achieves a final accuracy of **86.5%** on the test set. Note that this value may vary slightly due to the random nature of the train-test split.

### Loss Analysis Curve

The generated plot illustrates a consistent decrease in training loss over the epochs, indicating that the model is learning effectively.

*(Note: The script must be executed to generate the loss curve image. The path in the markdown below should be updated accordingly.)*

```
![Loss Curve](path/to/your/loss_curve.png)
```

-----

## Code Highlights 💻

The following sections showcase key components of the implementation.

### Custom PyTorch Dataset

This class structures the data for use with PyTorch's `DataLoader`. It handles the conversion of data to tensors and provides indexed access.

```python
class customDataset(Dataset):
    def __init__(self,features, labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, index):
        return self.features[index],self.labels[index]
```

### Neural Network Architecture

The `nn.Sequential` container is used to define a clean, feed-forward neural network architecture.

```python
class myNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, input_matrix):
        return self.network(input_matrix)
```

### The Training Loop

This is the core component where the model iteratively learns from the training data.

```python
for epoch in range(epochs):
    total_epoch_loss=0
    for features, labels in train_dataloader:
        # Forward pass
        y_pred=model(features)
        # Calculate loss
        loss=criterion(y_pred,labels)
        # Zero gradients before backpropagation
        optimizer.zero_grad()
        # Backpropagation to calculate gradients
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Accumulate loss from all batches
        total_epoch_loss+= loss.item()
    avg_loss=total_epoch_loss/len(train_dataloader)
    print("epoch:",epoch,"avg_loss:",avg_loss)
```
