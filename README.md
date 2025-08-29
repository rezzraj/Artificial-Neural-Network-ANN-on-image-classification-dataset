-----

# 🤖 Fashion MNIST Classifier with PyTorch 🤖

A deep dive into building a simple but effective Neural Network to classify clothing items from the Fashion MNIST dataset. No cap, this project is a solid introduction to the fundamentals of PyTorch, from data loading to model training and evaluation. 🚀

-----

## 📜 Table of Contents
  * [Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
  * Project Structure
  * Usage
  * Results
  * Code Highlights

-----

## About The Project ✨

This project demonstrates the end-to-end process of a machine learning classification task using **PyTorch**.

Here's the lowdown on what we're doing:

1.  **Load & Preprocess Data**: We take the `fmnist_small.csv` dataset, split it, and normalize it so our model doesn't freak out. 🧘
2.  **Custom Dataset & DataLoader**: We create a custom `Dataset` class and use `DataLoader` to efficiently feed data to our model in batches. It's like a buffet for the NN. 🍽️
3.  **Build a Neural Network**: We define a simple feed-forward neural network with a couple of hidden layers and ReLU activations.
4.  **Train the Model**: The main event\! We run a training loop for 100 epochs, using **Stochastic Gradient Descent (SGD)** to optimize our model's parameters and **Cross-Entropy Loss** to see how badly it's doing. 📉
5.  **Evaluate Performance**: After training, we unleash the model on the test set to see if it actually learned anything. The final accuracy is the big reveal. 🤯

**Tech Stack:**

  * PyTorch
  * Pandas
  * Scikit-learn
  * Matplotlib & Seaborn

-----

## Dataset 👗

We're using the **Fashion MNIST** dataset, which is basically the cooler, more stylish cousin of the original MNIST dataset. Instead of boring numbers, we have 10 classes of clothing items.

The dataset consists of 28x28 pixel grayscale images. The classes are:

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

The script is structured sequentially to make it easy to follow:

1.  **Imports & Seed**: Importing all the necessary libraries and setting a manual seed for reproducibility. Gotta make sure our "random" is the same every time. 🎲
2.  **Data Loading**: Reads `fmnist_small.csv` using Pandas.
3.  **Data Preparation**:
      * Separates features (`x`) and labels (`y`).
      * Splits the data into 80% training and 20% testing sets.
      * **Scales** the feature data by dividing pixel values by `255.0` to normalize them between 0 and 1.
4.  **PyTorch `Dataset` & `DataLoader`**:
      * A `customDataset` class is defined to handle the conversion of numpy arrays to PyTorch tensors.
      * `DataLoader` objects are created for both training and testing sets to manage batching and shuffling.
5.  **Model Definition**:
      * A `myNN` class inheriting from `nn.Module` defines the neural network architecture:
          * Input Layer: 784 features (28x28 pixels)
          * Hidden Layer 1: 128 neurons with ReLU activation
          * Hidden Layer 2: 64 neurons with ReLU activation
          * Output Layer: 10 neurons (one for each class)
6.  **Training Setup**:
      * **Hyperparameters**: `epochs` set to 100 and `learning_rate` to `0.01`.
      * **Loss Function**: `nn.CrossEntropyLoss()` is chosen (it conveniently combines Softmax and Negative Log-Likelihood Loss).
      * **Optimizer**: `optim.SGD()` is used to update the model's weights.
7.  **Training Loop**:
      * Iterates through the specified number of epochs.
      * In each epoch, it iterates through all batches from the `train_dataloader`.
      * Performs the classic five steps: forward pass, loss calculation, zeroing gradients, backpropagation, and updating weights.
8.  **Evaluation**:
      * The model is set to evaluation mode (`model.eval()`).
      * Using `torch.no_grad()` to prevent gradient calculations (we're just testing, not learning).
      * It calculates the accuracy by comparing predicted labels with the true labels on the test set.
9.  **Visualization**:
      * A plot is generated using Matplotlib to visualize the training loss over the epochs. This helps us see if the model is actually learning or just vibing. 📉

-----

## Getting Started 🏁

To get a local copy up and running, follow these simple steps.

### Prerequisites

You'll need Python 3.x and the following libraries. You can install them using pip.

```sh
pip install torch pandas scikit-learn matplotlib seaborn
```

### Installation

1.  Clone the repo (or just download the script lmao 💀):
    ```sh
    git clone https://github.com/your_username/your_repository_name.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd your_repository_name
    ```
3.  Make sure you have the `fmnist_small.csv` file in the same directory.

-----

## Usage 🎮

Just run the Python script. It's that simple. Get ready for some epic terminal output.

```sh
python your_script_name.py
```

The script will print the average loss for each epoch during training and the final test accuracy at the end. It will also display a plot of the loss curve.

-----

## The Final Boss: Results 🏆

After 100 epochs of intense training, the model achieves a respectable accuracy on the test set.

  * **Final Test Accuracy**: `86.5%` (Your result might vary slightly due to the random train/test split, but it should be in this ballpark).

### Loss Analysis Curve

The script generates the following plot, showing that the training loss consistently decreases over time, which is exactly what we want to see. It means our model is learning and not just guessing. We love a model that does its homework. 🤓

*(Note: You will need to run the script to generate and save your own loss curve image, then update the path below.)*

```
![Loss Curve](path/to/your/loss_curve.png)
```

-----

## Code Highlights 💻

Here are some of the key parts of the code that make the magic happen.

### Custom PyTorch Dataset

This class is the foundation for feeding our data into the model. It handles indexing and converts our data into the tensor format PyTorch loves.

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

Our simple but mighty neural network. The `nn.Sequential` container makes it super easy to stack layers.

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

This is where the learning happens. A classic loop that every deep learning practitioner knows and loves (or fears 💀).

```python
#training loop
for epoch in range(epochs):
    total_epoch_loss=0
    for features, labels in train_dataloader:
        #forward pass
        y_pred=model(features)
        #calculating loss
        loss=criterion(y_pred,labels)
        #resetting the grads to zero  before calculating new
        optimizer.zero_grad()
        #backpropagation (calculating grads)
        loss.backward()
        #updating parameters
        optimizer.step()

        #adding loss from all batches
        total_epoch_loss+= loss.item()
    avg_loss=total_epoch_loss/len(train_dataloader)
    print("epoch:",epoch,"avg_loss:",avg_loss)
```
