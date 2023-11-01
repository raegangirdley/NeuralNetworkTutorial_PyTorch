# Import dependencies
import torch # Import PyTorch library
from PIL import Image # Import Python Imaging Library to handle images
from torch import nn, save, load # Import specific PyTorch modules
# Adam optimizer is an optimization algorithm known for its effectiveness
# in training neural networks. Stands for Adaptive Moment Estimation
from torch.optim import Adam
from torch.utils.data import DataLoader # Handles batches of data
from torchvision import datasets # Import dataset handler
from torchvision.transforms import ToTensor # Converts images to tensors to be handled

# Get data
# Load MNIST dataset, convert images to tensors, store in 'train' variable
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# Create DataLoader to manage batches of data from 'train' dataset
# Batch size of 32
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Convolutional layer
            # 1 input channel, 32 output channels, 3x3 kernel
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            # Shows that finished because 64 = 64
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            # Flattern to 1d tensor to be analyzed
            nn.Flatten(),
            # Fully connected layer with 10 output units for prediction
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        # Forward psas through defined model
        return self.model(x)

# Creates instance of neural network
clf = ImageClassifier().to('cuda')
# Create Adam optimizer for training with learning rate of 0.001
opt = Adam(clf.parameters(), lr=1e-3)
# Defines loss function for classification
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        # Loads pre-trained model state
        clf.load_state_dict(load(f))

    img = Image.open('img_2.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))

    for epoch in range(10): 
        for batch in dataset:
            X,y = batch
            # Moves data to GPU
            X,y = X.to('cuda'), y.to('cuda')
            # Forward passes to make predictions
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpropagation and update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch}, loss is: {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        # Saves trained model state to a file (in this folder, called model_state.pt)
        save(clf.state_dict(), f) 