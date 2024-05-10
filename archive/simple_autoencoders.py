# digitize

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets
import torch.nn as nn

def activation_function(input):
    return torch.sigmoid(input)


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            # torch.nn.ReLU(),
            # torch.nn.Linear(36, 18),
            # torch.nn.ReLU(),
            # torch.nn.Linear(18, 9)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(9, 18),
            # torch.nn.ReLU(),
            # torch.nn.Linear(18, 36),
            # torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
 
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(784, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 32),
#             nn.ReLU(True)
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 784),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         enc = self.encoder(x)
#         dec = self.decoder(enc)
#         return dec

# class NeuralNetwork(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden_layer_size = 200
#         self.small_choke_size = 50

#         # size is 28*28 -> hidden_layer_size x2 -> small_choke_size -> hidden_layer_size x2 -> 28*28
#         self.theta1 = Parameter(torch.empty(self.hidden_layer_size, 28*28))
#         self.theta2 = Parameter(torch.empty(self.hidden_layer_size, self.hidden_layer_size))

#         self.choke = Parameter(torch.empty(self.small_choke_size, self.hidden_layer_size))

#         self.theta3 = Parameter(torch.empty(self.hidden_layer_size, self.small_choke_size))
#         self.theta4 = Parameter(torch.empty(self.hidden_layer_size, self.hidden_layer_size))

#         self.theta_final = Parameter(torch.empty(28*28, self.hidden_layer_size))
#         for param in self.parameters():
#             torch.nn.init.kaiming_uniform_(param)

#     def forward(self, x):
#         #print(x)
#         result = torch.matmul(self.theta1, x.t())
#         result = activation_function(result)
#         result = torch.matmul(self.theta2, result)
#         result = activation_function(result)

#         result = torch.matmul(self.choke, result)
#         result = activation_function(result)

#         result = torch.matmul(self.theta3, result)
#         result = activation_function(result)
#         result = torch.matmul(self.theta4, result)
#         result = activation_function(result)
#         result = torch.matmul(self.theta_final, result)
#         #print(f"result: {result}, shape:{result.shape}")
#         # no relu for output?
#         return result
    

# def minibatch_loss(net, X, y):
#     predictions = net.forward(X)
#     probs = torch.gather(predictions, dim=0, index=y.unsqueeze(0))
#     probs = probs.clamp(min=0.00000001, max=0.99999999)
#     losses = -torch.log(probs)
#     loss = torch.mean(losses)
#     return loss

# def minibatch_loss(net, X, y):
#     predictions = net.forward(X)
#     #print(f"predictions size: {predictions.shape}")
#     predictions = predictions.transpose(-1,-2)
#     #print(f"predictions new size: {predictions.shape}")
#     #print(f"X size: {X.shape}")
#     differences = predictions - X
#     losses = torch.square(differences)
#     loss = torch.mean(losses)
#     return loss

# def minibatch_gd(model, num_epochs, train_set, test_set, lr=0.01):
#     for _ in range(num_epochs):    
#         train_loader = DataLoader(train_set, batch_size=32)
#         for X, y in tqdm(train_loader):
#             loss = minibatch_loss(model, X, y)
#             loss.backward()
#             for param in model.parameters():
#                 with torch.no_grad():           
#                     param -= lr*param.grad
#                     param.grad = None
#         test_loader = DataLoader(test_set, batch_size=128)
#         accuracy = evaluate(model, test_loader)
#         print(f"Accuracy: {accuracy}")

# def minibatch_gd(model, num_epochs, train_set, test_set, lr=0.01):
#     for _ in range(num_epochs):
#         train_loader = DataLoader(train_set, batch_size=32)
#         for X, y in tqdm(train_loader):
#             loss = minibatch_loss(model, X, X)
#             loss.backward()
#             for param in model.parameters():
#                 with torch.no_grad():           
#                     param -= lr*param.grad
#                     param.grad = None
#         test_loader = DataLoader(test_set, batch_size=128)
#         accuracy = evaluate(model, test_loader)
#         print(f"Accuracy: {accuracy}")


# def evaluate(net, test_loader):
#     net.eval()
#     correct = 0
#     total = 0
#     for X, y in tqdm(test_loader):
#         predictions = net.forward(X)
#         preds = torch.max(predictions, 0)
#         correct += torch.sum(preds.indices == y).item()
#         total += torch.numel(y)
#     net.train()
#     return correct/total
# def evaluate(net, test_loader):
#     net.eval()
#     badness = 0
#     total = 0
#     for X, y in tqdm(test_loader):
#         predictions = net.forward(X).transpose(-1,-2)
#         #print(predictions.shape, X.shape)
#         howbad = torch.sum(torch.square(predictions - X)).item()
#         badness += howbad
#         total += torch.numel(X)
#     net.train()
#     return badness/total

def load_ten_of_each(dset):
    l = []
    n = 0
    i = 0
    while n < 10:
        image, label = dset[i]
        if label == n:
            l.append(image)
            n += 1
        i += 1
    return l

def load_mnist():
    def image_to_tensor(img):
        t = tensor(np.asarray(img)).flatten().float()
        return t / 255
    mnist_train_raw = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_test_raw = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    mnist_train = [(image_to_tensor(t[0]), t[1]) for t in mnist_train_raw]
    mnist_test = [(image_to_tensor(t[0]), t[1]) for t in mnist_test_raw]
    return mnist_train, mnist_test


class MNistClassifier:
    def __init__(self, filename):
        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
    
    def classify(self, img):
        predictions = self.model.forward(img.flatten())
        return predictions

def show(image):
    f = plt.figure()
    plt.imshow(image.reshape(28,28), cmap="gray")
    f.show()


# train_set, test_set = load_mnist()

if __name__ == "__main__":
    train_set, test_set = load_mnist()
    num_epochs = 5
    model = NeuralNetwork()
    distance = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-1,
                             weight_decay = 1e-8)
    
    for epoch in range(num_epochs):
        train_loader = DataLoader(train_set, batch_size=32)
        for data in tqdm(train_loader):
            img, _ = data
            output = model(img)
            loss = distance(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # look at accuracy
        print('epoch [{}/{}], loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
        #minibatch_gd(model, num_epochs, train_set, test_set)
    
    torch.save(model.state_dict(), "mnistmodel_old.pt")