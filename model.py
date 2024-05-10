# Train autoencoder and save trained model to disk
# inspired from https://www.eecs.qmul.ac.uk/~sgg/_ECS795P_/papers/WK07-8_PyTorch_Tutorial2.html

# load libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# create the model
class Autoencoder(nn.Module):
    
    def __init__(self, start_dim, hidden_1, hidden_2, activation):
        super().__init__()
        self.l1 = nn.Linear(start_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, hidden_1)
        self.l4 = nn.Linear(hidden_1, start_dim)
        self.activation = activation

    def encoder(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        return x
    
    def decoder(self, x):
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# model hyperparams
start_dim = 784
hidden_1 = 256
hidden_2 = 64
activation = torch.sigmoid


if __name__ == "__main__":
    # load MNIST data
    trainset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    testset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    # make dataloaders
    batch_size = 128
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)

    # create the model
    model = Autoencoder(start_dim, hidden_1, hidden_2, activation)

    # check if using CUDA, and if we are: tell user and send model to gpu
    cuda = torch.cuda.is_available()
    if cuda:
        print("Using CUDA")
        model.cuda()
    else:
        print("Not using CUDA")
    
    # define loss and optimizer
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # test model accuracy on data it didn't train on
    def eval(net, loader, lfunc):
        net.eval()
        totalLoss = 0
        num = 0
        for data, l in tqdm(loader):
            data = data.cuda() if cuda else data
            data = data.reshape(-1,784)
            out = net(data)
            totalLoss += lfunc(out, data)
            num += 1
        return totalLoss / num

    # train the model
    epochs = 100
    for i in range(epochs):
        for data, l in tqdm(trainloader):
            # if we can use CUDA, send data to gpu
            data = data.cuda() if cuda else data

            # put in model's expected dimension format
            inputs = torch.reshape(data, (-1, 784))

            # set gradient to zero
            optimizer.zero_grad()

            # get model output
            outputs = model(inputs)

            # generate loss from comparing output to input
            loss = loss_function(outputs, inputs)

            # calculate gradient for each param
            loss.backward()

            # update the weights based on the gradient and the optimzer
            optimizer.step()
            
        # print avg test loss
        print(f"Epoch {i} average test loss: {eval(model, testloader, loss_function)}")
    
    # now save the trained model to disk
    torch.save(model.state_dict(), "newmodel.pt")