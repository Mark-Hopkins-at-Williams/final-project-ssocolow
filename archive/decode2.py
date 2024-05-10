from simple_autoencoders import *


# class AutoEncoder:
#     def __init__(self, filename):
#         self.model = NeuralNetwork()
#         self.model.load_state_dict(torch.load(filename))
    
#     def decode(self, latent_space_vector):
#         result = torch.matmul(self.model.theta3, latent_space_vector)
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta4, result)
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta_final, result)
#         return result
    
#     def encode(self, image):
#         result = torch.matmul(self.model.theta1, image.t())
#         result = activation_function(result)
#         result = torch.matmul(self.model.theta2, result)
#         result = activation_function(result)

#         result = torch.matmul(self.model.choke, result)
#         result = activation_function(result)

#         return result


# class Autoencoder(nn.Module):

#     def __init__(self):
#         super(Autoencoder,self).__init__()
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 6, kernel_size=5),
#             nn.ReLU(True),
#             nn.Conv2d(6,16,kernel_size=5),
#             nn.ReLU(True))

#         self.decoder = nn.Sequential(             
#             nn.ConvTranspose2d(16,6,kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(6,1,kernel_size=5),
#             nn.ReLU(True),
#             nn.Sigmoid())

#     def forward(self,x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)
num_input = 784  # MNIST data input (img shape: 28*28)


# Building the encoder
class Autoencoder(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2):
        super(Autoencoder, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        # decoder part
        self.fc3 = nn.Linear(h_dim2, h_dim1)
        self.fc4 = nn.Linear(h_dim1, x_dim)

    def encoder(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def decoder(self, x):
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# When initialzing, it will run __init__() function as above

def getModelOutputTensor(image):
    # encode it
    encoding = auto.encoder(image)
    #print(f"Encoding: {encoding}")

    # now decode it
    out = auto.decoder(encoding)
    return out

def prepareToShow(image):
    ret = image.detach().numpy()
    return ret

def returnStepTowards(fromImageEnc, toImageEnc, scale):
    """ Return the scaled vector difference of two encodings"""
    diff = (toImageEnc - fromImageEnc) * scale
    return diff

def getFirstImage():
    digit = torch.randint(10,(1,1)).item()
    return firstTen[digit]

def takeStepFrom(currentImage, targetNum, scale):
    """Returns the image generated from taking a scaled step from current image to a target image"""
    # global currentDigit

    currentEnc = auto.encoder(currentImage)
    targetEnc = auto.encoder(firstTen[targetNum])

    diff = targetEnc - currentEnc
    betweenEnc = currentEnc + scale * diff

    ret = auto.decoder(betweenEnc)
    
    return ret


# if __name__ == "__main__":
# load the model and pass it into encoder
# auto = Autoencoder()
auto = Autoencoder(num_input, num_hidden_1, num_hidden_2)
auto.load_state_dict(torch.load("./archive/rando3.pt"))
auto.eval()

# # save current state of (what digit, how far we are to next image (0-1)) in tuple
# currentDigit = 0
# howFar = 0

# load in the first ten images
firstTen = torch.load("firstTenDigits.pt")

def view(image):
    show(prepareToShow(image))

if __name__ == "__main__":
    # now start
    currentImage = getFirstImage()
    view(currentImage)

    # take a step towards a 5
    for i in range(10):
        currentImage = takeStepFrom(currentImage, 5, 0.25)
        view(currentImage)
    # view(takeStepFrom(firstImage, 5, 0.6))

    for i in range(10):
        currentImage = takeStepFrom(currentImage, 4, 0.25)
        view(currentImage)
    input()
    # # look at the first image in the dataset, a 5
    # test_image, label = train_set[2]
    # print(f"Label: {label}")
    # show(test_image)

    # out = getModelOutput(test_image)
    # show(out)

    # # now look at first image
    # first, l = train_set[0]
    # show(first)

    # out1= getModelOutput(first)
    # show(out1)

    # between = getModelOutputTensor(test_image) + returnStepTowards(getModelOutputTensor(test_image), getModelOutputTensor(first), 0.5)
    # show(getModelOutput(between))

    # print(test_set[0])