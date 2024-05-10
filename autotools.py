from model import *

# create the model and load in pre-trained weights
auto = Autoencoder(start_dim, hidden_1, hidden_2, activation)
auto.load_state_dict(torch.load("newmodel.pt"))

# load in the first ten images
firstTen = torch.load("firstTenDigits.pt")

def getModelOutputTensor(image):
    # encode it
    encoding = auto.encoder(image)
    #print(f"Encoding: {encoding}")

    # now decode it
    out = auto.decoder(encoding)
    return out

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