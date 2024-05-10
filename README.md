# autoencoder-walks

A visualization of walking around the latent space of an Autoencoder trained on MNIST.  

Contributors:
- Milo Chang
- Jamie Tang
- Simon Socolow
  
For example, you can see what the model thinks is in between an 8 and a 9:  ![demo pic](https://raw.githubusercontent.com/ssocolow/autoencoder-walks/main/demo.png)  

## Quickstart
`python3 UI.py` will start the demo, which loads the pre-trained model from `newmodel.pt` and ten digits from `firstTenDigits.pt`.  
The code for the model can be found in `model.py`, and uses an architecture inspired by [this notebook](https://www.eecs.qmul.ac.uk/~sgg/_ECS795P_/papers/WK07-8_PyTorch_Tutorial2.html).  <br><br>Notes:
- If you start at a 1 and go towards a 3, it won't look very good. But if you take a few steps towards a 4, then go to a 3, you can get a much better 3.

## Colab Notebook
Check out [our notebook](https://github.com/ssocolow/autoencoder-walks/blob/main/autoencoder.ipynb) to learn more about Autoencoders and a hands-on denoising application!
![colab notebook pic](https://raw.githubusercontent.com/ssocolow/autoencoder-walks/main/autoencodercollab.png)
