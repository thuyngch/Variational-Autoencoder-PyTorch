# Variational-Autoencoder-PyTorch

This repository is to implement Variational Autoencoder and Conditional Autoencoder.


## Variational Autoencoder (VAE)
Variational Autoencoder is a specific type of Autoencoder. In which, the hidden representation (encoded vector) is forced to be a Normal distribution. As the result, by randomly sampling a vector in the Normal distribution, we can generate a new sample, which has the same distribution with the input (of the encoder of the VAE), in other word, the generated sample is realistic.

There are some results of VAE below:

<p align="center">
  <img src="https://github.com/AntiAegis/Variational-Autoencoder-PyTorch/blob/master/results/vae/recons0.png" width="700" alt="accessibility text">
</p>

<p align="center">
  <img src="https://github.com/AntiAegis/Variational-Autoencoder-PyTorch/blob/master/results/vae/recons5.png" width="700" alt="accessibility text">
</p>

<p align="center">
  <img src="https://github.com/AntiAegis/Variational-Autoencoder-PyTorch/blob/master/results/vae/recons9.png" width="700" alt="accessibility text">
</p>

<p align="center">
  <img src="https://github.com/AntiAegis/Variational-Autoencoder-PyTorch/blob/master/results/vae/generate.png" width="900" alt="accessibility text">
</p>

## Conditional Variational Autoencoder (CVAE)
