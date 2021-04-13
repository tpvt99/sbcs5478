import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim, input_shape, name="encoder", **kwargs):
        super(Encoder, self).__init__()

        self.dense_proj = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=10240, out_features=latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.dense_proj(x)

class Decoder(nn.Module):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, output_shape, latent_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__()
        self.dense_proj = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=8*10*128), # Must adjust number here
            nn.ReLU()
        )

        self.conv_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), output_padding=(1,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        x = self.dense_proj(inputs)
        x = x.view(-1, 128, 8, 10)
        x = self.conv_proj(x)
        return x


class AutoEncoder(nn.Module):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        input_shape=(64,80,3),
        latent_dim=16,
        name="autoencoder",
        **kwargs
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, input_shape=input_shape)
        self.decoder = Decoder(output_shape=input_shape, latent_dim=latent_dim)

    def forward(self, inputs):
        z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed


