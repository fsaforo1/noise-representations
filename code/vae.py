import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, batch_size, noise_size=None):
        super(VAE, self).__init__()
        self.batch_size = batch_size
        self.enc_layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),  #
                                        nn.BatchNorm2d(32),
                                        nn.ReLU())
        self.enc_layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  #
                                        nn.BatchNorm2d(64),
                                        nn.ReLU())
        self.enc_layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  #
                                        nn.BatchNorm2d(128),
                                        nn.ReLU())
        self.enc_fc1 = nn.Sequential(nn.Linear(128 * 4 * 4, 256))
        self.enc_fc2 = nn.Sequential(nn.Linear(128 * 4 * 4, 256))

        self.dec_fc = nn.Sequential(nn.Linear(256, 128 * 4 * 4),
                                    nn.ReLU())
        self.dec_layer1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.ReLU())
        self.dec_layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.ReLU())
        self.dec_layer3 = nn.Sequential(nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1, stride=2, output_padding=1),
                                        nn.ReLU())
        if noise_size is not None:
            self.noise = True
            self.noise_size = noise_size
        else:
            self.noise = False

    def encode(self, x):
        h1 = self.enc_layer1(x)
        h1 = self.enc_layer2(h1)
        h1 = self.enc_layer3(h1)
        h1 = h1.view(-1, 128 * 4 * 4)
        return self.enc_fc1(h1), self.enc_fc2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = self.dec_fc(z)
        h3 = h3.view(-1, 128, 4, 4)
        h3 = self.dec_layer1(h3)
        h3 = self.dec_layer2(h3)
        h3 = self.dec_layer3(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        yard_stick = max(z.max(), abs(z.min()))
        if self.noise and self.training:
            noise = torch.rand(self.batch_size, 256)
            if torch.cuda.is_available():
                noise = noise.cuda()
            noise = (noise * 2 - 1.0) * yard_stick * self.noise_size
            z = z + noise
        return self.decode(z), mu, logvar, z
