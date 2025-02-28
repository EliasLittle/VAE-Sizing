import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms

class ExpandingVAE(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        expanded_dim,
        latent_dim,
        use_expansion=True,
        learning_rate=1e-3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder layers
        if use_expansion:
            # Expansion phase
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, expanded_dim),
                nn.ReLU(),
                nn.BatchNorm1d(expanded_dim),
                nn.Linear(expanded_dim, 2 * latent_dim)
            )
        else:
            # Traditional encoder
            # Have an extra layer to ensure the input is normalized, and help preserve some param equality
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, 2 * latent_dim),
            )
            
        # Decoder layers
        if use_expansion:
            # Combined decoder for expansion and compression
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, expanded_dim),
                nn.ReLU(),
                nn.BatchNorm1d(expanded_dim),
                nn.Linear(expanded_dim, input_dim),
                nn.Sigmoid()
            )
        else:
            # Traditional decoder
            # Have an extra layer to ensure the input is normalized, and help preserve some param equality
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, input_dim),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid()
            )
            
    def encode(self, x):
        x = self.encoder(x)
        
        # Split into mu and log_var
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
    
    def training_step(self, batch, batch_idx):
        x = batch[0].view(batch[0].size(0), -1)
        x_hat, mu, log_var = self(x)
        
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        loss = recon_loss + kl_div
        
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kl_div', kl_div)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate) 