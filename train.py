import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from eVAE import ExpandingVAE
import torch
import matplotlib.pyplot as plt  # Add this import
import os  # Add this import
import glob  # Add this import
import numpy as np  # Add this import at the top

def train_vae(vae, train_loader, model_type, model_id=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae.to(device)
    print(f"Using device: {device}")

    # Set float32 matrix multiplication precision for Tensor Cores
    if device == 'cuda':
        torch.set_float32_matmul_precision('medium')

    # Updated checkpoint naming scheme
    if model_type == 'traditional':
        filename = f'vae-traditional-{model_id}-{{epoch:02d}}-{{train_loss:.2f}}'
    else:  # expanding
        filename = f'vae-expanding-{model_id}-{{epoch:02d}}-{{train_loss:.2f}}'

    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename=filename,
        monitor='train_loss',
        save_top_k=1,
        mode='min'
    )
    
    # Create a custom callback to store losses
    class LossCallback(pl.Callback):
        def __init__(self):
            super().__init__()
            self.losses = []
            
        def on_train_epoch_end(self, trainer, pl_module):
            # Get the loss from the last logged batch
            loss = trainer.callback_metrics['train_loss'].item()
            self.losses.append(loss)
    
    loss_callback = LossCallback()
    
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        callbacks=[checkpoint_callback, loss_callback]
    )
    
    trainer.fit(vae, train_loader)
    return loss_callback.losses  # Return the list of losses per epoch

def main(dataset, input_dim, latent_dim, num_trad, expansion_ratios, num_epochs=50):
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    traditional_losses = []
    for i in range(num_trad):
        # Updated checkpoint path pattern
        checkpoint_pattern = f'./checkpoints/vae-traditional-{i+1}-*.ckpt'
        matching_checkpoints = glob.glob(checkpoint_pattern)

        losses_filename = f'./checkpoints/vae-traditional-{i+1}_losses.npz'
        
        if matching_checkpoints:
            # Load the latest checkpoint if multiple exist
            checkpoint_path = max(matching_checkpoints, key=os.path.getctime)
            traditional_vae = ExpandingVAE.load_from_checkpoint(checkpoint_path)
            traditional_loss_epochs = np.load(losses_filename)['losses'].tolist()  # Load losses
            traditional_losses.append(traditional_loss_epochs)
            print(f"Loaded Traditional VAE from checkpoint: {checkpoint_path}")
        else:
            # Train traditional VAE
            traditional_vae = ExpandingVAE(
                input_dim=input_dim,
                expanded_dim=input_dim,
                latent_dim=latent_dim,
                use_expansion=False
            )
            traditional_loss_epochs = train_vae(traditional_vae, train_loader, 
                                             model_type='traditional', 
                                             model_id=i+1)
            traditional_losses.append(traditional_loss_epochs)
            # Save losses to numpy file
            np.savez(losses_filename, losses=traditional_loss_epochs)

    expansion_losses = []
    for ratio in expansion_ratios:
        expanded_dim = int(input_dim * ratio)
        # Updated checkpoint path pattern
        checkpoint_pattern = f'./checkpoints/vae-expanding-{ratio}-*.ckpt'
        matching_checkpoints = glob.glob(checkpoint_pattern)

        losses_filename = f'./checkpoints/vae-expanding-{ratio}_losses.npz'
        
        if matching_checkpoints:
            # Load the latest checkpoint if multiple exist
            checkpoint_path = max(matching_checkpoints, key=os.path.getctime)
            expanding_vae = ExpandingVAE.load_from_checkpoint(checkpoint_path)
            expanding_loss_epochs = np.load(losses_filename)['losses'].tolist()  # Load losses
            expansion_losses.append(expanding_loss_epochs)
            print(f"Loaded Expanding VAE from checkpoint: {checkpoint_path}")
        else:
            # Train expanding VAE
            expanding_vae = ExpandingVAE(
                input_dim=input_dim,
                expanded_dim=expanded_dim,
                latent_dim=latent_dim,
                use_expansion=True
            )
            expanding_loss_epochs = train_vae(expanding_vae, train_loader, 
                                           model_type='expanding', 
                                           model_id=ratio)
            expansion_losses.append(expanding_loss_epochs)

            # Save losses to numpy file
            np.savez(losses_filename, losses=expanding_loss_epochs)
    
    # Save losses before plotting
    losses_data = {
        'traditional_losses': traditional_losses,
        'expansion_losses': expansion_losses,
        'expansion_ratios': expansion_ratios
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Save losses to numpy file
    np.savez('./checkpoints/training_losses.npz',
             traditional_losses=np.array(traditional_losses, dtype=object),
             expansion_losses=np.array(expansion_losses, dtype=object),
             expansion_ratios=np.array(expansion_ratios))
    
    return traditional_losses, expansion_losses
    

def plot_losses(traditional_losses, expansion_losses, expansion_ratios):
    # Plotting the loss curves
    plt.figure(figsize=(10, 5))
    
    # Plot traditional VAE losses
    for i in range(len(traditional_losses)):
        print("Plotting loss curve for traditional VAE", i)
        plt.plot(range(1, len(traditional_losses[i]) + 1), traditional_losses[i], 
                label=f'Traditional VAE {i+1}', marker='o')
    
    # Plot expanding VAE losses
    for i in range(len(expansion_losses)):
        print("Plotting loss curve for expanding VAE ratio", expansion_ratios[i])
        plt.plot(range(1, len(expansion_losses[i]) + 1), expansion_losses[i], 
                label=f'Expanding VAE Ratio {expansion_ratios[i]}', marker='o')
    
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Curves by Epoch')
    plt.legend()
    plt.grid()
    plt.show()  # Show the plot
    plt.savefig('./training_loss_curves.png')  # Save the plot



# Example usage with MNIST
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    mnist = torchvision.datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    
    expansion_ratios = [1.2, 1.5, 2.0, 5.0, 10.0]

    traditional_losses, expansion_losses = main(
        dataset=mnist,
        input_dim=784,  # 28x28
        latent_dim=32,  # Compressed latent space
        num_trad=5,
        expansion_ratios=expansion_ratios
    ) 

    plot_losses(traditional_losses, expansion_losses, expansion_ratios)