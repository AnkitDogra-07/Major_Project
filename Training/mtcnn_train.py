import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from typing import Dict, List, Tuple
import os
import sys
from tqdm import tqdm
import logging
import wandb # For experiment tracking
from customMTCNN import CustomMTCNN

class MTCNNTrainer:
    def __init__(self, model: CustomMTCNN, config: Dict, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Initialize optimizers
        self.optimizers = {
            'pnet': optim.Adam(model.pnet.parameters(), lr=config['learning_rate']),
            'rnet': optim.Adam(model.rnet.parameters(), lr=config['learning_rate']),
            'onet': optim.Adam(model.onet.parameters(), lr=config['learning_rate'])
        }
        
        # Initialize learning rate schedulers
        self.schedulers = {
            name: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=config['lr_patience'],
                factor=0.1
            )
            for name, optimizer in self.optimizers.items()
        }
        
        # Initialize loss function
        self.criterion = MTCNNLoss(
            det_weight=config['det_weight'],
            box_weight=config['box_weight'],
            landmark_weight=config['landmark_weight']
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('MTCNNTrainer')
        
    def train_step(self, net_type: str, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass based on network type
        if net_type == 'pnet':
            det_pred, box_pred, landmark_pred = self.model.pnet(batch['image'])
        elif net_type == 'rnet':
            det_pred, box_pred, landmark_pred = self.model.rnet(batch['image'])
        else:  # onet
            det_pred, box_pred, landmark_pred = self.model.onet(batch['image'])
        
        # Compute losses
        losses = self.criterion(
            det_pred, box_pred, landmark_pred,
            batch['det_target'], batch['box_target'], batch['landmark_target']
        )
        
        # Optimization step
        self.optimizers[net_type].zero_grad()
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.state_dict()[net_type].parameters(),
            self.config['max_grad_norm']
        )
        
        self.optimizers[net_type].step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def validate(self, net_type: str, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if net_type == 'pnet':
                    det_pred, box_pred, landmark_pred = self.model.pnet(batch['image'])
                elif net_type == 'rnet':
                    det_pred, box_pred, landmark_pred = self.model.rnet(batch['image'])
                else:  # onet
                    det_pred, box_pred, landmark_pred = self.model.onet(batch['image'])
                
                losses = self.criterion(
                    det_pred, box_pred, landmark_pred,
                    batch['det_target'], batch['box_target'], batch['landmark_target']
                )
                val_losses.append({k: v.item() for k, v in losses.items()})
        
        # Compute average validation losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = sum(loss[key] for loss in val_losses) / len(val_losses)
        
        return avg_losses
    
    def train_network(self, net_type: str, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int) -> None:
        """Train a single network (P-Net, R-Net, or O-Net)"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
                for batch in pbar:
                    step_losses = self.train_step(net_type, batch)
                    train_losses.append(step_losses)
                    pbar.set_postfix(loss=step_losses['total'])
            
            # Compute average training losses
            avg_train_losses = {}
            for key in train_losses[0].keys():
                avg_train_losses[key] = sum(loss[key] for loss in train_losses) / len(train_losses)
            
            # Validation phase
            val_losses = self.validate(net_type, val_loader)
            
            # Learning rate scheduling
            self.schedulers[net_type].step(val_losses['total'])
            
            # Early stopping check
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                self.save_checkpoint(net_type, epoch, val_losses['total'])
            else:
                patience_counter += 1
            
            # Log metrics
            metrics = {
                f'{net_type}/train_{k}': v for k, v in avg_train_losses.items()
            }
            metrics.update({
                f'{net_type}/val_{k}': v for k, v in val_losses.items()
            })
            wandb.log(metrics)
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f'Early stopping triggered for {net_type}')
                break
    
    def train(self, train_loaders: Dict[str, DataLoader], val_loaders: Dict[str, DataLoader]) -> None:
        """Complete training procedure for all networks"""
        # Initialize wandb
        wandb.init(
            project="mtcnn-training",
            config=self.config
        )
        
        # Sequential training of networks
        for net_type in ['pnet', 'rnet', 'onet']:
            self.logger.info(f'Training {net_type}...')
            self.train_network(
                net_type,
                train_loaders[net_type],
                val_loaders[net_type],
                self.config['num_epochs'][net_type]
            )
        
        wandb.finish()
    
    def save_checkpoint(self,net_type: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizers[net_type].state_dict(),
            'val_loss': val_loss,
        }
        
        path = os.path.join(
            self.config['checkpoint_dir'],
            f'{net_type}_checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, path)
        self.logger.info(f'Checkpoint saved: {path}')

# Training configuration
config = {
    'learning_rate': 0.001,
    'num_epochs': {
        'pnet': 30,
        'rnet': 20,
        'onet': 20
    },
    'batch_size': 64,
    'det_weight': 1.0,
    'box_weight': 0.5,
    'landmark_weight': 0.5,
    'lr_patience': 3,
    'early_stopping_patience': 5,
    'max_grad_norm': 5.0,
    'checkpoint_dir': './checkpoints'
}