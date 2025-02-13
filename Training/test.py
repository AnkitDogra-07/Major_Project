import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from customMTCNN import CustomMTCNN
from mtcnn_train import MTCNNTrainer, config
# Initialize model and trainer
model = CustomMTCNN()
trainer = MTCNNTrainer(model, config)

# Create dataloaders using the previous dataset code
train_loaders = {
    'pnet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'pnet', 'train'),
    'rnet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'rnet', 'train'),
    'onet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'onet', 'train')
}

val_loaders = {
    'pnet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'pnet', 'val'),
    'rnet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'rnet', 'val'),
    'onet': create_dataloader(WIDERFaceDataset(...), CelebADataset(...), 'onet', 'val')
}
# Start training
trainer.train(train_loaders, val_loaders)