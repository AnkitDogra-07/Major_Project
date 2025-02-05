import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from typing import Tuple, List
import random
from PIL import Image

class EmbeddingNet(nn.Module):
    """Custom CNN for generating face embeddings"""
    def __init__(self, embedding_dim: int = 128):
        super(EmbeddingNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Inception-like blocks
        self.inception1 = InceptionBlock(64, 64)
        self.inception2 = InceptionBlock(256, 128)
        self.inception3 = InceptionBlock(512, 256)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, embedding_dim)
        
        # L2 normalization layer
        self.l2norm = lambda x: F.normalize(x, p=2, dim=1)
        
    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Inception blocks
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        # L2 normalization
        x = self.l2norm(x)
        
        return x

class InceptionBlock(nn.Module):
    """Simplified Inception block"""
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionBlock, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return torch.cat([b1, b2, b3], dim=1)

class TripletLoss(nn.Module):
    """Triplet loss with hard mining"""
    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pairwise_dist = torch.cdist(embeddings, embeddings)
        
        # For each anchor, find the hardest positive and negative
        mask_anchor_positive = self._get_anchor_positive_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_mask(labels)
        
        # Get hardest positive and negative distances
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive.float()).max(dim=1)[0]
        hardest_negative_dist = (pairwise_dist * mask_anchor_negative.float()).min(dim=1)[0]
        
        # Compute triplet loss
        loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        return loss.mean()
    
    def _get_anchor_positive_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get boolean mask for valid anchor-positive pairs"""
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        return labels_equal & indices_not_equal
    
    def _get_anchor_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get boolean mask for valid anchor-negative pairs"""
        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

class FaceDataset(Dataset):
    """Dataset for loading face images with identity labels"""
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

class FaceEmbedder:
    """Main class for training and using the face embedding system"""
    def __init__(self, embedding_dim: int = 128):
        self.model = EmbeddingNet(embedding_dim=embedding_dim)
        self.criterion = TripletLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def train(self, train_loader: DataLoader, num_epochs: int = 10):
        """Train the embedding network"""
        self.model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    def get_embedding(self, image: torch.Tensor) -> np.ndarray:
        """Generate embedding for a single image"""
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            embedding = self.model(image)
            return embedding.cpu().numpy()[0]
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

def create_triplet_dataloader(dataset: FaceDataset, batch_size: int = 32) -> DataLoader:
    """Create a DataLoader that yields triplets of images"""
    def collate_triplets(batch):
        images = []
        labels = []
        for image, label in batch:
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triplets)

# Example usage:
def main():
    # Initialize the embedding system
    embedder = FaceEmbedder(embedding_dim=128)
    
    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Example: Create a dummy dataset
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]  # Replace with actual paths
    labels = [0, 1]  # Replace with actual identity labels
    
    dataset = FaceDataset(image_paths, labels, transform=transform)
    train_loader = create_triplet_dataloader(dataset, batch_size=32)
    
    # Train the model
    embedder.train(train_loader, num_epochs=10)
    
    # Save the trained model
    embedder.save_model("face_embedder.pth")
    
    # Generate embedding for a single image
    sample_image = dataset[0][0]  # Get first image from dataset
    embedding = embedder.get_embedding(sample_image)
    print(f"Generated embedding shape: {embedding.shape}")

if __name__ == "__main__":
    main()