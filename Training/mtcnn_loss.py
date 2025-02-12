import torch
import torch.nn as nn
import torch.nn.functional as F

class MTCNNLoss(nn.Module):

    def __init__(self, det_weight: float = 1.0, 
                 box_weight: float = 0.5, 
                 landmark_weight: float = 0.5):

        super(MTCNNLoss, self).__init__()
        self.det_weight = det_weight
        self.box_weight = box_weight
        self.landmark_weight = landmark_weight
    
    def forward(self, 
                det_pred: torch.Tensor, 
                box_pred: torch.Tensor, 
                landmark_pred: torch.Tensor,
                det_target: torch.Tensor, 
                box_target: torch.Tensor, 
                landmark_target: torch.Tensor) -> dict:

        # Detection loss (cross-entropy)
        det_loss = F.cross_entropy(det_pred, det_target) * self.det_weight
        
        # Bounding box regression loss (smooth L1)
        # Only compute for positive samples (where det_target == 1)
        pos_indices = (det_target == 1).float()
        box_loss = F.smooth_l1_loss(
            box_pred * pos_indices, 
            box_target * pos_indices, 
            reduction='sum'
        ) / (pos_indices.sum() + 1e-6) * self.box_weight
        
        # Landmark localization loss (smooth L1)
        # Only compute for samples with valid landmark annotations
        landmark_mask = (landmark_target != -1).float()
        landmark_loss = F.smooth_l1_loss(
            landmark_pred * landmark_mask, 
            landmark_target * landmark_mask, 
            reduction='sum'
        ) / (landmark_mask.sum() + 1e-6) * self.landmark_weight
        
        # Compute total loss
        total_loss = det_loss + box_loss + landmark_loss
        
        return {
            'det_loss': det_loss,
            'box_loss': box_loss,
            'landmark_loss': landmark_loss,
            'total': total_loss
        }