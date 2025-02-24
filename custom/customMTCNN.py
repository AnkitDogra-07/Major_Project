import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Tuple, List

class PNet(nn.Module):
    """Proposal Network (P-Net) - First stage of MTCNN"""
    def __init__(self):
        super(PNet, self).__init__()

        # Feature extraction layer
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(32)

        # Detection branch
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        self.softmax4_1 = nn.Softmax(dim=1)

        # Bounding box regression branch
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        
        # Landmark localization branch
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))

        # Face classification
        det = self.conv4_1(x)
        det = self.softmax4_1(det)
        
        # Bounding box regression
        box = self.conv4_2(x)
        
        # Facial landmark localization
        landmark = self.conv4_3(x)
        
        return det, box, landmark

class RNet(nn.Module):
    """Refinement Network (R-Net) - Second stage of MTCNN"""
    def __init__(self):
        super(RNet, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(64)

        # Fully connected layers
        self.fc = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU(128)

        #Detection
        self.fc_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)

        # Bounding box regression
        self.fc_2 = nn.Linear(128, 4)
        
        # Landmark regression
        self.fc_3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = self.prelu4(self.fc(x))

        det = self.fc_1(x)
        det = self.softmax5_1(det)
        
        box = self.fc_2(x)
        landmark = self.fc_3(x)
        
        return det, box, landmark

class ONet(nn.Module):
    """Output Network (O-Net) - Third stage of MTCNN"""
    def __init__(self):
        super(ONet, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.prelu4 = nn.PReLU(128)

        # Fully connected layer
        self.fc = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU(256)

        # Detection
        self.fc_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)

        # Bounding box regression
        self.fc_2 = nn.Linear(256, 4)

        # Landmark regression
        self.fc_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))

        x = x.view(x.shape(0), -1)
        x = self.prelu5(self.fc(x))

        det = self.fc_1(x)
        det = self.softmax6_1(det)

        box = self.fc_2(x)
        landmark = self.fc_3(x)

        return det, box, landmark

class CustomMTCNN(nn.Module):
    """Custom MTCNN implementation"""
    def __init__(self):
        super(CustomMTCNN, self).__init__()

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

        self.min_face_size = 20.0
        self.scale_factor = 0.709
        self.thresholds = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net thresholds
    
    def create_image_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Create image pyramid for multi-scale detection"""
        height, width = img.shape[2:]
        min_length = min(height, width)
        
        scales = []
        m = 12.0 / self.min_face_size
        min_length *= m
        factor_count = 0
        
        while min_length > 12.0:
            scales.append(m * self.scale_factor ** factor_count)
            min_length *= self.scale_factor
            factor_count += 1
        return [F.interpolate(img, scale_factor=s, mode='bilinear', align_corners=True)for s in scales]
    
    def generate_bounding_boxes(self, det: torch.Tensor, box: torch.Tensor, scale: float, threshold: float) -> np.ndarray:
        """Generate bounding boxes from network outputs"""
        det = det.data.cpu().numpy()
        box = box.data.cpu().numpy()

        stride = 2
        cell_size = 12

        indices = np.where(det[:, 1] > threshold)

        if len(indices[0]) == 0:
            return np.array([])
        
        boxes = np.vstack([
            np.round((stride * indices[1] + 1) / scale),
            np.round((stride * indices[0] + 1) / scale),
            np.round((stride * indices[1] + cell_size) / scale),
            np.round((stride * indices[0] + cell_size) / scale),
            det[indices[0], indices[1]][:,np.newaxis],
            box[indices]
        ]).T
        
        return box

    def nms(self, boxes: np.ndarray, threshold: float, method: str = 'union') -> List[int]:
        """Non-maximum suppression"""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            if method == 'min':
                overlap = inter / np.minimum(area[i], area[indices[1:]])
            else:
                overlap = inter / (area[i] + area[indices[1:]] - inter)

            indices = indices[1:][overlap <= threshold]

        return keep

    def detect(self, img: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Detect faces in the input image"""
        # Stage 1: P-Net
        total_boxes = []
        img_pyramid = self.create_image_pyramid(img)

        for scaled_img in img_pyramid:
            det, box, _ = self.pnet(scaled_img)
            boxes = self.generate_bounding_boxes(det, box, 
                                              scaled_img.shape[2]/img.shape[2],
                                              self.thresholds[0])
            if len(boxes) > 0:
                total_boxes.extend(boxes)

        if len(total_boxes) == 0:
            return np.array([]), np.array([])

        total_boxes = np.vstack(total_boxes)
        keep = self.nms(total_boxes, 0.7)
        total_boxes = total_boxes[keep]

        # Stage 2: R-Net
        img_boxes = [self.extract_face(img, box) for box in total_boxes]
        img_boxes = torch.stack(img_boxes)

        det, box, _ = self.rnet(img_boxes)
        det = det.data.cpu().numpy()
        box = box.data.cpu().numpy()

        keep = np.where(det[:, 1] > self.thresholds[1])[0]
        total_boxes = total_boxes[keep]
        total_boxes[:, 4] = det[keep, 1]
        total_boxes[:, 5:] = box[keep]

        keep = self.nms(total_boxes, 0.7)
        total_boxes = total_boxes[keep]

        # Stage 3: O-Net
        img_boxes = [self.extract_face(img, box) for box in total_boxes]
        img_boxes = torch.stack(img_boxes)

        det, box, landmark = self.onet(img_boxes)
        det = det.data.cpu().numpy()
        box = box.data.cpu().numpy()
        landmark = landmark.data.cpu().numpy()

        keep = np.where(det[:, 1] > self.thresholds[2])[0]
        total_boxes = total_boxes[keep]
        total_boxes[:, 4] = det[keep, 1]
        total_boxes[:, 5:] = box[keep]
        landmarks = landmark[keep]

        return total_boxes, landmarks

    def extract_face(self, img: torch.Tensor, box: np.ndarray) -> torch.Tensor:
        """Extract face region from image based on bounding box"""
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        face = img[:, :, y1:y2, x1:x2]
        return F.interpolate(face, size=(24, 24), mode='bilinear', align_corners=True)

def load_model(model_path: str) -> CustomMTCNN:
    """Load trained model"""
    model = CustomMTCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model