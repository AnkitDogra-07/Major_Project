import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

class FacialEmbeddingSystem:
    def __init__(self):
        # Initialize face detection model
        self.mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize facial embedding model
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
            
    def get_embedding(self, frame):
        # Convert BGR (OpenCV format) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image3
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect face and get bounding boxes
        boxes, _ = self.mtcnn.detect(pil_image)
        
        if boxes is None:
            return None, None
        
        # Get facial embedding for the first detected face
        face = self.mtcnn(pil_image)
        
        if face is None:
            return None, None
            
        # If multiple faces were detected, we'll just use the first one
        if isinstance(face, list):
            face = face[0]
        
        # Handle the case where face is a 5D tensor
        if len(face.shape) == 5:
            face = face.squeeze(0)  # Remove the extra dimension
            
        # Add batch dimension if necessary
        if len(face.shape) == 3:
            face = face.unsqueeze(0)
            
        # Debug print
        print(f"Face tensor shape before embedding: {face.shape}")
        
        if torch.cuda.is_available():
            face = face.cuda()
            
        with torch.no_grad():
            embedding = self.resnet(face)
            
        return embedding.cpu().numpy()[0], boxes[0]

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize embedding system
    embedding_system = FacialEmbeddingSystem()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get embedding and bounding box
        embedding, box = embedding_system.get_embedding(frame)
        
        if embedding is not None and box is not None:
            # Draw bounding box around detected face
            box = box.astype(int)
            cv2.rectangle(frame, 
                         (box[0], box[1]), 
                         (box[2], box[3]), 
                         (0, 255, 0), 
                         2)
            
            # Display embedding dimensionality
            cv2.putText(frame, 
                       f"Embedding dim: {embedding.shape}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
        
        # Display the frame
        cv2.imshow('Facial Embedding Demo', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()