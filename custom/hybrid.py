import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict
import time
import hashlib

class HybridFaceRecognition:
    def __init__(self, mtcnn_model, embedding_model, cache_size=100):
        """
        Hybrid face recognition system
        
        Args:
            mtcnn_model: Custom MTCNN model for initial detection
            embedding_model: Lightweight embedding model
            cache_size: Maximum number of cached face embeddings
        """
        self.mtcnn = mtcnn_model
        self.embedder = embedding_model
        
        # Face tracking components
        self.trackers = {}
        self.next_id = 0
        
        # Embedding cache
        self.embedding_cache = {}
        self.cache_size = cache_size
        
        # Detection parameters
        self.detection_interval = 30  # Run MTCNN every 30 frames
        self.frame_count = 0
        
    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track faces in the frame
        
        Args:
            frame: Input video frame
        
        Returns:
            List of detected face dictionaries
        """
        self.frame_count += 1
        detected_faces = []
        
        # Periodically run MTCNN for new face detection
        if self.frame_count % self.detection_interval == 0:
            # Convert frame to torch tensor for MTCNN
            frame_tensor = self._preprocess_frame(frame)
            boxes, landmarks = self.mtcnn.detect(frame_tensor)
            
            # Update trackers and generate new IDs
            self._update_trackers(frame, boxes)
        
        # Update existing trackers
        for face_id, tracker in list(self.trackers.items()):
            success, bbox = tracker.update(frame)
            
            if success:
                # Get face embedding from cache or recompute
                embedding = self._get_cached_embedding(frame, bbox)
                
                detected_faces.append({
                    'id': face_id,
                    'bbox': bbox,
                    'embedding': embedding
                })
            else:
                # Remove lost tracker
                del self.trackers[face_id]
        
        return detected_faces
    
    def _update_trackers(self, frame: np.ndarray, boxes: np.ndarray):
        """
        Update face trackers based on new detections
        
        Args:
            frame: Current video frame
            boxes: Detected bounding boxes
        """
        # Create trackers for new faces
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2-x1, y2-y1))
            
            # Generate unique ID
            self.trackers[self.next_id] = tracker
            self.next_id += 1
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for MTCNN detection
        
        Args:
            frame: Input OpenCV frame
        
        Returns:
            Preprocessed torch tensor
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to torch tensor
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor
    
    def _get_cached_embedding(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Get or compute face embedding with caching
        
        Args:
            frame: Input video frame
            bbox: Face bounding box
        
        Returns:
            Face embedding vector
        """
        # Extract face region
        x, y, w, h = [int(coord) for coord in bbox]
        face_crop = frame[y:y+h, x:x+w]
        
        # Generate unique hash for face crop
        face_hash = self._compute_face_hash(face_crop)
        
        # Check cache
        if face_hash in self.embedding_cache:
            return self.embedding_cache[face_hash]
        
        # Compute embedding
        face_tensor = self._preprocess_face(face_crop)
        embedding = self.embedder.get_embedding(face_tensor)
        
        # Update cache
        self._update_cache(face_hash, embedding)
        
        return embedding
    
    def _preprocess_face(self, face: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for embedding
        
        Args:
            face: Face image crop
        
        Returns:
            Preprocessed torch tensor
        """
        # Resize and normalize
        face_resized = cv2.resize(face, (160, 160))
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5  # Normalize
        
        return face_tensor
    
    def _compute_face_hash(self, face: np.ndarray) -> str:
        """
        Compute a stable hash for a face image
        
        Args:
            face: Face image crop
        
        Returns:
            Unique hash string
        """
        # Resize to a fixed small size for hashing
        face_small = cv2.resize(face, (64, 64))
        return hashlib.md5(face_small.tobytes()).hexdigest()
    
    def _update_cache(self, face_hash: str, embedding: np.ndarray):
        """
        Update embedding cache
        
        Args:
            face_hash: Unique face hash
            embedding: Face embedding vector
        """
        # Remove oldest entry if cache is full
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        
        self.embedding_cache[face_hash] = embedding
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            threshold: Similarity threshold
        
        Returns:
            Boolean indicating if faces match
        """
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return similarity > threshold

def main():
    # Initialize models (replace with your actual models)
    mtcnn_model = CustomMTCNN()
    embedding_model = FaceEmbedder()
    
    # Create hybrid face recognition system
    face_recognizer = HybridFaceRecognition(
        mtcnn_model, 
        embedding_model
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and track faces
        detected_faces = face_recognizer.detect_and_track(frame)
        
        # Visualize detected faces
        for face in detected_faces:
            x1, y1, w, h = [int(coord) for coord in face['bbox']]
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {face['id']}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Hybrid Face Recognition', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()