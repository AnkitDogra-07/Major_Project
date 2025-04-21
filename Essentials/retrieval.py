import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

class FaceRetrievalPipeline:
    def __init__(self, db_manager=None, embedding_dir='face_embedding/inp_embedding', device=None):
        self.embedding_dir = embedding_dir
        self.db_manager = db_manager
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Initialize models
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        self.detector = MTCNN()
        
        # Load stored embeddings (from files or database)
        self.stored_embeddings = self.load_stored_embeddings()
        print(f"Loaded {len(self.stored_embeddings)} person IDs from embeddings")
        
    def load_stored_embeddings(self):
        """Load all stored embeddings from the database or directory."""
        embeddings = {}
        
        # Try loading from database first if available
        if self.db_manager:
            try:
                # This is just a conceptual approach - db_manager would need an implementation
                # to get all embeddings from FAISS index
                return embeddings
            except Exception as e:
                print(f"Error loading embeddings from database: {e}")
        
        # Fall back to loading from files
        return self._load_from_files()
        
    def _load_from_files(self):
        """Load embeddings from files as fallback."""
        embeddings = {}
        
        # Ensure embedding directory exists
        if not os.path.exists(self.embedding_dir):
            print(f"Warning: Embedding directory {self.embedding_dir} does not exist")
            return embeddings
            
        for file in Path(self.embedding_dir).glob('*.npy'):
            # Extract person_id from filename (Person_X part)
            filename = file.stem
            parts = filename.split('_')
            
            if len(parts) >= 2 and parts[0] == "Person":
                person_id = f"Person_{parts[1]}"
                
                try:
                    embedding = np.load(str(file))
                    
                    if person_id in embeddings:
                        if not isinstance(embeddings[person_id], list):
                            embeddings[person_id] = [embeddings[person_id]]
                        embeddings[person_id].append(embedding)
                    else:
                        embeddings[person_id] = [embedding]
                except Exception as e:
                    print(f"Error loading embedding {file}: {e}")
                    
        # Convert lists of embeddings to averages for each person
        for person_id in embeddings:
            if isinstance(embeddings[person_id], list) and len(embeddings[person_id]) > 0:
                embeddings[person_id] = np.mean(embeddings[person_id], axis=0)
                
        return embeddings
    
    def get_embedding(self, face_img):
        """Generate embedding for a face image."""
        try:
            if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                print("Error: Empty face image")
                return None
                
            face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                embedding = self.facenet(face).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
    
    def detect_faces(self, image):
        """Detect faces in image and return cropped face images with their boxes."""
        if image is None:
            print("Error: None image provided to face detector")
            return []
            
        try:
            faces = self.detector.detect_faces(image)
            results = []
            
            for face in faces:
                x, y, w, h = face['box']
                
                # Ensure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                    
                face_img = image[y:y+h, x:x+w]
                results.append({
                    'box': (x, y, w, h),
                    'confidence': face['confidence'],
                    'face_img': face_img
                })
                
            return results
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def retrieve_matches(self, query_img, threshold=0.85, k=3):
        """Find matches for a query image among stored embeddings."""
        # First try to detect faces
        faces = self.detect_faces(query_img)
        
        if not faces:
            #print("No faces detected in query image")
            return None
        
        # Process each detected face
        all_matches = []
        
        for face_data in faces:
            face_img = face_data['face_img']
            query_embedding = self.get_embedding(face_img)
            
            if query_embedding is None:
                print("Failed to generate embedding for face")
                continue
            
            # Check if we have database access
            if self.db_manager:
                db_matches = self.db_manager.retrieve_similar_faces(query_embedding, k=k, threshold=threshold)
                if db_matches:
                    # Add face box from detection
                    for match in db_matches:
                        match['box'] = face_data['box']
                    all_matches.extend(db_matches)
                    continue
            
            # Fallback: search in stored embeddings from memory
            matches = []
            for person_id, stored_embedding in self.stored_embeddings.items():
                # Compare with stored embedding
                similarity = self.compute_similarity(query_embedding, stored_embedding)
                if similarity >= threshold:
                    # For each match, add all the information we need
                    matches.append({
                        'person_id': person_id,
                        'similarity': similarity,
                        'box': face_data['box'],
                        # We don't have the face image stored in memory matches
                    })
            
            # Sort matches by similarity score
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Keep only top k matches
            all_matches.extend(matches[:k])
            
        return all_matches if all_matches else None
    
    def retrieve_live_matches(self, frame, threshold=0.75, k=3):
        """Specialized version for live camera retrieval with visualization."""
        matches = self.retrieve_matches(frame, threshold=threshold, k=k)
        
        # If no matches at all, return early
        if not matches:
            return frame, None
            
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw bounding boxes and labels for each match
        for match in matches:
            if 'box' in match:
                x, y, w, h = match['box']
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label with person_id and confidence
                person_id = match['person_id']
                similarity = match['similarity']
                label = f"{person_id}: {similarity:.2f}"
                
                # Calculate text position
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.putText(vis_frame, label, (x, y - 10 if y > 20 else y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_frame, matches
    
    def evaluate_retrieval(self, test_images_dir, ground_truth):
        """
        Evaluate retrieval performance on a test set.
        
        Args:
            test_images_dir: Directory containing test images
            ground_truth: Dict mapping image filenames to correct person_ids
        """
        predictions = []
        true_labels = []
        
        # Ensure test directory exists
        if not os.path.exists(test_images_dir):
            print(f"Test directory {test_images_dir} does not exist")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'total_samples': 0
            }
            
        for img_file, true_id in ground_truth.items():
            img_path = os.path.join(test_images_dir, img_file)
            if not os.path.exists(img_path):
                print(f"Warning: Test image {img_path} does not exist")
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Failed to load image {img_path}")
                    continue
                    
                matches = self.retrieve_matches(img)
                
                if matches:
                    # Get the person_id with highest similarity
                    pred_id = matches[0]['person_id']
                    predictions.append(pred_id)
                    true_labels.append(true_id)
                    print(f"Image: {img_file}, Predicted: {pred_id}, True: {true_id}")
                else:
                    print(f"No match found for {img_file}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # If no predictions were made, return zeros
        if len(predictions) == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'total_samples': 0
            }
            
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_samples': len(true_labels)
        }