import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path

class FaceRetrievalPipeline:
    def __init__(self, embedding_dir='face_embedding/inp_embedding', device=None):
        self.embedding_dir = embedding_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        # Initialize models
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        self.detector = MTCNN(device=self.device)
        
        # Load stored embeddings
        self.stored_embeddings = self.load_stored_embeddings()
        print(f"Loaded {len(self.stored_embeddings)} person IDs from embeddings")
        
    def load_stored_embeddings(self):
        """Load all stored embeddings from the directory."""
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
    
    def retrieve_matches(self, query_img, threshold=0.85):
        """Find matches for a query image among stored embeddings."""
        # Detect and get embedding for query face
        faces = self.detector.detect_faces(query_img)
        if not faces:
            print("No faces detected in query image")
            return None
        
        # Process the largest face in the image
        largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
        x, y, w, h = largest_face['box']
        
        # Ensure we don't go out of bounds
        x, y = max(0, x), max(0, y)
        w = min(w, query_img.shape[1] - x)
        h = min(h, query_img.shape[0] - y)
        
        if w <= 0 or h <= 0:
            print("Invalid face dimensions after boundary check")
            return None
            
        face_img = query_img[y:y+h, x:x+w]
        query_embedding = self.get_embedding(face_img)
        
        if query_embedding is None:
            print("Failed to generate embedding for query face")
            return None
        
        # Find best matches
        matches = {}
        for person_id, stored_embedding in self.stored_embeddings.items():
            # Compare with stored embedding
            similarity = self.compute_similarity(query_embedding, stored_embedding)
            if similarity >= threshold:
                matches[person_id] = similarity
        
        # Sort matches by similarity score
        sorted_matches = dict(sorted(matches.items(), key=lambda item: item[1], reverse=True))
        return sorted_matches if sorted_matches else None
    
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
                    pred_id = next(iter(matches))
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

def demo_retrieval():
    """Demo function to show how to use the pipeline."""
    # Initialize pipeline
    pipeline = FaceRetrievalPipeline()
    
    # Example of single image retrieval
    test_img_path = 'path_to_test_image.jpg'
    if os.path.exists(test_img_path):
        test_img = cv2.imread(test_img_path)
        
        if test_img is not None:
            matches = pipeline.retrieve_matches(test_img)
            
            if matches:
                print("\nMatches found:")
                for person_id, similarity in matches.items():
                    print(f"{person_id}: {similarity:.3f}")
            else:
                print("No matches found for test image")
        else:
            print(f"Failed to load test image: {test_img_path}")
    else:
        print(f"Test image path does not exist: {test_img_path}")
    
    # Example of evaluation
    test_dir = 'test_images_dir'
    if os.path.exists(test_dir):
        ground_truth = {
            'test1.jpg': 'Person_1',
            'test2.jpg': 'Person_2',
            # ... more test images
        }
        
        metrics = pipeline.evaluate_retrieval(test_dir, ground_truth)
        print("\nRetrieval Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
    else:
        print(f"Test directory does not exist: {test_dir}")

if __name__ == "__main__":
    demo_retrieval()