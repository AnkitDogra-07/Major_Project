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
        self.detector = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load stored embeddings
        self.stored_embeddings = self.load_stored_embeddings()
        
    def load_stored_embeddings(self):
        """Load all stored embeddings from the directory."""
        embeddings = {}
        for file in Path(self.embedding_dir).glob('*.npy'):
            person_id = file.stem.split('_')[0:2]  # Get Person_X part
            person_id = '_'.join(person_id)
            embedding = np.load(str(file))
            if person_id in embeddings:
                if not isinstance(embeddings[person_id], list):
                    embeddings[person_id] = [embeddings[person_id]]
                embeddings[person_id].append(embedding)
            else:
                embeddings[person_id] = [embedding]
        return embeddings
    
    def get_embedding(self, face_img):
        """Generate embedding for a face image."""
        try:
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
            return None, None
        
        x, y, w, h = faces[0]['box']
        face_img = query_img[y:y+h, x:x+w]
        query_embedding = self.get_embedding(face_img)
        
        if query_embedding is None:
            return None, None
        
        # Find best matches
        matches = {}
        for person_id, stored_embeddings in self.stored_embeddings.items():
            # Compare with all stored embeddings for this person
            similarities = [self.compute_similarity(query_embedding, stored_emb) 
                          for stored_emb in stored_embeddings]
            max_similarity = max(similarities)
            if max_similarity >= threshold:
                matches[person_id] = max_similarity
        
        return matches if matches else None
    
    def evaluate_retrieval(self, test_images_dir, ground_truth):
        """
        Evaluate retrieval performance on a test set.
        
        Args:
            test_images_dir: Directory containing test images
            ground_truth: Dict mapping image filenames to correct person_ids
        """
        predictions = []
        true_labels = []
        
        for img_file, true_id in ground_truth.items():
            img_path = os.path.join(test_images_dir, img_file)
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            matches = self.retrieve_matches(img)
            
            if matches:
                # Get the person_id with highest similarity
                pred_id = max(matches.items(), key=lambda x: x[1])[0]
                predictions.append(pred_id)
                true_labels.append(true_id)
        
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
    test_img = cv2.imread('path_to_test_image.jpg')
    matches = pipeline.retrieve_matches(test_img)
    
    if matches:
        print("Matches found:")
        for person_id, similarity in matches.items():
            print(f"{person_id}: {similarity:.3f}")
    
    # Example of evaluation
    ground_truth = {
        'test1.jpg': 'Person_1',
        'test2.jpg': 'Person_2',
        # ... more test images
    }
    
    metrics = pipeline.evaluate_retrieval('test_images_dir', ground_truth)
    print("\nRetrieval Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    demo_retrieval()