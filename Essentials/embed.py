import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
import datetime

class Embed:
    def __init__(self, output_dir='face_embedding', sub_dir='inp_embedding'):
        self.output_dir = output_dir
        self.sub_dir = sub_dir
        os.makedirs(os.path.join(self.output_dir, self.sub_dir), exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
    
    def get_embedding(self, face):
        """Convert face image to 128D embedding using FaceNet."""
        try:
            if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                print("Error: Empty face image")
                return None
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160))
            face = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                embedding = self.facenet(face).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def save_embedding(self, person_id, embedding, face_img):
        """Save embedding to file along with face image."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f"{self.output_dir}/{self.sub_dir}/{person_id}_{timestamp}.npy", embedding)
        cv2.imwrite(f"{self.output_dir}/{self.sub_dir}/{person_id}_{timestamp}.jpg", face_img)
        print(f"Saved embedding and image for {person_id}")
    
    def find_best_match(self, embedding, known_embeddings, threshold=0.85):
        """Find the best matching person_id for the given embedding."""
        best_match = None
        best_similarity = threshold

        for name, known_emb in known_embeddings.items():
            # If this is a list of embeddings, use the average
            if isinstance(known_emb, list):
                avg_emb = np.mean(known_emb, axis=0)
                similarity = np.dot(embedding, avg_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(avg_emb)
                )
            else:
                similarity = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                )
                
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        return best_match, best_similarity
    
    def update_person_identity(self, face_img, known_embeddings, embedding_buffer, next_id):
        """Update person identity based on face embedding.
        
        Returns:
            tuple: (person_id, similarity, next_id)
        """
        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None
        
        local_next_id = next_id  # Make a local copy to track changes

        # Find best match among existing embeddings
        best_match, similarity = self.find_best_match(embedding, known_embeddings)

        # Check with embedding buffer for more stable recognition
        if best_match and best_match in embedding_buffer:
            # Calculate similarity with average embedding in buffer
            avg_embedding = np.mean(embedding_buffer[best_match], axis=0)
            avg_similarity = np.dot(embedding, avg_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(avg_embedding)
            )
            
            # If good match, update buffer
            if avg_similarity > 0.80:
                embedding_buffer[best_match].append(embedding)
                if len(embedding_buffer[best_match]) > 10:
                    embedding_buffer[best_match].pop(0)
                    
                # Update known embedding with refined average
                if len(embedding_buffer[best_match]) >= 3:
                    known_embeddings[best_match] = np.mean(embedding_buffer[best_match], axis=0)
                    
                return best_match, similarity, local_next_id

        # Assign new ID if no match found
        new_id = f"Person_{local_next_id}"
        local_next_id += 1
        known_embeddings[new_id] = embedding
        embedding_buffer[new_id] = [embedding]
        self.save_embedding(new_id, embedding, face_img)
        
        return new_id, 1.0, local_next_id