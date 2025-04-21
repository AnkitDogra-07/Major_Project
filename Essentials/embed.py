import cv2
import torch
import numpy as np
import os
import datetime
from facenet_pytorch import InceptionResnetV1

class Embed:
    def __init__(self, output_dir='face_embedding', sub_dir='inp_embedding', db_manager=None):
        self.output_dir = output_dir
        self.sub_dir = sub_dir
        self.db_manager = db_manager
        os.makedirs(os.path.join(self.output_dir, self.sub_dir), exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        print(f"Embed module initialized with device: {self.device}")
    
    def get_embedding(self, face):
        """Convert face image to 128D embedding using FaceNet."""
        try:
            if face is None or face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                print("Error: Empty or invalid face image")
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
        file_path = f"{self.output_dir}/{self.sub_dir}/{person_id}_{timestamp}.npy"
        img_path = f"{self.output_dir}/{self.sub_dir}/{person_id}_{timestamp}.jpg"
        
        np.save(file_path, embedding)
        cv2.imwrite(img_path, face_img)
        
        # If database manager is available, also save to database
        if self.db_manager:
            self.db_manager.save_face_data(person_id, face_img, embedding)
            
        print(f"Saved embedding and image for {person_id}")
        return file_path, img_path
    
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
    
    def update_person_identity(self, face_img, known_embeddings, embedding_buffer, next_id, save_to_db=True):
        """Update person identity based on face embedding.
        
        Args:
            face_img: Image containing the face
            known_embeddings: Dictionary of known embeddings
            embedding_buffer: Dictionary for buffering recent embeddings
            next_id: Next available ID to assign
            save_to_db: Whether to save the data to database
            
        Returns:
            tuple: (person_id, similarity, next_id)
        """
        embedding = self.get_embedding(face_img)
        if embedding is None:
            print("Failed to generate embedding for face")
            return None
        
        local_next_id = next_id  # Make a local copy to track changes

        # If database manager exists, try finding a match in the vector database first
        db_match = None
        if self.db_manager and save_to_db:
            matches = self.db_manager.retrieve_similar_faces(embedding, k=1, threshold=0.75)
            if matches:
                db_match = matches[0]["person_id"]
                print(f"Found database match: {db_match} with similarity {matches[0]['similarity']:.3f}")

        # Find best match among existing embeddings (memory)
        best_match, similarity = self.find_best_match(embedding, known_embeddings)
        
        # If we have a database match, prioritize it
        if db_match:
            if best_match and best_match != db_match:
                print(f"Database match {db_match} differs from memory match {best_match}")
            best_match = db_match
            # Make sure this ID is in our embedding buffer
            if best_match not in embedding_buffer:
                embedding_buffer[best_match] = []

        # Check with embedding buffer for more stable recognition
        if best_match and best_match in embedding_buffer:
            # Calculate similarity with average embedding in buffer
            if embedding_buffer[best_match]:  # Make sure buffer is not empty
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
                    
                    # If requested, save to database
                    if save_to_db and self.db_manager:
                        self.db_manager.save_face_data(best_match, face_img, embedding)
                        
                    return best_match, similarity, local_next_id
            else:
                # If buffer exists but is empty, initialize it
                embedding_buffer[best_match] = [embedding]

        # Assign new ID if no match found
        new_id = f"Person_{local_next_id}"
        local_next_id += 1
        known_embeddings[new_id] = embedding
        embedding_buffer[new_id] = [embedding]
        
        # Save the embedding and image
        if save_to_db and self.db_manager:
            self.db_manager.save_face_data(new_id, face_img, embedding)
        else:
            self.save_embedding(new_id, embedding, face_img)
        
        return new_id, 1.0, local_next_id

    def load_embeddings_from_db(self):
        """Load embeddings from database if available."""
        if self.db_manager:
            # This is a placeholder - the db_manager would need a method to retrieve all embeddings
            # which it currently doesn't have. You would implement this in db_manager.py
            pass
        
        # Default: load from files
        return self.load_embeddings_from_files()
        
    def load_embeddings_from_files(self):
        """Load embeddings from files in the embedding directory."""
        embeddings = {}
        embedding_dir = os.path.join(self.output_dir, self.sub_dir)
        
        if not os.path.exists(embedding_dir):
            print(f"Embedding directory does not exist: {embedding_dir}")
            return embeddings
            
        for file in os.listdir(embedding_dir):
            if file.endswith('.npy'):
                try:
                    # Extract person_id from filename (Person_X part)
                    parts = file.split('_')
                    if len(parts) >= 2 and parts[0] == "Person":
                        person_id = f"Person_{parts[1]}"
                        
                        # Load embedding
                        embedding = np.load(os.path.join(embedding_dir, file))
                        
                        if person_id in embeddings:
                            if not isinstance(embeddings[person_id], list):
                                embeddings[person_id] = [embeddings[person_id]]
                            embeddings[person_id].append(embedding)
                        else:
                            embeddings[person_id] = embedding
                except Exception as e:
                    print(f"Error loading embedding {file}: {e}")
        
        # Convert lists to averages
        for person_id in embeddings:
            if isinstance(embeddings[person_id], list):
                embeddings[person_id] = np.mean(embeddings[person_id], axis=0)
                
        print(f"Loaded {len(embeddings)} embeddings from files")
        return embeddings