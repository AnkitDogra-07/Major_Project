import os
import datetime
import numpy as np
import psycopg2
from psycopg2.extensions import Binary
import faiss
import cv2
from sklearn.cluster import DBSCAN
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize

class DatabaseManager:
    def __init__(self, db_params=None, index_dir='face_indexes'):
        # Default PostgreSQL connection parameters
        self.db_params = db_params or {
            'dbname': 'major_project_updated',
            'user': 'postgres',
            'password': 'root',
            'host': 'localhost',
            'port': '5432'
        }
        
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        # Initialize databases
        self.init_postgresql()
        self.init_vector_db()
        
    def init_postgresql(self):
        """Initialize PostgreSQL database with required tables."""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Create face_data table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_data (
                    id SERIAL PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    image BYTEA NOT NULL,
                    embedding_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create an index on person_id for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_person_id ON face_data (person_id)
            ''')
            
            conn.commit()
            print("PostgreSQL database initialized successfully")
        except Exception as e:
            print(f"Error initializing PostgreSQL: {e}")
        finally:
            if conn:
                conn.close()
    
    def init_vector_db(self):
        """Initialize FAISS vector database or load existing index."""
        self.index_path = os.path.join(self.index_dir, 'face_index.faiss')
        self.mapping_path = os.path.join(self.index_dir, 'id_mapping.pkl')
        
        # Check if index exists, otherwise create new
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.mapping_path, 'rb') as f:
                    self.id_mapping = pickle.load(f)
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Dimension of FaceNet embeddings is 512
        dimension = 512
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.id_mapping = {}  # Maps FAISS indices to person_ids
        print("Created new FAISS index")
    
    def save_face_data(self, person_id, face_img, embedding, full_img=None):
        """Save face data to PostgreSQL and the embedding to FAISS."""
        try:
            # Save to PostgreSQL
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            # Convert image to binary for storage
            _, img_encoded = cv2.imencode('.jpg', face_img)
            img_binary = Binary(img_encoded.tobytes())

            full_img_binary = None
            if full_img is not None:
                _, full_encoded = cv2.imencode('.jpg', full_img)
                full_img_binary = Binary(full_encoded.tobytes())
            
            # Save embedding to local file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            embedding_filename = f"{person_id}_{timestamp}.npy"
            embedding_path = os.path.join(self.index_dir, embedding_filename)
            np.save(embedding_path, embedding)
            
            # Insert record into PostgreSQL
            cursor.execute(
                "INSERT INTO face_data (person_id, image, embedding_path, created_at, full_image) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (person_id, img_binary, embedding_path, datetime.datetime.now(), full_img_binary)
            )

            db_id = cursor.fetchone()[0]
            conn.commit()
            
            # Add to FAISS index
            faiss_idx = self.index.ntotal
            self.index.add(np.array([embedding], dtype=np.float32))
            self.id_mapping[faiss_idx] = {"person_id": person_id, "db_id": db_id}
            
            # Save updated index
            self._save_index()
            
            print(f"Saved face data for {person_id} (DB ID: {db_id}, Vector ID: {faiss_idx})")
            return db_id
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error saving face data: {e}")
        finally:
            if conn:
                conn.close()
    
    def retrieve_similar_faces(self, embedding, k=5, threshold=0.7):
        """Retrieve similar faces from FAISS index."""
        try:
            if self.index.ntotal == 0:
                print("Vector database is empty")
                return []
            
            # Convert to float32 array with correct shape for FAISS
            query_vector = np.array([embedding], dtype=np.float32)
            
            # Search for similar faces
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                # Convert L2 distance to similarity score (higher is better)
                # For L2 distance, smaller values mean more similar
                similarity = 1 / (1 + dist)
                
                if idx != -1 and idx in self.id_mapping and similarity > threshold:
                    match_info = self.id_mapping[idx]
                    person_id = match_info["person_id"]
                    db_id = match_info["db_id"]
                    
                    # Retrieve image from PostgreSQL
                    face_img = self.get_face_image(db_id)
                    if face_img is not None:
                        results.append({
                            "person_id": person_id,
                            "db_id": db_id,
                            "similarity": similarity,
                            "face_img": face_img
                        })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving similar faces: {e}")
            return []
    
    def get_face_image(self, db_id):
        """Retrieve face image from PostgreSQL by database ID."""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            cursor.execute("SELECT full_image FROM face_data WHERE id = %s", (db_id,))
            result = cursor.fetchone()
            
            if result:
                # Convert binary data back to image
                img_bytes = bytes(result[0])
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            else:
                print(f"No image found for DB ID: {db_id}")
                return None
                
        except Exception as e:
            print(f"Error retrieving face image: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def cluster_embeddings(self, eps=0.5, min_samples=3):
        """Cluster face embeddings using DBSCAN."""
        try:
            if self.index.ntotal < min_samples:
                print(f"Not enough samples for clustering (need at least {min_samples}, have {self.index.ntotal})")
                return {}
            
            # Extract all embeddings from FAISS
            embeddings = embeddings = np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])
            embeddings = normalize(embeddings)
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=0.3, min_samples=3, metric='cosine').fit(embeddings)
            labels = clustering.labels_
            
            # Group by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    # Noise points
                    continue
                    
                if label not in clusters:
                    clusters[label] = []
                
                if i in self.id_mapping:
                    clusters[label].append(self.id_mapping[i])
            
            # Save clusters info
            clusters_path = os.path.join(self.index_dir, 'face_clusters.pkl')
            with open(clusters_path, 'wb') as f:
                pickle.dump(clusters, f)
            
            print(f"Clustered {self.index.ntotal} face embeddings into {len(clusters)} groups")
            return clusters
            
        except Exception as e:
            print(f"Error clustering embeddings: {e}")
            return {}
    
    def _save_index(self):
        """Save FAISS index and ID mapping to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.mapping_path, 'wb') as f:
                pickle.dump(self.id_mapping, f)
                
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_faces_by_person(self, person_id):
        """Load all faces for a given person ID."""
        try:
            conn = psycopg2.connect(**self.db_params)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, image, created_at FROM face_data WHERE person_id = %s", (person_id,))
            results = cursor.fetchall()
            
            faces = []
            for db_id, img_binary, created_at in results:
                # Convert binary data back to image
                img_bytes = bytes(img_binary)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                faces.append({
                    "db_id": db_id,
                    "face_img": img,
                    "created_at": created_at
                })
            
            return faces
            
        except Exception as e:
            print(f"Error loading faces for person {person_id}: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def import_from_directory(self, directory_path):
        """Import existing face embeddings from a directory."""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                print(f"Directory {directory_path} does not exist")
                return 0
            
            count = 0
            # Process all NPY files (embeddings)
            for file in directory.glob('*.npy'):
                try:
                    # Extract person_id from filename
                    filename = file.stem
                    parts = filename.split('_')
                    
                    if len(parts) >= 2 and parts[0] == "Person":
                        person_id = f"Person_{parts[1]}"
                        
                        # Load embedding
                        embedding = np.load(str(file))
                        
                        # Look for corresponding image with same name
                        img_path = file.with_suffix('.jpg')
                        if img_path.exists():
                            face_img = cv2.imread(str(img_path))
                            if face_img is not None:
                                # Save to database
                                self.save_face_data(person_id, face_img, embedding)
                                count += 1
                except Exception as e:
                    print(f"Error importing {file}: {e}")
                    continue
            
            print(f"Imported {count} faces from {directory_path}")
            return count
            
        except Exception as e:
            print(f"Error importing from directory: {e}")
            return 0