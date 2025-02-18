import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
import datetime

# Create output directory for embeddings if it doesn't exist
output_dir = 'face_embedding'
sub_dir = 'inp_embedding'
os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dev_str = 'cuda' if torch.cuda.is_available() else 'cpu'
facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
detector = MTCNN(device=dev_str)

# Initialize variables
known_embeddings = {}  # Store embeddings
embedding_buffer = {}  # Store recent embeddings for tracked faces
trackers = {}  # Store multiple trackers {id: tracker}
next_id = 1  # Unique ID for each new face
frame_count = 0

def get_embedding(face):
    """Convert face image to 128D embedding using FaceNet."""
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = torch.tensor(face).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            embedding = facenet(face).cpu().numpy().flatten()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def save_embedding(person_id, embedding, face_img):
    """Save embedding to file along with face image."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(f"{output_dir}/{sub_dir}/{person_id}_{timestamp}.npy", embedding)
    cv2.imwrite(f"{output_dir}/{sub_dir}/{person_id}_{timestamp}.jpg", face_img)
    print(f"Saved embedding and image for {person_id}")

def find_best_match(embedding, known_embeddings, threshold=0.85):
    """Find the best matching person_id for the given embedding."""
    best_match = None
    best_similarity = threshold
    
    for name, known_emb in known_embeddings.items():
        similarity = np.dot(embedding, known_emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(known_emb)
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    return best_match, best_similarity

def update_person_identity(face_img):
    """Update person identity based on face embedding."""
    global next_id
    embedding = get_embedding(face_img)
    if embedding is None:
        return None, None

    best_match, similarity = find_best_match(embedding, known_embeddings)

    if best_match and best_match in embedding_buffer:
        avg_embedding = np.mean(embedding_buffer[best_match], axis=0)
        avg_similarity = np.dot(embedding, avg_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(avg_embedding)
        )
        if avg_embedding > 0.85:
            embedding_buffer[best_match].append(embedding)
            if len(embedding_buffer[best_match]) > 5:
                embedding_buffer[best_match].pop(0)
            return best_match, similarity

    # Assign new ID if no match
    new_id = f"Person_{next_id}"
    next_id += 1
    known_embeddings[new_id] = embedding
    embedding_buffer[new_id] = [embedding]
    save_embedding(new_id, embedding, face_img)
    return new_id, 1.0

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow('Multi-Face Tracking')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    # Run detection every 10 frames or if no active trackers
    if frame_count % 10 == 0 or len(trackers) == 0:
        faces = detector.detect_faces(frame)
        trackers = {}  # Reset trackers

        if faces:
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                w = min(w, frame_width - x)
                h = min(h, frame_height - y)

                if w > 0 and h > 0:
                    face_img = frame[y:y+h, x:x+w]
                    person_id, similarity = update_person_identity(face_img)

                    if person_id:
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        trackers[person_id] = tracker

    # Update all active trackers
    for person_id, tracker in list(trackers.items()):
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, person_id, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            del trackers[person_id]  # Remove lost trackers

    # Display known faces count
    cv2.putText(display_frame, f"Tracked Faces: {len(trackers)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Multi-Face Tracking', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save final embeddings
print(f"Saving {len(known_embeddings)} embeddings to {output_dir}")
for person_id, embedding in known_embeddings.items():
    np.save(f"{output_dir}/{sub_dir}/{person_id}_final.npy", embedding)

cap.release()
cv2.destroyAllWindows()
