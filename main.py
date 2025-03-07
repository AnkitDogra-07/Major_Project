import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
import datetime
from mtcnn import MTCNN
from Essentials.embed import Embed


# # Initialize variables
output_dir = 'face_embedding'
sub_dir = 'inp_embedding'
detector = MTCNN()
os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
Emb = Embed()
known_embeddings = {}  # Store embeddings
embedding_buffer = {}  # Store recent embeddings for tracked faces
trackers = {}  # Store multiple trackers {id: tracker}
next_id = 1  # Unique ID for each new face
frame_count = 0

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
    detection_interval = max(5, 20 - len(trackers))

    # Run detection every 10 frames or if no active trackers
    if frame_count % detection_interval == 0 or len(trackers) == 0:
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
                    best_match, similarity, person_id = Emb.update_person_identity(face_img, known_embeddings, embedding_buffer, next_id)

                    if person_id:
                        tracker = cv2.TrackerKCF.create()
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