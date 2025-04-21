import cv2
import torch
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1
import datetime
from mtcnn import MTCNN
from Essentials.embed import Embed


# Initialize variables
output_dir = 'face_embedding'
sub_dir = 'inp_embedding'
input_dir = "Images"
detector = MTCNN()
os.makedirs(input_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
Emb = Embed()
known_embeddings = {}  # Store embeddings
embedding_buffer = {}  # Store recent embeddings for tracked faces
trackers = {}  # Store multiple trackers {id: tracker}
next_id = 1  # Unique ID for each new face
frame_count = 0

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow('Multi-Face Tracking')

# Create modern trackers - OpenCV 4.5.1+ compatible
def create_tracker():
    # Use CSRT for better accuracy, or KCF for speed
    tracker_types = {'KCF': cv2.TrackerKCF_create,
                     'CSRT': cv2.TrackerCSRT_create,
                     'MOSSE': cv2.legacy.TrackerMOSSE_create}
    
    # Try different tracker types based on OpenCV version
    for tracker_name in tracker_types:
        try:
            return tracker_types[tracker_name]()
        except (AttributeError, cv2.error):
            continue
            
    # Fallback for newer OpenCV versions
    return cv2.TrackerKCF.create() if hasattr(cv2, 'TrackerKCF') else None

def make_album():
    
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     display_frame = frame.copy()
#     detection_interval = max(5, 20 - len(trackers))

#     # Run detection every few frames or if no active trackers
#     if frame_count % detection_interval == 0 or len(trackers) == 0:
#         faces = detector.detect_faces(frame)
        
#         if faces:
#             # Store current tracker IDs to avoid losing tracked faces
#             current_ids = list(trackers.keys())
#             current_tracker_regions = {}
            
#             # Get regions for current trackers
#             for person_id, tracker in trackers.items():
#                 success, bbox = tracker.update(frame)
#                 if success:
#                     current_tracker_regions[person_id] = bbox
            
#             # Process detected faces
#             for face in faces:
#                 x, y, w, h = face['box']
#                 x, y = max(0, x), max(0, y)
#                 w = min(w, frame_width - x)
#                 h = min(h, frame_height - y)

#                 if w > 0 and h > 0:
#                     face_img = frame[y:y+h, x:x+w]
                    
#                     # Check if this overlaps with any existing tracker
#                     matched_id = None
#                     for person_id, (tx, ty, tw, th) in current_tracker_regions.items():
#                         # Check overlap (simple IoU)
#                         if (x < tx + tw and x + w > tx and 
#                             y < ty + th and y + h > ty):
#                             matched_id = person_id
#                             break
                    
#                     if matched_id:
#                         # Update existing tracker
#                         if matched_id in trackers:
#                             trackers[matched_id] = create_tracker()
#                             trackers[matched_id].init(frame, (x, y, w, h))
#                     else:
#                         # Process as new face
#                         result = Emb.update_person_identity(face_img, known_embeddings, embedding_buffer, next_id)
#                         if result:
#                             best_match, similarity, new_next_id = result
#                             if new_next_id > next_id:
#                                 next_id = new_next_id  # Update next_id if it was incremented
                            
#                             new_tracker = create_tracker()
#                             if new_tracker:
#                                 new_tracker.init(frame, (x, y, w, h))
#                                 trackers[best_match] = new_tracker
#                             else:
#                                 print("Warning: Failed to create tracker, OpenCV version may be incompatible")

#     # Update all active trackers
#     for person_id, tracker in list(trackers.items()):
#         success, bbox = tracker.update(frame)
#         if success:
#             x, y, w, h = [int(v) for v in bbox]
#             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(display_frame, person_id, (x, y - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         else:
#             del trackers[person_id]  # Remove lost trackers

#     # Display known faces count
#     cv2.putText(display_frame, f"Tracked Faces: {len(trackers)}", (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#     cv2.imshow('Multi-Face Tracking', display_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Save final embeddings
print(f"Saving {len(known_embeddings)} embeddings to {output_dir}")
for person_id, embedding in known_embeddings.items():
    np.save(f"{output_dir}/{sub_dir}/{person_id}_final.npy", embedding)

cap.release()
cv2.destroyAllWindows()