import cv2
import torch
import numpy as np
import os
import datetime
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from facenet_pytorch import InceptionResnetV1
from mtcnn import MTCNN
from Essentials.embed import Embed
from Essentials.retrieval import FaceRetrievalPipeline
from Essentials.db_manager import DatabaseManager


# Fix: OpenMP conflict warning (Intel vs LLVM)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Optional: Avoid CPU count warning from loky/joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Since actual core count if known

# Initialize detector and embedding module
detector = MTCNN()
Emb = Embed()
db_manager = DatabaseManager()

# Initialize variables
known_embeddings = {}  # Store embeddings
embedding_buffer = {}  # Store recent embeddings for tracked faces
trackers = {}  # Store multiple trackers {id: tracker}
next_id = 1  # Unique ID for each new face
frame_count = 0
display_matches = False  # Toggle for displaying matched faces
retrieved_matches = []  # Store retrieved matches

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

def upload_images():
    """Open a file dialog to upload and process multiple images."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_paths = filedialog.askopenfilenames(
        title="Select Face Images",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if not file_paths:
        return
    
    upload_count = 0
    for file_path in file_paths:
        try:
            # Read image
            img = cv2.imread(file_path)
            if img is None:
                print(f"Failed to load image: {file_path}")
                continue
                
            # Detect faces
            faces = detector.detect_faces(img)
            if not faces:
                print(f"No faces detected in: {file_path}")
                continue
                
            # Process largest face
            largest_face = max(faces, key=lambda face: face['box'][2] * face['box'][3])
            x, y, w, h = largest_face['box']
            x, y = max(0, x), max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            if w <= 0 or h <= 0:
                print(f"Invalid face dimensions: {file_path}")
                continue
                
            face_img = img[y:y+h, x:x+w]
            
            # Generate embedding
            embedding = Emb.get_embedding(face_img)
            if embedding is None:
                print(f"Failed to generate embedding: {file_path}")
                continue
                
            # Update person identity and store in database
            global next_id
            result = Emb.update_person_identity(face_img, known_embeddings, embedding_buffer, next_id)
            if result:
                person_id, similarity, new_next_id = result
                if new_next_id > next_id:
                    next_id = new_next_id
                
                # Save to database
                db_id = db_manager.save_face_data(person_id, face_img, embedding, full_img=img)
                if db_id:
                    upload_count += 1
                    print(f"Uploaded and processed: {file_path} as {person_id}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if upload_count > 0:
        messagebox.showinfo("Upload Complete", f"Successfully processed {upload_count} images")
        
        # Cluster embeddings after uploading new faces
        db_manager.cluster_embeddings()

def toggle_display_matches():
    """Toggle display of matching faces."""
    global display_matches
    display_matches = not display_matches

def run_live_detection():
    """Run live face detection and recognition using webcam."""
    global frame_count, trackers, next_id, retrieved_matches, display_matches
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create windows
    cv2.namedWindow('Multi-Face Tracking')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()
        detection_interval = max(5, 20 - len(trackers))

        # Run detection every few frames or if no active trackers
        if frame_count % detection_interval == 0 or len(trackers) == 0:
            faces = detector.detect_faces(frame)
            
            if faces:
                # Store current tracker IDs to avoid losing tracked faces
                current_ids = list(trackers.keys())
                current_tracker_regions = {}
                
                # Get regions for current trackers
                for person_id, tracker in trackers.items():
                    success, bbox = tracker.update(frame)
                    if success:
                        current_tracker_regions[person_id] = bbox
                
                # Process detected faces
                for face in faces:
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    w = min(w, frame_width - x)
                    h = min(h, frame_height - y)

                    if w > 0 and h > 0:
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Generate embedding for retrieval
                        embedding = Emb.get_embedding(face_img)
                        if embedding is not None:
                            # Search for matches in the vector database
                            matches = db_manager.retrieve_similar_faces(embedding, k=3, threshold=0.7)
                            if matches:
                                retrieved_matches = matches
                                # Use the best match person_id for tracking
                                best_match = matches[0]["person_id"]
                                
                                # Check if this overlaps with any existing tracker
                                matched_id = None
                                for person_id, (tx, ty, tw, th) in current_tracker_regions.items():
                                    # Check overlap (simple IoU)
                                    if (x < tx + tw and x + w > tx and 
                                        y < ty + th and y + h > ty):
                                        matched_id = person_id
                                        break
                                
                                if matched_id:
                                    # Update existing tracker
                                    if matched_id in trackers:
                                        trackers[matched_id] = create_tracker()
                                        trackers[matched_id].init(frame, (x, y, w, h))
                                else:
                                    # Create new tracker for detected face
                                    new_tracker = create_tracker()
                                    if new_tracker:
                                        new_tracker.init(frame, (x, y, w, h))
                                        trackers[best_match] = new_tracker
                                    else:
                                        print("Warning: Failed to create tracker")
                            else:
                                # No match found, treat as new face
                                result = Emb.update_person_identity(
                                    face_img, 
                                    known_embeddings, 
                                    embedding_buffer, 
                                    next_id, 
                                    save_to_db=False)
                                if result:
                                    person_id, similarity, new_next_id = result
                                    if new_next_id > next_id:
                                        next_id = new_next_id
                                    
                                    # Save to database
                                    # db_manager.save_face_data(person_id, face_img, embedding)
                                    
                                    # Create new tracker
                                    new_tracker = create_tracker()
                                    if new_tracker:
                                        new_tracker.init(frame, (x, y, w, h))
                                        trackers[person_id] = new_tracker
                                    else:
                                        print("Warning: Failed to create tracker")

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

        # Display matched faces if enabled
        if display_matches and retrieved_matches:
            # Create a separate window or panel within the main window
            match_display = np.zeros((200, frame_width, 3), dtype=np.uint8)
            
            # Display up to 3 matched faces
            spacing = frame_width // min(len(retrieved_matches), 3)
            for i, match in enumerate(retrieved_matches[:3]):
                # Resize matched face to fit in the panel
                match_img = match["face_img"]
                match_img = cv2.resize(match_img, (150, 150))
                h, w, _ = match_img.shape
                
                # Calculate position
                pos_x = i * spacing + (spacing - w) // 2
                
                # Place the image in the panel
                if pos_x + w <= frame_width:
                    match_display[10:10+h, pos_x:pos_x+w] = match_img
                
                # Add text with person_id and similarity
                person_id = match["person_id"]
                similarity = match["similarity"]
                text = f"{person_id}: {similarity:.2f}"
                cv2.putText(match_display, text, (pos_x, h + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the matches panel
            cv2.imshow('Matched Faces', match_display)

        # Display tracked faces count
        cv2.putText(display_frame, f"Tracked Faces: {len(trackers)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add keyboard controls info
        cv2.putText(display_frame, "Q: Quit  U: Upload  M: Toggle Matches  C: Cluster", 
                   (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Multi-Face Tracking', display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):
            # Pause camera temporarily
            cap.release()
            upload_images()
            # Resume camera
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif key == ord('m'):
            toggle_display_matches()
        elif key == ord('c'):
            # Run clustering on existing embeddings
            clusters = db_manager.cluster_embeddings()
            if clusters:
                print(f"Created {len(clusters)} clusters of similar faces")

    # Save final embeddings
    print(f"Saving {len(known_embeddings)} embeddings")
    for person_id, embedding in known_embeddings.items():
        np.save(f"face_embedding/inp_embedding/{person_id}_final.npy", embedding)

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Create argument parser for different modes
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--upload', action='store_true', help='Upload images mode')
    parser.add_argument('--import', dest='import_dir', help='Import existing embeddings from directory')
    parser.add_argument('--cluster', action='store_true', help='Cluster existing embeddings')
    args = parser.parse_args()
    
    # Import existing embeddings if specified
    if args.import_dir:
        print(f"Importing embeddings from {args.import_dir}")
        count = db_manager.import_from_directory(args.import_dir)
        print(f"Imported {count} embeddings")
    
    # Run clustering if specified
    if args.cluster:
        print("Clustering existing embeddings")
        clusters = db_manager.cluster_embeddings()
        print(f"Created {len(clusters)} clusters")
    
    # Upload mode
    if args.upload:
        upload_images()
    else:
        # Default: run live detection
        run_live_detection()

if __name__ == "__main__":
    main()