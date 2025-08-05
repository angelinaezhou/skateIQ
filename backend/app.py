from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp 
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
import time
import threading

UPLOADS_FOLDER = Path('uploads')
UPLOADS_FOLDER.mkdir(exist_ok=True)

def cleanup_old_files():
    """Remove uploaded videos and pose files older than 1 hour"""
    while True:
        try:
            current_time = time.time()
            one_hour_ago = current_time - 3600 
            
            # Clean uploads folder (videos)
            if UPLOADS_FOLDER.exists():
                for file_path in UPLOADS_FOLDER.glob('*'):
                    if file_path.stat().st_mtime < one_hour_ago:
                        file_path.unlink()
                        print(f"🗑️ Cleaned up old video: {file_path.name}")
            
            # Clean frames folder (pose data)
            if FRAMES_FOLDER.exists():
                for file_path in FRAMES_FOLDER.glob('*.npy'):
                    if file_path.stat().st_mtime < one_hour_ago:
                        file_path.unlink()
                        print(f"🗑️ Cleaned up old pose data: {file_path.name}")
                        
            print(f"🧹 Cleanup completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(3600)

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = 0.30

# load trained model and scaler
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# constants from training script
MAX_FRAMES = 30
POSE_DIM = 132
TARGET_LEN = 30

# mediapipe setup 
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

# directories
FRAMES_FOLDER = Path('frames')
FRAMES_FOLDER.mkdir(exist_ok=True)
EXISTING_FOLDER = Path('existing')
EXISTING_FOLDER.mkdir(exist_ok=True)

# allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def center_landmarks(landmarks):
    """Center landmarks relative to hip center"""
    try:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][:3]  # x,y,z
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][:3]  # x,y,z
        center = (left_hip + right_hip) / 2
        centered = landmarks[:, :3] - center
        return np.concatenate([centered, landmarks[:, 3:4]], axis=1) 
    except Exception as e:
        print(f"Warning: Could not center landmarks: {e}")
        return landmarks
    
def resample_pose_sequence(pose_seq, target_len=TARGET_LEN):
    """Resample pose sequence to target length"""
    if len(pose_seq) == target_len:
        return pose_seq
    
    indices = np.linspace(0, len(pose_seq) - 1, target_len)
    return np.array([
        np.interp(indices, np.arange(len(pose_seq)), pose_seq[:, i])
        for i in range(pose_seq.shape[1])
    ]).T

def process_video_from_bytes(file_bytes, original_filename):
    """Process video file from bytes in memory"""

    # temporary file that gets automatically deleted
    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4') as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()

        cap = cv2.VideoCapture(temp_file.name)
        all_landmarks = []

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {original_filename}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Processing {original_filename}: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        
        frame_count = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                lm_array = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ])
                centered_landmarks = center_landmarks(lm_array)
                all_landmarks.append(centered_landmarks.flatten())
                processed_frames += 1

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({processed_frames} with pose)")

        cap.release()

        print(f"Final: {processed_frames}/{frame_count} frames had pose landmarks")

        if len(all_landmarks) < 2:
            raise ValueError(f"Too few frames with pose detection: {len(all_landmarks)}")
        
        pose_seq = np.array(all_landmarks)
        resampled = resample_pose_sequence(pose_seq, TARGET_LEN)

        return resampled, {
            'fps': fps, 
            'total_frames': processed_frames, 
            'duration': duration, 
            'pose_detection_rate': processed_frames / frame_count if frame_count > 0 else 0
        }

def get_video_info(video_path):
    """Extract video metadata including FPS, duration, frame count"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        'fps': fps, 
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def load_and_flatten(pose_file: Path, max_frames=MAX_FRAMES):
    """Same preprocessing as training script"""
    pose = np.load(pose_file)
    if len(pose) > max_frames:
        pose = pose[:max_frames]
    elif len(pose) < max_frames:
        padding = np.zeros((max_frames - len(pose), pose.shape[1]))
        pose = np.vstack([pose, padding])
    return pose.flatten()

@app.route('/')
def index():
    return 'SkateIQ backend is running!'

@app.route('/api/upload-video', methods=['POST'])
def upload_video():
    """Handle video upload and process it without saving video to dksk"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only MP4, MOV, AVI, MKV allowed'}), 400
        
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0) 

        max_size = 100 * 1024 * 1024
        if file_size > max_size:
            return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

        # generate unique filename
        unique_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        
        print(f"Processing video in memory: {original_filename} ({file_size / 1024 / 1024:.1f}MB)")

        file_bytes = file.read()

        pose_data, video_info = process_video_from_bytes(file_bytes, original_filename)

        pose_filename = f"{unique_id}.npy"
        pose_path = FRAMES_FOLDER / pose_filename
        np.save(str(pose_path), pose_data)

        print(f"Saved pose data: {pose_filename}, shape: {pose_data.shape}")
        print(f"Video processed successfully - original video not stored")

        return jsonify({
            'success': True, 
            'message': 'Video processed successfully', 
            'pose_filename': pose_filename, 
            'video_id': unique_id, 
            'original_filename': original_filename, 
            'pose_shape': list(pose_data.shape), 
            'video_info': video_info, 
            'file_size_mb': round(file_size / 1024 /1024, 1)
        })
    
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/classify-jump', methods=['POST'])
def classify_jump():
    try:
        data = request.json
        filename = data.get('filename')
        skater = data.get('skater')
        event = data.get('event')
        frames_dir = Path('frames')
    
        frames_path = Path('existing') / filename 
        print(f"Frames directory exists: {frames_dir.exists()}")

        if frames_dir.exists():
            print(f"Files in frames directory: {list(frames_dir.glob('*.npy'))}")
        
        frames_path = Path('existing') / filename
        print(f"Looking for: {frames_path}")
        print(f"File exists: {frames_path.exists()}")
        
        if not frames_path.exists():
            return jsonify({'error': f'Frames file not found: {filename}'}), 404
            
        features = load_and_flatten(frames_path)
        
        # scale the features using the saved scaler
        features_scaled = scaler.transform([features])
        
        # make prediction
        prediction = model.predict(features_scaled)[0]
        
        # get prediction probabilities
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'jump_type': 'Unable to classify',
                'confidence': float(confidence),
                'message': f'Confidence ({confidence:.1%}) is too low.)',
                'all_probabilities': {class_name: float(prob) for i, (class_name, prob) in enumerate(zip(model.classes_, probabilities))},
                'top_predictions':[
                    {
                        'rank': i + 1,
                        'jump_type': class_name,
                        'probability': float(prob)
                    }
                    for i, (class_name, prob) in enumerate(sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3])
                ],
                'skater': skater,
                'event': event,
                'classification_attempted': True,
            })
    
        class_probs = {}
        for i, class_name in enumerate(model.classes_):
            class_probs[class_name] = float(probabilities[i])
        
        # list top 3 predictions
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {
                'rank': i + 1,
                'jump_type': class_name,
                'probability': float(prob)
            }
            for i, (class_name, prob) in enumerate(sorted_probs[:3])
        ]
        
        return jsonify({
            'jump_type': prediction,
            'confidence': float(confidence),
            'all_probabilities': class_probs,
            'top_predictions': top_predictions,
            'skater': skater,
            'event': event
        })
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500
    
@app.route('/api/classify-own-jump', methods=['POST'])
def classify_own_jump():
    try:
        data = request.json
        filename = data.get('filename')
        skater = data.get('skater')
        event = data.get('event')
        frames_dir = Path('frames')
    
        frames_path = Path('frames') / filename 
        print(f"Frames directory exists: {frames_dir.exists()}")

        if frames_dir.exists():
            print(f"Files in frames directory: {list(frames_dir.glob('*.npy'))}")
        
        frames_path = Path('frames') / filename
        print(f"Looking for: {frames_path}")
        print(f"File exists: {frames_path.exists()}")
        
        if not frames_path.exists():
            return jsonify({'error': f'Frames file not found: {filename}'}), 404
            
        features = load_and_flatten(frames_path)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'jump_type': 'Unable to classify',
                'confidence': float(confidence),
                'message': f'Confidence ({confidence:.1%}) is too low.)',
                'all_probabilities': {class_name: float(prob) for i, (class_name, prob) in enumerate(zip(model.classes_, probabilities))},
                'top_predictions':[
                    {
                        'rank': i + 1,
                        'jump_type': class_name,
                        'probability': float(prob)
                    }
                    for i, (class_name, prob) in enumerate(sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3])
                ],
                'skater': skater,
                'event': event,
                'classification_attempted': True,
            })
        
        class_probs = {}
        for i, class_name in enumerate(model.classes_):
            class_probs[class_name] = float(probabilities[i])
        
        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {
                'rank': i + 1,
                'jump_type': class_name,
                'probability': float(prob)
            }
            for i, (class_name, prob) in enumerate(sorted_probs[:3])
        ]
        
        return jsonify({
            'jump_type': prediction,
            'confidence': float(confidence),
            'all_probabilities': class_probs,
            'top_predictions': top_predictions,
            'skater': skater,
            'event': event
        })
        
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500
    
@app.route('/api/upload-and-classify', methods=['POST'])
def upload_and_classify():
    """Upload video and immediately classify without saving video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only MP4, MOV, AVI, MKV allowed'}), 400
        
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)

        max_size = 100 * 1024 * 1024
        if file_size > max_size:
            return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413

        # get additional data from form 
        skater = request.form.get('skater', 'User')
        event = request.form.get('event', 'Upload')

        # generate unique filename
        unique_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)

        print(f"Processing and classifying video: {original_filename} ({file_size / 1024 / 1024:.1f}MB)")

        # read file into memory
        file_bytes = file.read()
        video_filename = f"{unique_id}.mp4"
        video_path = UPLOADS_FOLDER / video_filename

        with open(video_path, 'wb') as f:
            f.write(file_bytes)

        print(f"Saved video temporarily: {video_filename}")
        pose_data, video_info = process_video_from_bytes(file_bytes, original_filename)

        # save pose data temporarily for classification
        pose_filename = f"{unique_id}.npy"
        pose_path = FRAMES_FOLDER / pose_filename
        np.save(str(pose_path), pose_data)

        features = load_and_flatten(pose_path)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)

        class_probs = {}
        for i, class_name in enumerate(model.classes_):
            class_probs[class_name] = float(probabilities[i])

        sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [
            {
                'rank': i + 1,
                'jump_type': class_name,
                'probability': float(prob)
            }
            for i, (class_name, prob) in enumerate(sorted_probs[:3])
        ]

        result = {
            'success': True, 
            'jump_type': prediction, 
            'confidence': float(confidence), 
            'all_probabilities': class_probs, 
            'top_predictions': top_predictions,
            'skater': skater, 
            'event': event, 
            'video_id': unique_id, 
            'original_filename': original_filename, 
            'pose_filename': pose_filename, 
            'video_info': video_info,
            'file_size_mb': round(file_size / 1024 / 1024, 1)
        }

        if confidence < CONFIDENCE_THRESHOLD:
            result.update({
                'jump_type': 'Unable to classify', 
                'message': f'Confidence is too low',
                'classification_attempted': True,
            })

        print(f"Classification complete")
        return jsonify(result)
    
    except Exception as e: 
        print(f"Upload and classify error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()
print("🧹 Auto-cleanup thread started - files will be deleted after 1 hour")

@app.route('/api/video/<video_id>')
def serve_video(video_id):
    """Serve uploaded video files from the uploads folder"""
    try:
        # construct the video filename (assuming .mp4 extension)
        video_filename = f"{video_id}.mp4"
        
        # check if file exists in uploads folder
        video_path = UPLOADS_FOLDER / video_filename
        
        if not video_path.exists():
            print(f"Video not found: {video_path}")
            return jsonify({'error': f'Video not found: {video_filename}'}), 404
        
        print(f"Serving video: {video_filename}")
        
        # serve the video file with proper headers for video streaming
        return send_from_directory(
            str(UPLOADS_FOLDER),  # convert Path to string
            video_filename, 
            mimetype='video/mp4',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Error serving video {video_id}: {str(e)}")
        return jsonify({'error': f'Failed to serve video: {str(e)}'}), 500

# route added to list available videos (for debugging)
@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all available videos in uploads folder"""
    try:
        if not UPLOADS_FOLDER.exists():
            return jsonify({'videos': []})
        
        videos = []
        for video_file in UPLOADS_FOLDER.glob('*.mp4'):
            video_id = video_file.stem 
            videos.append({
                'video_id': video_id,
                'filename': video_file.name,
                'size_mb': round(video_file.stat().st_size / (1024 * 1024), 1),
                'created': video_file.stat().st_mtime
            })
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        print(f"Error listing videos: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)