#!/usr/bin/env python3
"""
Simple Video to NPY Pose Converter
Based on your original script with improvements and fixes
"""

import os
import numpy as np
import cv2
from pathlib import Path
import mediapipe as mp
import time

# === Config ===
DATA_DIR = Path(__file__).parent / "data" / "lutz" / 'h'
TARGET_LEN = 30  # number of frames to resample to
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

# === Initialize MediaPipe Pose ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

# === Functions ===
def center_landmarks(landmarks):
    """Center pose landmarks around the midpoint of the hips."""
    try:
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][:3]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][:3]
        center = (left_hip + right_hip) / 2
        centered = landmarks[:, :3] - center  # Ignore visibility for centering
        return np.concatenate([centered, landmarks[:, 3:4]], axis=1)  # Add back visibility
    except Exception as e:
        print(f"⚠️ Warning: Could not center landmarks: {e}")
        return landmarks

def resample_pose_sequence(sequence, target_len):
    """Resample a pose sequence to fixed length."""
    sequence = np.array(sequence)
    original_len = len(sequence)
    
    if original_len == target_len:
        return sequence
    
    if original_len < 2:
        if original_len == 1:
            # Duplicate the single frame
            return np.tile(sequence[0], (target_len, 1))
        else:
            raise ValueError(f"Not enough frames to resample: {original_len}")
    
    # Simple linear interpolation
    indices = np.linspace(0, original_len - 1, target_len)
    resampled = np.zeros((target_len, sequence.shape[1]))
    
    for i in range(target_len):
        idx = indices[i]
        lower_idx = int(np.floor(idx))
        upper_idx = min(int(np.ceil(idx)), original_len - 1)
        
        if lower_idx == upper_idx:
            resampled[i] = sequence[lower_idx]
        else:
            # Linear interpolation
            alpha = idx - lower_idx
            resampled[i] = (1 - alpha) * sequence[lower_idx] + alpha * sequence[upper_idx]
    
    return resampled

def process_video_to_npy(video_path: Path):
    """Convert one video to a pose .npy file saved in the same folder."""
    print(f"🎥 Processing: {video_path.name}")
    start_time = time.time()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open {video_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"   📊 {fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
    
    all_landmarks = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
            centered = center_landmarks(landmarks)
            all_landmarks.append(centered.flatten())
        
        frame_count += 1
        
        # Progress for long videos
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"   ⏳ {progress:.1f}% ({len(all_landmarks)} poses)")
    
    cap.release()
    
    pose_rate = len(all_landmarks) / frame_count if frame_count > 0 else 0
    print(f"   ✅ {len(all_landmarks)}/{frame_count} frames with pose ({pose_rate:.1%})")
    
    if len(all_landmarks) < 2:
        print(f"⚠️ Skipping {video_path.name}: not enough pose frames")
        return
    
    pose_seq = np.array(all_landmarks)
    resampled = resample_pose_sequence(pose_seq, TARGET_LEN)
    
    npy_path = video_path.with_suffix('.npy')
    np.save(npy_path, resampled)
    
    processing_time = time.time() - start_time
    print(f"   💾 Saved: {npy_path.name} {resampled.shape} ({processing_time:.1f}s)")

def process_all_videos():
    """Process all videos in the data directory."""
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        print(f"   Please create the directory or put your videos there")
        return
    
    print(f"📁 Looking for videos in: {DATA_DIR}")
    
    video_files = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                video_files.append(file_path)
    
    if not video_files:
        print(f"⚠️ No video files found in {DATA_DIR}")
        print(f"   Supported formats: {', '.join(ALLOWED_EXTENSIONS)}")
        return
    
    print(f"🔍 Found {len(video_files)} video files")
    
    successful = 0
    for i, file_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}]")
        try:
            process_video_to_npy(file_path)
            successful += 1
        except Exception as e:
            print(f"❌ Error in {file_path.name}: {e}")
    
    print(f"\n🎉 Done! Processed {successful}/{len(video_files)} videos successfully")

# === Run the whole thing ===
if __name__ == "__main__":
    print("🚀 Starting pose extraction...")
    print(f"   Target frames: {TARGET_LEN}")
    print(f"   Data directory: {DATA_DIR}")
    
    try:
        process_all_videos()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("✨ Script completed!")