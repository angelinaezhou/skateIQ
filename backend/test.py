import pickle
import numpy as np
from pathlib import Path

TEST_DIR = Path("data/examples")

MAX_FRAMES = 30
POSE_DIM = 132

def load_and_flatten(pose_file: Path, max_frames=MAX_FRAMES):
    """Same preprocessing as training script"""
    pose = np.load(pose_file)
    
    print(f"   Loaded shape: {pose.shape}")
    
    if len(pose) > max_frames:
        pose = pose[:max_frames]
    elif len(pose) < max_frames:
        padding = np.zeros((max_frames - len(pose), pose.shape[1]))
        pose = np.vstack([pose, padding])
    
    flattened = pose.flatten()
    print(f"   Flattened shape: {flattened.shape}")
    return flattened

def test_single_file(file_path: Path, model, scaler):
    """Test classification on a single file"""
    print(f"Testing: {file_path}")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    try:
        features = load_and_flatten(file_path)
        features_scaled = scaler.transform([features])
        print(f"   Scaled shape: {features_scaled.shape}")
        
        # make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = max(probabilities)
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        
        # Show all class probabilities
        print(f"All probabilities:")
        for i, class_name in enumerate(model.classes_):
            prob = probabilities[i]
            print(f"{class_name}: {prob:.3f}")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"Error testing {file_path.name}: {e}")
        return None, None


def test_directory(test_dir: Path, model, scaler, max_files=10):
    """Test classification on multiple files from a directory"""
    print(f"\Testing directory: {test_dir}")
    
    if not test_dir.exists():
        print(f"Directory not found: {test_dir}")
        return
    
    npy_files = list(test_dir.glob("*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {test_dir}")
        return
    
    print(f"Found {len(npy_files)} .npy files")
    
    test_files = npy_files[:max_files]
    results = []
    
    for file_path in test_files:
        pred, conf = test_single_file(file_path, model, scaler)
        if pred is not None:
            results.append((file_path.name, pred, conf))
    
    print(f"\n📈 Summary of {len(results)} tests:")
    for filename, pred, conf in results:
        print(f"   {filename}: {pred} ({conf:.3f})")

def main():
    """Main test function"""
    print("Model and Scaler Test Script")
    print("=" * 50)
    
    # load model and scaler
    try:
        print("Loading model...")
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded: {type(model).__name__}")
        print(f"Classes: {model.classes_}")
        
        print("Loading scaler...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded: {type(scaler).__name__}")
        print(f"Expected features: {scaler.n_features_in_}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure model.pkl and scaler.pkl are in the current directory")
        return
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # test a directory
    if TEST_DIR.exists():
        test_directory(TEST_DIR, model, scaler)
    else:
        print(f"Test directory not found: {TEST_DIR}")
        print("Update TEST_DIR path in the script")
    
    print("Testing completed!")

if __name__ == "__main__":
    main()