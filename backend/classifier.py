import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
MAX_FRAMES = 30
POSE_DIM = 132
TARGET_LEN = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

JUMP_TYPES = ['axel', 'salchow', 'toe_loop', 'loop', 'flip', 'lutz']

def load_and_flatten(pose_file: Path, max_frames=MAX_FRAMES):
    """Load pose data and flatten - same as Flask API preprocessing"""
    pose = np.load(pose_file)
    
    # ensure consistent shape
    if len(pose) > max_frames:
        pose = pose[:max_frames]
    elif len(pose) < max_frames:
        padding = np.zeros((max_frames - len(pose), pose.shape[1]))
        pose = np.vstack([pose, padding])
    
    return pose.flatten()

def load_dataset():
    """Load all pose data and create dataset"""
    X = []  # features (flattened pose sequences)
    y = []  # labels (jump types)
    filenames = []  # keep track of source files
    
    print("📂 Loading dataset...")
    
    for jump_type in JUMP_TYPES:
        jump_dir = DATA_DIR / jump_type
        
        if not jump_dir.exists():
            print(f"Warning: {jump_dir} doesn't exist, skipping...")
            continue
        
        # find all .npy files in this jump type folder
        npy_files = list(jump_dir.glob("*.npy"))
        
        if not npy_files:
            print(f"Warning: No .npy files found in {jump_dir}")
            continue
        
        print(f"   {jump_type}: {len(npy_files)} files")
        
        for npy_file in npy_files:
            try:
                # load and preprocess pose data
                features = load_and_flatten(npy_file)
                
                # verify shape
                expected_shape = MAX_FRAMES * POSE_DIM
                if len(features) != expected_shape:
                    print(f"Skipping {npy_file.name}: wrong shape {len(features)} != {expected_shape}")
                    continue
                
                X.append(features)
                y.append(jump_type)
                filenames.append(str(npy_file))
                
            except Exception as e:
                print(f"Error loading {npy_file.name}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\Dataset loaded:")
    print(f"   Total samples: {len(X)}")
    print(f"   Feature dimensions: {X.shape}")
    print(f"   Classes: {len(set(y))}")
    
    # show class distribution
    class_counts = Counter(y)
    for jump_type in JUMP_TYPES:
        count = class_counts.get(jump_type, 0)
        print(f"   {jump_type}: {count} samples")
    
    return X, y, filenames

def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning"""
    print("Training Random Forest Classifier...")
    
    # smaller parameter grid for faster tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print(f"   Training with {len(X_train)} samples using 5-fold CV...")
    grid_search.fit(X_train, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV score: {grid_search.best_score_:.3f}")
    
    # return the best model
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model performance"""
    print("\n📈 Model Performance:")
    
    # predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")
    
    # classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print(f"Confusion Matrix:")
    print(f"{' ':>10}", end="")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"   {true_name:>10}", end="")
        for j, pred_name in enumerate(class_names):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    return accuracy, y_pred, y_pred_proba

def save_model_and_scaler(model, scaler):
    """Save trained model and scaler"""
    print("Saving model and scaler...")
    
    # save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("model.pkl saved")
    
    # save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("scaler.pkl saved")
    
    # save model info
    model_info = {
        'model_type': type(model).__name__,
        'feature_shape': scaler.n_features_in_,
        'classes': list(model.classes_),
        'n_classes': len(model.classes_)
    }
    
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("model_info.pkl saved")

def main():
    """Main training pipeline"""
    print("Starting figure skating jump classification training...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Expected jump types: {JUMP_TYPES}")
    
    # check if data directory exists
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return
    
    # load dataset
    X, y, filenames = load_dataset()
    
    if len(X) == 0:
        print("No data found! Make sure .npy files exist in jump type folders.")
        return
    
    # check for minimum samples per class
    class_counts = Counter(y)
    min_samples = min(class_counts.values())
    if min_samples < 2:
        print(f"Warning: Some classes have very few samples (min: {min_samples})")
        print("Consider collecting more data for better performance.")
    
    # Split the data
    print(f"Splitting data (test size: {TEST_SIZE})...")
    print(f"Before split - X: {X.shape}, y: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
        stratify=y if min_samples >= 2 else None
    )
    
    print(f"After first split:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"After scaling:")
    print(f"X_train_scaled: {X_train_scaled.shape}")
    print(f"X_test_scaled: {X_test_scaled.shape}")
    
    # train model
    model = train_random_forest(X_train_scaled, y_train)
    
    # evaluate model
    class_names = sorted(list(set(y)))
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, class_names)
    
    # cross validation score
    print("Cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    save_model_and_scaler(model, scaler)
    
    print(f"Training completed!")
    print(f"Final test accuracy: {accuracy:.3f}")
    print(f"Model saved as 'model.pkl'")
    print(f"Scaler saved as 'scaler.pkl'")
    print("Your Flask API is ready to use these files!")

if __name__ == "__main__":
    main()