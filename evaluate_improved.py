"""
Improved evaluation with adaptive classification logic
Handles model calibration issues
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import torchvision.transforms as transforms

from deepfake_detection import DeepfakeEfficientNet
from improved_classification import ImprovedClassifier, classify_with_adaptive_threshold


def extract_frames_from_video(video_path, num_frames=30):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def evaluate_with_improved_logic(model_path, dataset_root, frames_per_video=30, method='ensemble'):
    """
    Evaluate model with improved classification logic
    
    Args:
        model_path: Path to model checkpoint
        dataset_root: Root directory with real_videos/ and fake_videos/
        frames_per_video: Frames to extract per video
        method: 'ensemble', 'adaptive', 'majority', or 'statistical'
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = DeepfakeEfficientNet(pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # Fix key mismatch: rename 'net.' to 'efficientnet.'
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('net.'):
            new_key = key.replace('net.', 'efficientnet.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Find videos
    dataset_path = Path(dataset_root)
    real_videos = list((dataset_path / 'real_videos').glob('*.mp4'))
    fake_videos = list((dataset_path / 'fake_videos').glob('*.mp4'))
    
    print(f"\nFound {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos")
    print(f"Classification method: {method.upper()}\n")
    
    if len(real_videos) == 0 and len(fake_videos) == 0:
        print("❌ No videos found!")
        return
    
    # Initialize classifier
    classifier = ImprovedClassifier()
    
    # Process all videos
    all_predictions = []
    all_labels = []
    all_confidences = []
    video_details = []
    
    print("Processing videos...")
    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]
    
    for video_path, label in tqdm(all_videos):
        # Extract frames
        frames = extract_frames_from_video(video_path, frames_per_video)
        if len(frames) == 0:
            continue
        
        # Get predictions for all frames
        frame_probs = []
        with torch.no_grad():
            for frame in frames:
                pil_img = Image.fromarray(frame)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                
                output = model(tensor).squeeze()
                prob = torch.sigmoid(output).item()
                frame_probs.append(prob)
        
        # Classify video using selected method
        if method == 'ensemble':
            is_fake, confidence, info = classifier.classify_video_ensemble(frame_probs)
        elif method == 'majority':
            is_fake, confidence, info = classifier.classify_video_majority(frame_probs)
        elif method == 'statistical':
            is_fake, confidence, info = classifier.classify_video_statistical(frame_probs)
        elif method == 'adaptive':
            is_fake, confidence = classify_with_adaptive_threshold(frame_probs)
            info = {'mean_prob': np.mean(frame_probs)}
        else:
            # Default: simple threshold
            mean_prob = np.mean(frame_probs)
            is_fake = mean_prob > 0.5
            confidence = abs(mean_prob - 0.5) * 2
            info = {'mean_prob': mean_prob}
        
        all_predictions.append(int(is_fake))
        all_labels.append(label)
        all_confidences.append(confidence)
        
        video_details.append({
            'path': video_path.name,
            'true_label': 'FAKE' if label == 1 else 'REAL',
            'prediction': 'FAKE' if is_fake else 'REAL',
            'confidence': confidence,
            'mean_prob': np.mean(frame_probs),
            'std_prob': np.std(frame_probs),
            'info': info
        })
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print("\n" + "="*70)
    print("IMPROVED EVALUATION RESULTS")
    print("="*70)
    print(f"Method: {method.upper()}")
    print(f"Videos: {len(real_videos)} real, {len(fake_videos)} fake")
    print("-"*70)
    
    print(f"\n### VIDEO-LEVEL METRICS ###")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Real    Fake")
    print(f"Actual Real   {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Fake   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    print(f"\nTrue Negatives (Real correctly classified):  {cm[0,0]}")
    print(f"False Positives (Real classified as Fake):   {cm[0,1]}")
    print(f"False Negatives (Fake classified as Real):   {cm[1,0]}")
    print(f"True Positives (Fake correctly classified):  {cm[1,1]}")
    
    print(f"\n### CONFIDENCE STATISTICS ###")
    print(f"Mean confidence: {np.mean(all_confidences):.4f}")
    print(f"Std confidence:  {np.std(all_confidences):.4f}")
    
    # Show per-video details
    print(f"\n### PER-VIDEO DETAILS ###")
    print(f"{'Video':<30} {'True':<6} {'Pred':<6} {'Conf':<6} {'Mean Prob':<10}")
    print("-"*70)
    for detail in video_details:
        correct = "✓" if detail['true_label'] == detail['prediction'] else "✗"
        print(f"{detail['path']:<30} {detail['true_label']:<6} {detail['prediction']:<6} "
              f"{detail['confidence']:.3f}  {detail['mean_prob']:.4f}  {correct}")
    
    print("="*70)
    
    # Recommendations
    print("\n### RECOMMENDATIONS ###")
    if accuracy < 0.6:
        print("⚠️  Accuracy still low. Try:")
        print("   1. Use method='ensemble' for best results")
        print("   2. Retrain model on more diverse data")
        print("   3. Collect some of your videos for fine-tuning")
    elif accuracy < 0.8:
        print("✓ Decent performance. Can improve by:")
        print("   1. Fine-tuning on your specific video sources")
        print("   2. Adjusting confidence thresholds")
    else:
        print("✅ Good performance!")
    
    print("="*70)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'video_details': video_details
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate with improved classification logic')
    parser.add_argument('--model_path', type=str, default='./weights/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset_root', type=str, default='./dataset/raw',
                       help='Root directory with real_videos/ and fake_videos/')
    parser.add_argument('--frames', type=int, default=30,
                       help='Number of frames to extract per video')
    parser.add_argument('--method', type=str, default='ensemble',
                       choices=['ensemble', 'adaptive', 'majority', 'statistical', 'simple'],
                       help='Classification method')
    
    args = parser.parse_args()
    
    evaluate_with_improved_logic(
        args.model_path,
        args.dataset_root,
        args.frames,
        args.method
    )
