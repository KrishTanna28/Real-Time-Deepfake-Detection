import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np 
import cv2
from collections import deque
import time

from face_detection import detect_bounding_box

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize models
mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

# Create EfficientNet-B0 model with custom classifier for binary deepfake detection
class DeepfakeEfficientNet(nn.Module):
    """EfficientNet-B0 backbone with binary classification head"""
    def __init__(self, pretrained=True, dropout=0.5):
        super(DeepfakeEfficientNet, self).__init__()
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        
        # Get the number of features from the last layer
        num_features = self.efficientnet._fc.in_features
        
        # Replace the classifier with GENERALIZED architecture
        # This matches the train_generalized_colab.py model
        # More layers with BatchNorm for better generalization
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.efficientnet(x)
    
    def get_feature_extractor(self):
        """Get the last convolutional layer for GradCAM"""
        return self.efficientnet._conv_head

# Initialize model
print("Initializing EfficientNet-B0 for deepfake detection...")
model = DeepfakeEfficientNet(pretrained=True)

# Load trained deepfake detection weights if available
import os

weights_paths = [
    os.path.join(os.path.dirname(__file__), "weights", "best_model.pth")
]

model_loaded = False
for weights_path in weights_paths:
    if os.path.exists(weights_path):
        print(f"Loading trained model from {weights_path}")
        try:
            checkpoint = torch.load(weights_path, map_location=DEVICE)
            # Handle potential state dict key mismatches
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # Fix key mismatch (net. -> efficientnet.)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('net.'):
                    new_key = key.replace('net.', 'efficientnet.')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            print("✓ Trained model loaded successfully")
            model_loaded = True
            break
        except Exception as e:
            print(f"⚠️  Warning: Could not load {weights_path}: {e}")
            continue

if not model_loaded:
    print(f"⚠️  Warning: No trained model found")
    print("Using pretrained ImageNet weights from EfficientNet-B0")
    print("NOTE: Model needs to be retrained for optimal deepfake detection")

model.to(DEVICE)
model.eval()


class TemporalTracker:
    """Layer 2: Temporal Consistency Tracker for stable predictions"""
    
    def __init__(self, window_size=60, high_confidence_threshold=0.75):
        """
        Args:
            window_size: Number of frames to track (60 frames ~ 2 seconds at 30fps)
            high_confidence_threshold: Threshold for high confidence detection
        """
        self.window_size = window_size
        self.high_confidence_threshold = high_confidence_threshold
        self.score_history = deque(maxlen=window_size)
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
    def update(self, fake_probability):
        """Update with new frame's fake probability"""
        self.score_history.append(fake_probability)
        
    def get_temporal_average(self):
        """Get running average of fake probability"""
        if len(self.score_history) == 0:
            return 0.0
        return sum(self.score_history) / len(self.score_history)
    
    def get_stability_score(self):
        """Calculate how stable/consistent the predictions are (lower variance = more stable)"""
        if len(self.score_history) < 10:
            return 0.0
        scores = list(self.score_history)
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return 1.0 - min(variance * 4, 1.0)  # Normalize to 0-1, higher is more stable
    
    def should_trigger_forensic_analysis(self):
        """Determine if we should trigger Layer 3 (Gemini) analysis"""
        if len(self.score_history) < self.window_size // 2:
            return False
            
        avg_score = self.get_temporal_average()
        stability = self.get_stability_score()
        current_time = time.time()
        
        # Trigger if: high average fake score + stable predictions + cooldown passed
        if (avg_score > self.high_confidence_threshold and 
            stability > 0.7 and 
            current_time - self.last_alert_time > self.alert_cooldown):
            self.last_alert_time = current_time
            return True
        return False
    
    def get_confidence_level(self):
        """Get confidence level: 'HIGH_FAKE' or 'HIGH_REAL'"""
        avg = self.get_temporal_average()
        
        # Simple binary classification based on average
        # If fake probability > 50%, classify as FAKE, otherwise REAL
        if avg >= 0.5:
            return 'FAKE'
        else:
            return 'REAL'
    
    def reset(self):
        """Reset the tracker"""
        self.score_history.clear()


class DeepfakeDetector:
    """3-Layer Deepfake Detection System"""
    
    def __init__(self, enable_gradcam=False):
        self.enable_gradcam = enable_gradcam
        self.temporal_tracker = TemporalTracker(window_size=60, high_confidence_threshold=0.75)
        self.frame_count = 0
        
    def preprocess_face_quality(self, face_region):
        """Adaptive preprocessing to handle different video qualities"""
        # Convert to grayscale for quality assessment
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate image quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Adaptive preprocessing based on quality
        processed = face_region.copy()
        
        # If image is blurry (low laplacian variance), apply sharpening
        if laplacian_var < 100:
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        # If image is very noisy or low quality, apply denoising
        if laplacian_var < 50:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
        
        # Normalize contrast and brightness using CLAHE
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
        
        # Resize to consistent size before MTCNN (helps with varying resolutions)
        target_size = 256
        h, w = processed.shape[:2]
        if h < target_size or w < target_size:
            # Upscale small faces
            scale = max(target_size / h, target_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        elif h > target_size * 2 or w > target_size * 2:
            # Downscale very large faces
            scale = min(target_size * 2 / h, target_size * 2 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return processed
    
    def analyze_face(self, face_region):
        """Layer 1: Per-frame analysis using InceptionResnetV1"""
        try:
            # Apply adaptive quality preprocessing
            preprocessed = self.preprocess_face_quality(face_region)
            
            # Preprocess face
            input_face = Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
            input_face = mtcnn(input_face)
            
            if input_face is None:
                return None, None, None
            
            input_face = input_face.unsqueeze(0)
            # EfficientNet-B0 expects 224x224 input
            input_face = F.interpolate(input_face, size=(224, 224), mode="bilinear", align_corners=False)
            input_face = input_face.to(DEVICE).to(torch.float32) / 255.0
            
            # Normalize using ImageNet statistics (EfficientNet pretrained on ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
            input_face = (input_face - mean) / std
            
            # Get prediction
            with torch.no_grad():
                logit = model(input_face).squeeze(0)
                # Apply sigmoid to convert logit to probability
                output = torch.sigmoid(logit)
                # Output represents FAKE probability
                fake_probability = output.item()
            
            # Optional: Generate GradCAM visualization
            gradcam_img = None
            if self.enable_gradcam:
                try:
                    # Use the last convolutional layer of EfficientNet
                    target_layers = [model.get_feature_extractor()]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(0)]
                    grayscale_cam = cam(input_tensor=input_face, targets=targets, eigen_smooth=True)
                    grayscale_cam = grayscale_cam[0, :]
                    # Denormalize for visualization
                    vis_img = input_face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    vis_img = vis_img * std + mean
                    vis_img = np.clip(vis_img, 0, 1)
                    gradcam_img = show_cam_on_image(
                        vis_img,
                        grayscale_cam,
                        use_rgb=True
                    )
                except Exception as e:
                    print(f"GradCAM error: {e}")
            
            return fake_probability, output.item(), gradcam_img
            
        except Exception as e:
            print(f"Face analysis error: {e}")
            return None, None, None
    
    def get_box_color(self, confidence_level):
        """Get color based on temporal confidence level"""
        if confidence_level == 'HIGH_FAKE':
            return (0, 0, 255)  # Red
        elif confidence_level == 'HIGH_REAL':
            return (0, 255, 0)  # Green
        else:
            return (0, 255, 255)  # Yellow (uncertain)
    
    def draw_detection_overlay(self, frame, x, y, w, h, fake_prob, confidence_level):
        """Draw enhanced detection overlay with color-coded boxes and info"""
        color = self.get_box_color(confidence_level)
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Prepare text
        temporal_avg = self.temporal_tracker.get_temporal_average()
        stability = self.temporal_tracker.get_stability_score()
        
        # Main label
        if confidence_level == 'HIGH_FAKE':
            label = f"FAKE: {fake_prob*100:.1f}%"
        elif confidence_level == 'HIGH_REAL':
            label = f"REAL: {(1-fake_prob)*100:.1f}%"
        else:
            label = f"UNCERTAIN: {fake_prob*100:.1f}%"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw temporal info (smaller text below box)
        if len(self.temporal_tracker.score_history) >= 10:
            temporal_info = f"Avg:{temporal_avg*100:.0f}% Stab:{stability*100:.0f}%"
            cv2.putText(frame, temporal_info, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def predict(self, frame):
        """Main prediction function with 3-layer analysis"""
        self.frame_count += 1
        
        # Detect faces
        faces = detect_bounding_box(frame)
        
        trigger_forensic = False
        forensic_frame = None
        
        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            
            # Layer 1: Per-frame analysis
            fake_prob, real_score, gradcam = self.analyze_face(face_region)
            
            if fake_prob is None:
                continue
            
            # Layer 2: Update temporal tracker
            self.temporal_tracker.update(fake_prob)
            confidence_level = self.temporal_tracker.get_confidence_level()
            
            # Check if we should trigger Layer 3
            if self.temporal_tracker.should_trigger_forensic_analysis():
                trigger_forensic = True
                forensic_frame = frame.copy()
            
            # Draw overlay
            frame = self.draw_detection_overlay(frame, x, y, w, h, fake_prob, confidence_level)
            
            # Print detailed info every 30 frames
            if self.frame_count % 30 == 0:
                print(f"Frame {self.frame_count} | Fake: {fake_prob*100:.1f}% | "
                      f"Temporal Avg: {self.temporal_tracker.get_temporal_average()*100:.1f}% | "
                      f"Confidence: {confidence_level}")
        
        return frame, trigger_forensic, forensic_frame


# Global detector instance
detector = DeepfakeDetector(enable_gradcam=False)


def predict(frame):
    """Legacy function for backward compatibility"""
    result_frame, _, _ = detector.predict(frame)
    return result_frame


def predict_with_forensics(frame):
    """Enhanced prediction with forensic trigger info"""
    return detector.predict(frame)

