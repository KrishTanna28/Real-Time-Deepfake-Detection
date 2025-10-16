import cv2
import numpy as np
import mss
import mss.tools
from face_detection import detect_bounding_box
from deepfake_detection import predict

def select_region_manual(monitor):
    """Manual region selection when GUI is not available"""
    print("\n=== Manual Region Input ===")
    print(f"Screen resolution: {monitor['width']}x{monitor['height']}")
    print("\nEnter the coordinates for the video region:")
    print("(You can use a screenshot tool to find coordinates)")
    
    try:
        left = int(input("Left (x start, e.g., 100): "))
        top = int(input("Top (y start, e.g., 100): "))
        width = int(input("Width (e.g., 640): "))
        height = int(input("Height (e.g., 480): "))
        
        # Validate inputs
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            print("Invalid coordinates!")
            return None
        
        if left + width > monitor['width'] or top + height > monitor['height']:
            print("Region exceeds screen boundaries!")
            return None
        
        selected_region = {
            "top": monitor["top"] + top,
            "left": monitor["left"] + left,
            "width": width,
            "height": height
        }
        
        print(f"\n✓ Selected region: {selected_region}")
        confirm = input("Use this region? (y/n): ").strip().lower()
        if confirm == 'y':
            return selected_region
        else:
            print("Region selection cancelled.")
            return None
            
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid input or cancelled.")
        return None

def select_screen_region():
    """Allow user to select a region of the screen to capture"""
    print("=== Screen Region Selection ===")
    print("Instructions:")
    print("1. A screenshot of your entire screen will appear")
    print("2. Click and drag to select the region where the video is playing")
    print("3. Press ENTER to confirm, or 'c' to cancel and reselect")
    print("=" * 35)
    
    with mss.mss() as sct:
        # Capture the entire screen
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Try GUI selection first
        try:
            # Let user select ROI (Region of Interest)
            clone = img.copy()
            cv2.namedWindow("Select Video Region", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select Video Region", 1280, 720)
            
            roi = cv2.selectROI("Select Video Region", clone, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Video Region")
            
            if roi[2] == 0 or roi[3] == 0:
                print("No region selected. Exiting...")
                return None
            
            # Create monitor dict for the selected region
            selected_region = {
                "top": monitor["top"] + int(roi[1]),
                "left": monitor["left"] + int(roi[0]),
                "width": int(roi[2]),
                "height": int(roi[3])
            }
            
            print(f"Selected region: {selected_region}")
            return selected_region
            
        except cv2.error as e:
            print(f"\n⚠️ OpenCV GUI not available: {e}")
            print("\nFalling back to manual region input...")
            return select_region_manual(monitor)

def analyze_frames_and_classify(sct, region, num_frames=30):
    """
    Capture and analyze a specified number of frames to classify the video
    
    Args:
        sct: mss screen capture object
        region: screen region to capture
        num_frames: number of frames to analyze (default: 30)
    
    Returns:
        tuple: (final_classification, confidence, fake_percentage)
    """
    predictions = []
    confidences = []
    frames_with_faces = 0
    
    print(f"\nAnalyzing {num_frames} frames...")
    
    for i in range(num_frames):
        # Capture the selected screen region
        screenshot = sct.grab(region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Detect faces
        faces = detect_bounding_box(frame)
        
        if len(faces) > 0:
            frames_with_faces += 1
            # Get prediction and confidence from the frame
            from deepfake_detection import mtcnn, model, DEVICE
            import torch
            import torch.nn.functional as F
            from PIL import Image
            
            for (x, y, w, h) in faces:
                face_region = frame[y:y + h, x:x + w]
                input_face = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                input_face = mtcnn(input_face)
                
                if input_face is not None:
                    input_face = input_face.unsqueeze(0)
                    # EfficientNet-B0 expects 224x224 input
                    input_face = F.interpolate(input_face, size=(224, 224), mode="bilinear", align_corners=False)
                    input_face = input_face.to(DEVICE).to(torch.float32) / 255.0
                    
                    # Normalize using ImageNet statistics (EfficientNet pretrained on ImageNet)
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)
                    input_face = (input_face - mean) / std
                    
                    with torch.no_grad():
                        # Get raw logit output
                        raw_output = model(input_face).squeeze(0)
                        # Apply sigmoid to get probability
                        output = torch.sigmoid(raw_output)
                        
                        # Output represents FAKE probability
                        fake_confidence = output.item()
                        real_confidence = 1.0 - fake_confidence
                        
                        # Debug output for first few frames to understand model behavior
                        if i < 5:
                            print(f"  Frame {i+1}: Logit={raw_output.item():.3f}, Fake={fake_confidence:.3f}, Real={real_confidence:.3f}")
                        
                        # Classify based on fake confidence
                        prediction = "Fake" if fake_confidence >= 0.5 else "Real"
                        
                        predictions.append(prediction)
                        confidences.append(fake_confidence)
                        break  # Only analyze first face per frame
        
        # Show progress
        progress = int((i + 1) / num_frames * 100)
        print(f"Progress: {progress}% ({i + 1}/{num_frames} frames)", end='\r')
    
    print()  # New line after progress
    
    if len(predictions) == 0:
        return None, 0.0, 0.0
    
    # Calculate statistics
    fake_count = predictions.count("Fake")
    real_count = predictions.count("Real")
    fake_percentage = (fake_count / len(predictions)) * 100
    avg_confidence = sum(confidences) / len(confidences)
    
    # Classify video based on majority voting
    final_classification = "FAKE" if fake_count > real_count else "REAL"
    
    return final_classification, avg_confidence, fake_percentage

def main():
    print("=== Screen Video Deepfake Detection ===")
    print("This tool will capture frames from a selected screen region")
    print("and classify the video as fake or real.\n")
    
    # Get number of frames to analyze
    try:
        num_frames = int(input("Enter number of frames to analyze (default 30): ") or "30")
        if num_frames <= 0:
            print("Invalid number. Using default: 30")
            num_frames = 30
    except ValueError:
        print("Invalid input. Using default: 30")
        num_frames = 30
    
    # Let user select screen region
    region = select_screen_region()
    if region is None:
        return
    
    print(f"\nWill analyze {num_frames} frames from the selected region.")
    input("Press ENTER to start analysis...")
    
    with mss.mss() as sct:
        result, confidence, fake_percentage = analyze_frames_and_classify(sct, region, num_frames)
        
        if result is None:
            print("\n❌ No faces detected in the captured frames!")
            print("Make sure a video with visible faces is playing in the selected region.")
        else:
            print("\n" + "=" * 50)
            print("ANALYSIS COMPLETE")
            print("=" * 50)
            print(f"Classification: {result}")
            print(f"Average Fake Confidence: {confidence * 100:.2f}%")
            print(f"Fake Detection Rate: {fake_percentage:.2f}%")
            print(f"Real Detection Rate: {100 - fake_percentage:.2f}%")
            print("=" * 50)
            
            if result == "FAKE":
                print("\n⚠️  WARNING: This video is classified as a DEEPFAKE!")
            else:
                print("\n✓ This video appears to be REAL.")

if __name__ == "__main__":
    main()
