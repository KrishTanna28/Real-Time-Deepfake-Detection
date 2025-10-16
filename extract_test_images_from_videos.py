"""
Extract frames from your videos to create test images
This lets us test the model on image-level (like it was trained)
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_frames_to_images(video_folder, output_folder, frames_per_video=50):
    """
    Extract frames from videos and save as images
    
    Args:
        video_folder: Folder with real_videos/ and fake_videos/
        output_folder: Where to save extracted images
        frames_per_video: Number of frames to extract per video
    """
    
    video_path = Path(video_folder)
    output_path = Path(output_folder)
    
    # Create output directories
    (output_path / "Real").mkdir(parents=True, exist_ok=True)
    (output_path / "Fake").mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EXTRACTING FRAMES FROM VIDEOS")
    print("="*70)
    
    # Process real videos
    real_videos = list((video_path / "real_videos").glob("*.mp4"))
    print(f"\nFound {len(real_videos)} real videos")
    
    frame_count = 0
    for video in tqdm(real_videos, desc="Real videos"):
        cap = cv2.VideoCapture(str(video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            continue
        
        # Extract evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, min(frames_per_video, total_frames), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Save frame
                output_file = output_path / "Real" / f"real_{video.stem}_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(output_file), frame)
                frame_count += 1
        
        cap.release()
    
    print(f"✓ Extracted {frame_count} real frames")
    
    # Process fake videos
    fake_videos = list((video_path / "fake_videos").glob("*.mp4"))
    print(f"\nFound {len(fake_videos)} fake videos")
    
    frame_count = 0
    for video in tqdm(fake_videos, desc="Fake videos"):
        cap = cv2.VideoCapture(str(video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            continue
        
        frame_indices = np.linspace(0, total_frames - 1, min(frames_per_video, total_frames), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                output_file = output_path / "Fake" / f"fake_{video.stem}_frame_{frame_count:04d}.jpg"
                cv2.imwrite(str(output_file), frame)
                frame_count += 1
        
        cap.release()
    
    print(f"✓ Extracted {frame_count} fake frames")
    
    print(f"\n✅ Test images created at: {output_path}")
    print(f"   Real: {len(list((output_path / 'Real').glob('*')))} images")
    print(f"   Fake: {len(list((output_path / 'Fake').glob('*')))} images")
    print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default='./dataset/raw',
                       help='Folder with real_videos/ and fake_videos/')
    parser.add_argument('--output_folder', type=str, default='./dataset/Dataset/Test',
                       help='Where to save extracted images')
    parser.add_argument('--frames_per_video', type=int, default=50,
                       help='Frames to extract per video')
    
    args = parser.parse_args()
    
    extract_frames_to_images(args.video_folder, args.output_folder, args.frames_per_video)
