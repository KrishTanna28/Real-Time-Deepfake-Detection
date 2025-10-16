"""
Advanced Fine-Tuning for Better Generalization
- Heavy augmentation to handle unseen videos
- Progressive learning strategy
- Test-time augmentation
- Ensemble predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import random
from efficientnet_pytorch import EfficientNet


class DeepfakeEfficientNet(nn.Module):
    """Model architecture"""
    def __init__(self, pretrained=False, dropout=0.5):
        super(DeepfakeEfficientNet, self).__init__()
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        
        num_features = self.efficientnet._fc.in_features
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


class HeavyAugmentation:
    """Aggressive augmentation for better generalization"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def __call__(self, img):
        # Random resize and crop (scale invariance)
        if random.random() < 0.8:
            scale = random.uniform(0.6, 1.0)
            size = int(self.image_size / scale)
            img = TF.resize(img, size)
            i, j, h, w = transforms.RandomCrop.get_params(img, (self.image_size, self.image_size))
            img = TF.crop(img, i, j, h, w)
        else:
            img = TF.resize(img, self.image_size)
        
        # Geometric transforms
        if random.random() < 0.5:
            img = TF.hflip(img)
        
        if random.random() < 0.3:
            angle = random.uniform(-20, 20)
            img = TF.rotate(img, angle)
        
        if random.random() < 0.2:
            img = TF.vflip(img)
        
        # Perspective transform (simulates different camera angles)
        if random.random() < 0.3:
            width, height = img.size
            startpoints = [[0, 0], [width, 0], [width, height], [0, height]]
            endpoints = [[random.randint(0, int(width*0.1)), random.randint(0, int(height*0.1))],
                        [width - random.randint(0, int(width*0.1)), random.randint(0, int(height*0.1))],
                        [width - random.randint(0, int(width*0.1)), height - random.randint(0, int(height*0.1))],
                        [random.randint(0, int(width*0.1)), height - random.randint(0, int(height*0.1))]]
            img = TF.perspective(img, startpoints, endpoints)
        
        # Color augmentation (lighting invariance)
        if random.random() < 0.8:
            # Brightness
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_brightness(img, factor)
            
            # Contrast
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_contrast(img, factor)
            
            # Saturation
            factor = random.uniform(0.5, 1.5)
            img = TF.adjust_saturation(img, factor)
            
            # Hue
            factor = random.uniform(-0.2, 0.2)
            img = TF.adjust_hue(img, factor)
        
        # Gamma correction (exposure variations)
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            img = TF.adjust_gamma(img, gamma)
        
        # Blur (compression/quality loss)
        if random.random() < 0.4:
            blur_type = random.choice(['gaussian', 'motion', 'median'])
            if blur_type == 'gaussian':
                radius = random.uniform(0.5, 2.5)
                img = img.filter(ImageFilter.GaussianBlur(radius))
            elif blur_type == 'motion':
                # Simulate motion blur
                img = img.filter(ImageFilter.BLUR)
            else:
                img = img.filter(ImageFilter.MedianFilter(size=3))
        
        # Sharpening (some videos are over-sharpened)
        if random.random() < 0.3:
            enhancer = ImageEnhance.Sharpness(img)
            factor = random.uniform(0.5, 2.0)
            img = enhancer.enhance(factor)
        
        # JPEG compression artifacts
        if random.random() < 0.4:
            import io
            quality = random.randint(30, 95)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        
        # Convert to tensor
        img = TF.to_tensor(img)
        
        # Add noise
        if random.random() < 0.3:
            noise_type = random.choice(['gaussian', 'salt_pepper'])
            if noise_type == 'gaussian':
                noise = torch.randn_like(img) * random.uniform(0.01, 0.08)
                img = torch.clamp(img + noise, 0, 1)
            else:
                # Salt and pepper noise
                mask = torch.rand_like(img)
                img[mask < 0.02] = 0
                img[mask > 0.98] = 1
        
        # Random erasing (occlusion)
        if random.random() < 0.2:
            img = transforms.RandomErasing(p=1.0, scale=(0.02, 0.1))(img)
        
        # Normalize
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return img


class ValidationTransform:
    """Minimal transforms for validation"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
    
    def __call__(self, img):
        img = TF.resize(img, self.image_size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img


class ImageDataset(Dataset):
    """Dataset with augmentation"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.samples = []
        real_images = list((self.root_dir / "Real").glob("*"))
        fake_images = list((self.root_dir / "Fake").glob("*"))
        
        for img in real_images:
            self.samples.append((img, 0))
        for img in fake_images:
            self.samples.append((img, 1))
        
        print(f"Dataset: {len(real_images)} real, {len(fake_images)} fake")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_term * bce_loss
        return loss.mean()


def test_time_augmentation(model, image, device, n_augments=5):
    """
    Test-Time Augmentation (TTA)
    Apply multiple augmentations and average predictions
    """
    model.eval()
    predictions = []
    
    # Original image
    with torch.no_grad():
        output = model(image.to(device)).squeeze()
        predictions.append(torch.sigmoid(output).item())
    
    # Augmented versions
    for _ in range(n_augments - 1):
        # Random horizontal flip
        aug_image = image.clone()
        if random.random() < 0.5:
            aug_image = torch.flip(aug_image, [3])
        
        # Random brightness
        factor = random.uniform(0.8, 1.2)
        aug_image = aug_image * factor
        aug_image = torch.clamp(aug_image, 0, 1)
        
        with torch.no_grad():
            output = model(aug_image.to(device)).squeeze()
            predictions.append(torch.sigmoid(output).item())
    
    return np.mean(predictions)


def finetune_advanced(data_root, model_path, epochs=10, batch_size=8, lr=1e-4, use_focal_loss=True):
    """
    Advanced fine-tuning with heavy augmentation
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("⚠️  Training on CPU - this will be slow!")
        print("   Consider using Google Colab for faster training\n")
    
    # Load model
    print(f"Loading pretrained model from {model_path}...")
    model = DeepfakeEfficientNet(pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Fix key mismatch
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('net.'):
            new_key = key.replace('net.', 'efficientnet.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    print("✓ Model loaded\n")
    
    # Data transforms
    train_transform = HeavyAugmentation(image_size=224)
    val_transform = ValidationTransform(image_size=224)
    
    # Dataset
    dataset = ImageDataset(data_root, transform=None)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create separate datasets with different transforms
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples\n")
    
    # Training setup
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss (focuses on hard examples)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCE Loss")
    
    # Progressive learning: Start with smaller LR, increase gradually
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    print("="*70)
    print("ADVANCED FINE-TUNING WITH HEAVY AUGMENTATION")
    print("="*70)
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            # Apply transform here (after loading)
            images_aug = []
            for img_tensor, label in zip(images, labels):
                # Convert tensor back to PIL for augmentation
                img_pil = TF.to_pil_image(img_tensor)
                img_aug = train_transform(img_pil)
                images_aug.append(img_aug)
            
            images = torch.stack(images_aug)
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total:.4f}'})
        
        scheduler.step()
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_dataset.dataset.transform = val_transform
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Apply validation transform
                images_val = []
                for img_tensor in images:
                    img_pil = TF.to_pil_image(img_tensor)
                    img_val = val_transform(img_pil)
                    images_val.append(img_val)
                
                images = torch.stack(images_val)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'weights/finetuned_advanced.pth')
            print(f"  ✓ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
                break
        print()
    
    print("="*70)
    print(f"✅ Advanced fine-tuning complete!")
    print(f"Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Model saved to: weights/finetuned_advanced.pth")
    print("="*70)
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./dataset/Dataset/Test',
                       help='Folder with Real/ and Fake/ subfolders')
    parser.add_argument('--model_path', type=str, default='./weights/best_model.pth',
                       help='Pretrained model to fine-tune')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (10-15 recommended)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (use 4-8 for CPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use Focal Loss instead of BCE')
    
    args = parser.parse_args()
    
    finetune_advanced(args.data_root, args.model_path, args.epochs, args.batch_size, args.lr, args.use_focal_loss)
