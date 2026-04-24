import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased" # Switched to an open multilingual model to avoid login
MAX_SEQ_LEN = 128
NUM_FRAMES = 8
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

DATA_DIRS = [
    {
        "excel": "Corporate.xlsx",
        "video_dir": "Corporate_Video/Neetish_CorporateScene"
    },
    {
        "excel": "Grocery.xlsx",
        "video_dir": "Grocery_Video/Neetish"
    }
]

# --- 1. Pre-compute Text Embeddings ---
def precompute_text_embeddings():
    print(f"Loading tokenizer and text model ({TEXT_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
    text_model.eval()

    all_data = []
    
    for d in DATA_DIRS:
        df = pd.read_excel(d["excel"])
        video_dir = d["video_dir"]
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {d['excel']}"):
            video_id = str(row['Video ID']).strip()
            
            # The video file extension is .mp4
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            
            # Use 'Hinglish Text' for embeddings. If missing, fallback to 'Hindi Text'
            text = str(row['Hinglish Text']) if pd.notna(row['Hinglish Text']) else str(row['Hindi Text'])
            if pd.isna(text) or text == 'nan':
                text = ""
                
            label = str(row['Label']).strip()
            
            # Compute text embedding
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(DEVICE)
                outputs = text_model(**inputs)
                # CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            
            all_data.append({
                "video_id": video_id,
                "video_path": video_path,
                "text_embedding": embedding,
                "label": label
            })
            
    # Cache embeddings
    with open('cached_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print("Text embeddings pre-computed and cached in 'cached_data.pkl'.")
    return all_data

# --- 2. Dataset Definition ---
class MultimodalDataset(Dataset):
    def __init__(self, data, label_to_idx):
        self.data = data
        self.label_to_idx = label_to_idx
        
        # Standard ImageNet transforms for MobileNet/EfficientNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)
        
    def _extract_frames(self, video_path):
        frames = []
        if not os.path.exists(video_path):
            return torch.zeros((NUM_FRAMES, 3, 224, 224))
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return torch.zeros((NUM_FRAMES, 3, 224, 224))
            
        # Sample evenly spaced frames
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        cap.release()
        
        # Padding in case extraction failed
        while len(frames) < NUM_FRAMES:
            frames.append(torch.zeros((3, 224, 224)))
            
        return torch.stack(frames)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        video_path = item["video_path"]
        video_frames = self._extract_frames(video_path)
        
        text_emb = torch.tensor(item["text_embedding"], dtype=torch.float32)
        label = self.label_to_idx[item["label"]]
        
        return video_frames, text_emb, torch.tensor(label, dtype=torch.long)

# --- 3. Model Definition ---
class MultimodalIntentModel(nn.Module):
    def __init__(self, num_classes, text_emb_dim=768):
        super(MultimodalIntentModel, self).__init__()
        
        # Video Branch: MobileNetV3 Small (Lightweight CNN)
        self.vision_model = models.mobilenet_v3_small(pretrained=True)
        # MobileNetV3 small classifier output feature dim before classifier is 576
        vision_out_dim = self.vision_model.classifier[0].in_features
        # Replace classifier to just output features
        self.vision_model.classifier = nn.Identity()
        
        # Fusion & Classification Layer
        combined_dim = text_emb_dim + vision_out_dim
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
            # No Softmax needed here because CrossEntropyLoss applies it internally
        )

    def forward(self, video_frames, text_emb):
        # video_frames shape: (batch_size, num_frames, 3, 224, 224)
        batch_size, num_frames, C, H, W = video_frames.size()
        
        # Flatten batch and frames for CNN
        frames_flat = video_frames.view(-1, C, H, W)
        
        vision_features = self.vision_model(frames_flat) # (batch_size * num_frames, vision_out_dim)
        
        # Reshape and apply average pooling over frames
        vision_features = vision_features.view(batch_size, num_frames, -1)
        video_emb = torch.mean(vision_features, dim=1) # (batch_size, vision_out_dim)
        
        # Feature Concatenation
        combined_emb = torch.cat((text_emb, video_emb), dim=1) # (batch_size, combined_dim)
        
        # MLP
        logits = self.mlp(combined_emb)
        
        return logits

# --- 4. Main Execution ---
def main():
    if not os.path.exists('cached_data.pkl'):
        all_data = precompute_text_embeddings()
    else:
        print("Loading cached data from 'cached_data.pkl'...")
        with open('cached_data.pkl', 'rb') as f:
            all_data = pickle.load(f)
            
    # Filter out data where videos are missing
    valid_data = []
    missing_count = 0
    for d in all_data:
        if os.path.exists(d["video_path"]):
            valid_data.append(d)
        else:
            missing_count += 1
            
    print(f"Total instances parsed: {len(all_data)}")
    print(f"Valid instances (videos found): {len(valid_data)}")
    if missing_count > 0:
        print(f"Warning: {missing_count} videos were not found and will be skipped.")
    
    if len(valid_data) == 0:
        print("No valid data found. Exiting.")
        return

    # Extract unique labels
    unique_labels = sorted(list(set([item["label"] for item in valid_data])))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print(f"Number of distinct classes: {len(unique_labels)}")
    
    dataset = MultimodalDataset(valid_data, label_to_idx)
    
    # Train-val-test split (70-15-15)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # num_workers=0 to prevent multiprocessing issues on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model instantiation
    model = MultimodalIntentModel(num_classes=len(unique_labels), text_emb_dim=768).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for video_frames, text_embs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            video_frames, text_embs, labels = video_frames.to(DEVICE), text_embs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(video_frames, text_embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * video_frames.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for video_frames, text_embs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                video_frames, text_embs, labels = video_frames.to(DEVICE), text_embs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(video_frames, text_embs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * video_frames.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {train_loss/len(train_dataset):.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_dataset):.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_multimodal_intent_model.pth")
            print("  --> Saved new best model to 'best_multimodal_intent_model.pth'")
            
    print("Training complete! Final model saved to 'final_multimodal_intent_model.pth'")
    torch.save(model.state_dict(), "final_multimodal_intent_model.pth")

    # --- 5. Final Evaluation on Test Set ---
    print("\n--- Testing Best Model on Test Set ---")
    model.load_state_dict(torch.load("best_multimodal_intent_model.pth"))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for video_frames, text_embs, labels in tqdm(test_loader, desc="Testing"):
            video_frames, text_embs, labels = video_frames.to(DEVICE), text_embs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(video_frames, text_embs)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    print(f"\n=========================================")
    print(f"Final Test Accuracy on Unseen Data: {test_acc:.2f}%")
    print(f"=========================================")

if __name__ == "__main__":
    main()
