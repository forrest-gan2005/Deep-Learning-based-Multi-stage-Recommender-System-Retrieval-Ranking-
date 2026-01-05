import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import MovieLensDataLoader
from dataset import MovieLensDataset
from models.deepfm import DeepFM

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
loader = MovieLensDataLoader("data/raw/ml-1m")
final_df, field_dims = loader.load_and_preprocess()

# Create Dataset & Dataloader
dataset = MovieLensDataset(final_df)
train_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Initialize Model
model = DeepFM(field_dims, embed_dim=16).to(device)
criterion = nn.BCELoss() # Binary Cross Entropy for CTR prediction
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simple Training Loop
model.train()
for epoch in range(5):
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        #Backward pass
        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Calculate gradients
        optimizer.step()        # Update weights
        
        total_loss += loss.item()
        
        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"epoch [{epoch+1}/5], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "deepfm_model.pth")