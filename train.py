import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataloader_pupil import FacePupilDataset  # Import your dataset script

# Load your dataset
pupil_dir_tr = "./pupil_data/"  # Adjust as needed
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
train_set = FacePupilDataset(pupil_dir_tr, transform=transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

# Define the MobileNetV3Small model
class MobileNetV3Small(nn.Module):
    def __init__(self, n_class=4):  # Assuming 4 outputs for x, y of left and right pupils
        super(MobileNetV3Small, self).__init__()
        self.mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(1024, n_class)  # Adjust output layer

    def forward(self, x):
        return self.mobilenet(x)

# Model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3Small(n_class=4).to(device)  # Assuming 4 outputs for x, y of left and right pupils
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100  # Adjust based on your needs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        images = batch['image'].to(device)
        labels = batch['label']  # Make sure your labels are properly formatted
        labels = torch.tensor(labels, dtype=torch.float32).to(device)  # Convert to tensor if not already

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save your model after training
MODEL_SAVE_PATH = './models/mnet3s-00.pt'
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved to:", MODEL_SAVE_PATH)
