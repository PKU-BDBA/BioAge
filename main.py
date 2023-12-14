import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from dataset import MultiModalDataset
from model.classifier import ClassifierModel
from torch.utils.data import DataLoader, random_split


# Replace these with your actual data paths and labels
face_image_paths = ['path/to/face/image1.jpg', 'path/to/face/image2.jpg', ...]
tongue_image_paths = ['path/to/tongue/image1.jpg', 'path/to/tongue/image2.jpg', ...]
fundus_image_paths = ['path/to/fundus/image1.jpg', 'path/to/fundus/image2.jpg', ...]
labels = [0, 1, ...]  # Your labels here

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
num_classes = 2  # Update with your number of classes
encoder_dim = 128  # Example dimension, adjust to your Swin Transformer
hidden_dim = 512  # MLP hidden dimension
num_encoder_blocks = 4  # Number of Swin Transformer blocks

dataset = MultiModalDataset(face_image_paths, tongue_image_paths, fundus_image_paths, labels)
val_size = int(len(dataset) * 0.2)
train_size = len(dataset) - val_size

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Prepare the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = ClassifierModel(num_classes, encoder_dim, hidden_dim, num_encoder_blocks)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i, (faces, tongues, fundus, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        faces = faces.to(device)
        tongues = tongues.to(device)
        fundus = fundus.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(faces, tongues, fundus)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

# Evaluation
# Assuming you have a validation set loaded in a DataLoader called `val_loader`
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for faces, tongues, fundus, labels in val_loader:
        faces = faces.to(device)
        tongues = tongues.to(device)
        fundus = fundus.to(device)
        labels = labels.to(device)
        outputs = model(faces, tongues, fundus)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the validation images: {} %'.format(100 * correct / total))

# Load and test the model
# To demonstrate how to load and use the model for prediction
loaded_model = ClassifierModel(num_classes, encoder_dim, hidden_dim, num_encoder_blocks)
loaded_model.load_state_dict(torch.load('model.ckpt'))
loaded_model.to(device)
loaded_model.eval()

# Dummy data for prediction
# In practice, replace these with real images
test_faces, test_tongues, test_fundus = torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224), torch.randn(1, 3, 224, 224)
test_faces, test_tongues, test_fundus = test_faces.to(device), test_tongues.to(device), test_fundus.to(device)

with torch.no_grad():
    predictions = loaded_model(test_faces, test_tongues, test_fundus)
    _, predicted_class = torch.max(predictions.data, 1)
    print(f'Predicted class: {predicted_class.item()}')
