import torch
import torch.nn as nn
import torch.optim as optim
from datasets.barkData import get_dataloader
from models.barkClassifyModel import BarkClassifyModel

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BarkClassifyModel(num_classes=10).to(device)
    dataloader = get_dataloader(data_dir='./data', batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')
