import torch
from datasets.barkData import get_dataloader
from models.barkClassifyModel import BarkClassifyModel

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BarkClassifyModel(num_classes=10).to(device)
    model.load_state_dict(torch.load('model.pth'))
    dataloader = get_dataloader(data_dir='./data', batch_size=32, shuffle=False)
    
    accuracy = evaluate(model, dataloader, device)
    print(f"Accuracy: {accuracy * 100:.2f}%")
