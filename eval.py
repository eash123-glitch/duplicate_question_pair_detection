import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for q1, q2, labels in test_loader:
            q1, q2, labels = q1.to(device), q2.to(device), labels.to(device)
            outputs = model(q1, q2)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {correct / total * 100:.2f}%")