import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import seaborn as sns
import json
from datetime import datetime
import argparse

class DigitMLP(nn.Module):
    def __init__(self):
        super(DigitMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def train_model(force_retrain=False):
    model_path = '/app/modelo_entrenado/digit_mlp.pth'
    metrics_dir = '/app/modelo_entrenado/metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"El modelo ya existe en '{model_path}'.")
        print("Para reentrenar el modelo, use la opción --force-retrain")
        return
    
    if force_retrain:
        print("Forzando reentrenamiento del modelo...")
        if os.path.exists(model_path):
            os.remove(model_path)
            print("Modelo anterior eliminado.")
    
    print("Iniciando el entrenamiento del modelo...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model = DigitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    train_f1_scores = []
    test_f1_scores = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        train_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_test_preds = []
        all_test_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_test_preds.extend(pred.cpu().numpy())
                all_test_targets.extend(target.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * correct / total
        test_f1 = f1_score(all_test_targets, all_test_preds, average='weighted')
        
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)
        test_f1_scores.append(test_f1)
        
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train F1: {train_f1:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test F1: {test_f1:.4f}')
    
    torch.save(model.state_dict(), model_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cm = confusion_matrix(all_test_targets, all_test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.savefig(os.path.join(metrics_dir, f'confusion_matrix_{timestamp}.png'))
    plt.close()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Precisión durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(train_f1_scores, label='Train F1')
    plt.plot(test_f1_scores, label='Test F1')
    plt.title('F1-Score durante el entrenamiento')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f'training_metrics_{timestamp}.png'))
    plt.close()
    
    metrics = {
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'final_train_accuracy': train_accuracies[-1],
        'final_test_accuracy': test_accuracies[-1],
        'final_train_f1': train_f1_scores[-1],
        'final_test_f1': test_f1_scores[-1],
        'classification_report': classification_report(all_test_targets, all_test_preds),
        'confusion_matrix': cm.tolist(),
        'training_history': {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_f1_scores': train_f1_scores,
            'test_f1_scores': test_f1_scores
        }
    }
    
    with open(os.path.join(metrics_dir, f'metrics_{timestamp}.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\nMétricas guardadas en '{metrics_dir}'")
    print(f"Modelo guardado en '{model_path}'")
    print("\nReporte de clasificación final:")
    print(metrics['classification_report'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar modelo de reconocimiento de dígitos')
    parser.add_argument('--force-retrain', action='store_true', help='Forzar el reentrenamiento del modelo')
    args = parser.parse_args()
    
    train_model(force_retrain=args.force_retrain) 