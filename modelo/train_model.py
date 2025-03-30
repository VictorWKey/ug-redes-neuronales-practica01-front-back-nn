import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# Definir la arquitectura del modelo MLP
class DigitMLP(nn.Module):
    def __init__(self):
        super(DigitMLP, self).__init__()
        # Arquitectura simple de MLP para MNIST
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

def train_model():
    # Configurar transformaciones para el dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Cargar dataset de MNIST
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Crear dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Inicializar modelo, función de pérdida y optimizador
    model = DigitMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Entrenamiento
    epochs = 5
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluar modelo
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    # Crear directorio para guardar el modelo
    os.makedirs('/app/modelo_entrenado', exist_ok=True)
    
    # Guardar el modelo
    torch.save(model.state_dict(), '/app/modelo_entrenado/digit_mlp.pth')
    
    # Guardar un ejemplo para visualización
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('/app/modelo_entrenado/training_metrics.png')
    
    print(f"Modelo guardado en '/app/modelo_entrenado/digit_mlp.pth'")
    print(f"Métricas de entrenamiento guardadas en '/app/modelo_entrenado/training_metrics.png'")

if __name__ == "__main__":
    train_model() 