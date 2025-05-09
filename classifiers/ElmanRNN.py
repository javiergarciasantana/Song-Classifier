#RNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from collections import Counter
from sklearn.utils import shuffle

from auxfunctions import Vocabulary, SentenceDataset, collate_fn, read_and_tokenize_with_labels


# ----------------------------- Modelo ElmanRNN -----------------------------
class ElmanRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim):
        super(ElmanRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Output 1 for binary classification

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, sequence_length, embed_size]
        output, hidden = self.rnn(embedded)  # Shape: [batch_size, sequence_length, hidden_dim]
        logits = self.fc(hidden[-1])  # Use the final hidden state
        return logits

# ----------------------------- Función de Entrenamiento (ElmanRNN) -----------------------------
def train_classification_model(model, data_loader, num_epochs, learning_rate, device="cpu"):
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_batches = len(data_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze(-1)  # Shape: [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}"
                )
        avg_loss = total_loss / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {avg_loss:.4f}")

# ----------------------------- Función de Evaluación (ElmanRNN) -----------------------------
def evaluate_model(model, data_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(-1)  # Shape: [batch_size]
            predictions = (torch.sigmoid(outputs) > 0.5).long()  # Threshold at 0.5
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# ----------------------------- Script Principal  -----------------------------
if __name__ == "__main__":
    # --- Configuración ---
    # Using apple Silicon GPU, metal performance
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")
    # Paso 1: Leer y tokenizar oraciones de dos archivos
    file1 = r"../corpus/cleaned_train.txt"

    sentences, labels = read_and_tokenize_with_labels(file1)
  
    # Reducir el tamaño del dataset para pruebas rápidas
    #sentences_class0 = sentences_class0[:30]
    #sentences_class1 = sentences_class1[:30]

    # Paso 2: Construir vocabulario solo desde el conjunto de entrenamiento
    vocab = Vocabulary(min_freq=5)  # Usar min_freq para entrenar con <UNK>
    train_sentences = sentences[int(len(sentences) * 0.8):]  # Usar el 80% de las oraciones para el vocabulario
    vocab.build_vocab(train_sentences)  # Construir vocabulario con min_freq

    # Mezclar el conjunto combinado
    # import random
    # combined = list(zip(sentences, labels))
    # random.shuffle(combined)
    # sentences, labels = zip(*combined)

    # Dividir en conjuntos de entrenamiento y prueba
    train_ratio = 0.8
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[split_idx:]
    test_sentences = sentences[:split_idx]
    train_labels = labels[split_idx:]
    test_labels = labels[:split_idx]
    # Crear datasets y dataloaders
    context_size = 256
    train_dataset = SentenceDataset(train_sentences, train_labels, vocab, context_size)
    test_dataset = SentenceDataset(test_sentences, test_labels, vocab, context_size)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    # --- Inicialización del Modelo (ElmanRNN) ---
    print("Inicializando modelo ElmanRNN...")
    model = ElmanRNN(
        vocab_size=vocab.size,
        embed_size=128,
        hidden_dim=64
    ).to(device)
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # --- Entrenamiento y Evaluación ---

    train_classification_model(model, train_loader, num_epochs=30, learning_rate=0.01, device=device)
    # --- Evaluación Final ---
    print("\n--- Evaluación Final ---")
    # Evaluate the model on the test set
    criterion = nn.CrossEntropyLoss()
    test_accuracy = evaluate_model(model, test_loader, device=device)
    print(f"Test Accuracy (ElmanRNN): {test_accuracy:.4f}")

    # --- Guardar el modelo ---
    save_path = "models/ElmanRNN.pth"  # Puedes elegir otro directorio/nombre
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado en {save_path}")
   