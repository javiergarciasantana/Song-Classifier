import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from auxfunctions import Vocabulary, SentenceDataset, read_and_tokenize_with_labels

def collate_fn_PAD(batch):
    """
    Función de collate para manejar batches de datos.
    Asegura que las secuencias tengan la misma longitud mediante padding.
    """
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=vocab.token_to_idx['<PAD>'])
    labels = torch.tensor(labels)
    return inputs, labels

# ----------------------------- Modelo BengioNN -----------------------------
class BengioNN(nn.Module):
    def __init__(self, vocab_size, context_size, embed_size, hidden_dim):
        super(BengioNN, self).__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Capa de embedding
        self.fc1 = nn.Linear(embed_size * context_size, hidden_dim)  # Capa fully connected
        self.fc2 = nn.Linear(hidden_dim, 1)  # Salida para clasificación binaria

    def forward(self, x):
        embedded = self.embedding(x)  # Shape: [batch_size, context_size, embed_size]
        flattened = embedded.view(embedded.size(0), -1)  # Aplanar los embeddings
        hidden = torch.relu(self.fc1(flattened))  # Aplicar activación ReLU
        logits = self.fc2(hidden)  # Salida final
        return logits

# ----------------------------- Función de Entrenamiento (MODIFICADA para BengioNN) -----------------------------
def train_classification_model(model, train_loader, val_loader, vocab, num_epochs, learning_rate, device='cpu', clip_value=1.0):
    """Entrena el modelo de clasificación usando BengioNN."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Pérdida para clasificación binaria
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()  # Modo de entrenamiento
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (tokens, labels) in enumerate(train_loader):
            tokens, labels = tokens.to(device), labels.float().to(device)  # Etiquetas como float para BCEWithLogitsLoss
            optimizer.zero_grad()
            # Forward pass: Obtener logits
            logits = model(tokens).squeeze()  # Shape: [batch_size]
            loss = criterion(logits, labels)  # Calcular pérdida
            loss.backward()  # Backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Recortar gradientes
            optimizer.step()  # Actualizar pesos
            train_loss += loss.item()
            predicted = (torch.sigmoid(logits) > 0.5).float()  # Umbral en 0.5
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            if (batch_idx + 1) % 50 == 0:
                print(f"  Época [{epoch+1}/{num_epochs}], Lote [{batch_idx+1}/{len(train_loader)}], Pérdida: {loss.item():.4f}")
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        print(f"Época [{epoch+1}/{num_epochs}] Resumen de Entrenamiento:")
        print(f"  Pérdida Promedio: {avg_train_loss:.4f}, Precisión: {train_accuracy:.2f}%")
        # --- Paso de Validación ---
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        print(f"Época [{epoch+1}/{num_epochs}] Resumen de Validación:")
        print(f"  Pérdida Promedio: {val_loss:.4f}, Precisión: {val_accuracy:.2f}%")
        print("-" * 30)

# ----------------------------- Función de Evaluación (MODIFICADA para BengioNN) -----------------------------
def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Evalúa el modelo de clasificación en un conjunto de datos usando BengioNN."""
    model.to(device)
    model.eval()  # Modo de evaluación
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Desactivar cálculo de gradientes
        for tokens, labels in data_loader:
            tokens, labels = tokens.to(device), labels.float().to(device)  # Etiquetas como float para BCEWithLogitsLoss
            # Forward pass: Obtener logits
            logits = model(tokens).squeeze()  # Shape: [batch_size]
            loss = criterion(logits, labels)  # Calcular pérdida
            total_loss += loss.item()
            predicted = (torch.sigmoid(logits) > 0.5).float()  # Umbral en 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

# ----------------------------- Script Principal (BengioNN) -----------------------------
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
    context_size = 128
    train_dataset = SentenceDataset(train_sentences, train_labels, vocab, context_size)
    test_dataset = SentenceDataset(test_sentences, test_labels, vocab, context_size)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn_PAD)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_PAD)
    # --- Inicialización del Modelo (BengioNN) ---
    print("Inicializando modelo BengioNN...")
    model = BengioNN(
        vocab_size=vocab.size,
        context_size=context_size,
        embed_size=128,
        hidden_dim=64
    ).to(device)
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # --- Entrenamiento y Evaluación ---
    train_classification_model(
        model,
        train_loader,
        test_loader,  # Pasa el cargador de validación
        vocab,
        num_epochs=30,
        learning_rate=0.01,
        device=device,
    )
    # --- Evaluación Final ---
    print("\n--- Evaluación Final ---")
    final_train_loss, final_train_acc = evaluate_model(model, train_loader, nn.BCEWithLogitsLoss(), device)
    final_val_loss, final_val_acc = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(), device)
    print(f"Entrenamiento Final: Pérdida={final_train_loss:.4f}, Precisión={final_train_acc:.2f}%")
    print(f"Validación Final: Pérdida={final_val_loss:.4f}, Precisión={final_val_acc:.2f}%")

    # --- Guardar el modelo ---
    save_path = "models/bengioNN.pth"  # Puedes elegir otro directorio/nombre
    torch.save(model.state_dict(), save_path)
    print(f"Modelo guardado en {save_path}")