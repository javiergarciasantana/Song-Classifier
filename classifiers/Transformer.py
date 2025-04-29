#Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math
from collections import Counter

# ----------------------------- Gestión del Vocabulario -----------------------------
class Vocabulary:
    def __init__(self, min_freq=3):
        """
        Inicializa el vocabulario.
        Args:
            min_freq (int): Frecuencia mínima para incluir un token en el vocabulario.
                            Tokens con frecuencia menor serán reemplazados por <UNK>.
        """
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.size = 0
        self.min_freq = min_freq
        self.add_token('<PAD>')   # Añadir token de padding
        self.add_token('<UNK>')   # Añadir token desconocido

    def add_token(self, token):
        """Añade un token al vocabulario si no existe."""
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.size
            self.idx_to_token[self.size] = token
            self.size += 1

    def build_vocab(self, sentences):
        """
        Construye el vocabulario a partir de las oraciones del conjunto de entrenamiento.
        Tokens con frecuencia menor que min_freq se reemplazan por <UNK>.
        Args:
            sentences (list): Lista de listas de tokens (oraciones).
        """
        token_freq = Counter(token for sentence in sentences for token in sentence)
        for token, freq in token_freq.items():
            if freq >= self.min_freq:
                self.add_token(token)

    def encode(self, tokens):
        """
        Codifica una lista de tokens en sus índices correspondientes.
        Tokens desconocidos se mapean al índice de <UNK>.
        """
        unk_idx = self.token_to_idx['<UNK>']
        return [self.token_to_idx.get(token, unk_idx) for token in tokens]

    def decode(self, indices):
        """
        Decodifica una lista de índices en sus tokens correspondientes.
        Índices desconocidos se mapean al token <UNK>.
        """
        unk_token = '<UNK>'
        return [self.idx_to_token.get(idx, unk_token) for idx in indices]


# ----------------------------- Transformer Model Definitions -----------------------------
def positional_encoding(max_len, emb_dim):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
    pe = torch.zeros(max_len, emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class AttentionHead(nn.Module):
    def __init__(self, emb_dim, d_h):
        super().__init__()
        self.W_Q = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_Q = nn.Parameter(torch.empty(d_h))
        self.W_K = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_K = nn.Parameter(torch.empty(d_h))
        self.W_V = nn.Parameter(torch.empty(emb_dim, d_h))
        self.b_V = nn.Parameter(torch.empty(d_h))
        self.d_h = d_h
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.zeros_(self.b_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.zeros_(self.b_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.zeros_(self.b_V)

    def forward(self, x, causal_mask=None, padding_mask=None):
        Q = x @ self.W_Q + self.b_Q
        K = x @ self.W_K + self.b_K
        V = x @ self.W_V + self.b_V
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1), float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        d_h = emb_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(emb_dim, d_h) for _ in range(num_heads)])
        self.W_O = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, x, causal_mask=None, padding_mask=None):
        head_outputs = [head(x, causal_mask, padding_mask) for head in self.heads]
        x_concat = torch.cat(head_outputs, dim=-1)
        return x_concat @ self.W_O

class MLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or emb_dim * 4
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, feedforward_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.mlp = MLP(emb_dim, hidden_dim=feedforward_hidden_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, src_key_padding_mask=None):
        attn_output = self.attn(self.norm1(x), causal_mask=causal_mask, padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_output)
        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        return x

class DecoderForClassification(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, context_size, num_classes, pad_idx, dropout=0.1):
        super().__init__()
        self.context_size = context_size
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.register_buffer('positional_encodings', positional_encoding(context_size, emb_dim))
        self.pos_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([DecoderBlock(emb_dim, num_heads, dropout=dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.classifier_head = nn.Linear(emb_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, src_key_padding_mask=None):
        batch_size, seq_len = x.size()
        x_emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        pos_enc = self.positional_encodings[:seq_len, :]
        hidden_states = self.pos_dropout(x_emb + pos_enc.unsqueeze(0))
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        hidden_states = self.final_norm(hidden_states)
        sequence_lengths = (~src_key_padding_mask).sum(dim=1) if src_key_padding_mask is not None else torch.tensor([seq_len] * batch_size)
        last_token_indices = torch.clamp(sequence_lengths - 1, min=0)
        batch_indices = torch.arange(batch_size, device=x.device)
        last_token_hidden_states = hidden_states[batch_indices, last_token_indices, :]
        logits = self.classifier_head(last_token_hidden_states)
        return logits

# ----------------------------- Funciones de Lectura y Preprocesamiento -----------------------------
def read_and_tokenize(filename):
    """
    Lee un archivo y tokeniza las oraciones, sustituyendo saltos de línea por 'EOL'.
    Args:
        filename (str): Ruta al archivo de entrada.
    Returns:
        list: Una lista de oraciones, donde cada oración es una lista de tokens.
    """
    sentences = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read()
        raw_sentences = content.split('\n\n')
        for raw_sentence in raw_sentences:
            processed_sentence = raw_sentence.replace('\n', ' EOL ').strip()
            tokens = processed_sentence.split()
            if tokens and tokens[0] == "EOL":
                tokens.pop(0)
            if tokens and tokens[-1] == "EOL":
                tokens.pop()
            if tokens:
                sentences.append(tokens)
    return sentences

# ----------------------------- Conjunto de Datos Personalizado -----------------------------
class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, vocab, context_size):
        """
        Dataset personalizado para el modelo Transformer.
        Args:
            sentences (list): Lista de oraciones (listas de tokens).
            labels (list): Lista de etiquetas correspondientes.
            vocab (Vocabulary): Objeto Vocabulary para codificar tokens.
            context_size (int): Tamaño máximo del contexto.
        """
        self.vocab = vocab
        self.context_size = context_size
        self.pad_idx = vocab.token_to_idx['<PAD>']
        self.data = []
        for sentence, label in zip(sentences, labels):
            encoded_sentence = vocab.encode(sentence)
            if len(encoded_sentence) < context_size:
                encoded_sentence += [self.pad_idx] * (context_size - len(encoded_sentence))
            else:
                encoded_sentence = encoded_sentence[:context_size]
            self.data.append((encoded_sentence, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, label = self.data[idx]
        return torch.tensor(inputs), torch.tensor(label)

def collate_fn(batch):
    """
    Función de collate para manejar batches de datos.
    Asegura que las secuencias tengan la misma longitud mediante padding y genera una máscara de padding.
    """
    inputs, labels = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=vocab.token_to_idx['<PAD>'])
    labels = torch.tensor(labels)
    src_key_padding_mask = (inputs == vocab.token_to_idx['<PAD>'])
    return inputs, labels, src_key_padding_mask


# ----------------------------- Función de Entrenamiento (Transformers) -----------------------------
def train_classification_model(model, train_loader, val_loader, num_epochs, learning_rate, device='cpu', clip_value=1.0):
    """Entrena el modelo de clasificación usando Transformers."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # Pérdida para clasificación multiclase
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_idx, (tokens, labels, src_key_padding_mask) in enumerate(train_loader):
            tokens, labels = tokens.to(device), labels.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            optimizer.zero_grad()
            logits = model(tokens, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
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

# ----------------------------- Función de Evaluación (Transformers) -----------------------------
def evaluate_model(model, data_loader, criterion, device='cpu'):
    """Evalúa el modelo de clasificación en un conjunto de datos usando Transformers."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, labels, src_key_padding_mask in data_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            src_key_padding_mask = src_key_padding_mask.to(device)
            logits = model(tokens, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy

# ----------------------------- Script Principal (Transformers) -----------------------------
if __name__ == "__main__":
    # --- Configuración ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    # Paso 1: Leer y tokenizar oraciones de dos archivos
    file1 = r"./all_lyrics1.txt"
    file2 = r"./all_lyrics2.txt"
    sentences_class0 = read_and_tokenize(file1)
    sentences_class1 = read_and_tokenize(file2)
    # Reducir el tamaño del dataset para pruebas rápidas
    #sentences_class0 = sentences_class0[:30]
    #sentences_class1 = sentences_class1[:30]

    # Paso 2: Construir vocabulario solo desde el conjunto de entrenamiento
    vocab = Vocabulary(min_freq=5)
    train_sentences_class0 = sentences_class0[:int(len(sentences_class0) * 0.8)]
    train_sentences_class1 = sentences_class1[:int(len(sentences_class1) * 0.8)]
    train_sentences = train_sentences_class0 + train_sentences_class1
    vocab.build_vocab(train_sentences)

    # Paso 3: Asignar etiquetas (0 para clase 0, 1 para clase 1)
    labels_class0 = [0] * len(sentences_class0)
    labels_class1 = [1] * len(sentences_class1)

    # Combinar todas las oraciones y etiquetas
    sentences = sentences_class0 + sentences_class1
    labels = labels_class0 + labels_class1
    # Mezclar el conjunto combinado

    import random
    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    sentences, labels = zip(*combined)
    # Dividir en conjuntos de entrenamiento y prueba
    train_ratio = 0.8
    split_idx = int(len(sentences) * train_ratio)
    train_sentences = sentences[:split_idx]
    test_sentences = sentences[split_idx:]
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]
    # Crear datasets y dataloaders
    context_size = 256
    train_dataset = SentenceDataset(train_sentences, train_labels, vocab, context_size)
    test_dataset = SentenceDataset(test_sentences, test_labels, vocab, context_size)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

    # --- Inicialización del Modelo (Transformer) ---
    print("Inicializando modelo Transformer...")
    model = DecoderForClassification(
        vocab_size=vocab.size,
        emb_dim=128,
        num_heads=8,
        num_blocks=6,
        context_size=context_size,
        num_classes=2,
        pad_idx=vocab.token_to_idx['<PAD>'],
        dropout=0.1
    ).to(device)
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Entrenamiento y Evaluación ---
    train_classification_model(
        model,
        train_loader,
        test_loader,
        num_epochs=30,
        learning_rate=0.001,
        device=device,
    )
    # --- Evaluación Final ---
    print("\n--- Evaluación Final ---")
    final_train_loss, final_train_acc = evaluate_model(model, train_loader, nn.CrossEntropyLoss(), device)
    final_val_loss, final_val_acc = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Entrenamiento Final: Pérdida={final_train_loss:.4f}, Precisión={final_train_acc:.2f}%")
    print(f"Validación Final: Pérdida={final_val_loss:.4f}, Precisión={final_val_acc:.2f}%")
