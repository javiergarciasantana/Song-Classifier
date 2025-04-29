import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# ----------------------------- Gestión del Vocabulario -----------------------------
class Vocabulary:
    def __init__(self, min_freq=3):  # Parámetro para frecuencia mínima
        """
        Inicializa el vocabulario.
        Args:
            min_freq (int): Frecuencia mínima para que un token sea incluido en el vocabulario.
                            Tokens con frecuencia menor serán reemplazados por <UNK>.
        """
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.size = 0
        self.min_freq = min_freq  # Guardar el umbral de frecuencia mínima
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
        # Contar la frecuencia de cada token
        token_freq = Counter(token for sentence in sentences for token in sentence)
        # Añadir tokens al vocabulario si su frecuencia es mayor o igual a min_freq
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

# ----------------------------- Conjunto de Datos Personalizado -----------------------------
class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, vocab, context_size):
        """
        Dataset personalizado para el modelo.
        Args:
            sentences (list): Lista de oraciones (listas de tokens).
            labels (list): Lista de etiquetas correspondientes.
            vocab (Vocabulary): Objeto Vocabulary para codificar tokens.
            context_size (int): Tamaño del contexto.
        """
        self.vocab = vocab
        self.context_size = context_size
        self.data = []
        self.pad_idx = vocab.token_to_idx['<PAD>']
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
    
# ----------------------------- Funciones de Lectura y Preprocesamiento -----------------------------
def read_and_tokenize_with_labels(filename):
    """
    Lee un archivo con canciones de rap y rock, extrayendo las canciones y sus etiquetas.
    Args:
        filename (str): Ruta al archivo de entrada con las canciones etiquetadas.
    Returns:
        tuple: (sentences_class0, sentences_class1), donde
            sentences_class0 son las canciones de rap,
            sentences_class1 son las canciones de rock.
    """
    sentences_tokenized = []  # Lista para canciones
    labels = []  # Lista para etiquetas

    with open(filename, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
        # Dividir el archivo en canciones basadas en el marcador de etiquetas
        raw_sentences = content.split('\n\n')  # Dividir por doble salto de línea
        for sentence in raw_sentences:
            if sentence.strip():  # Ignorar líneas vacías
                # Dividir la etiqueta y el contenido de la canción
                #print(f"Procesando canción: {sentence[:30]}...")  # Mostrar un fragmento de la canción
                parts = sentence.split(' ', 1)
                if len(parts) > 1:
                    #print(f"Partes: {parts[1]}")  # Mostrar partes de la canción
                    label, song = parts[0], parts[1]
                    #print(f"Etiqueta: {label}, Canción: {song[:30]}...")  # Mostrar etiqueta y un fragmento de la canción
                    tokens = song.split()  # tokenizar
                    #print(f"Tokens: {tokens[:10]}...")  # Mostrar los primeros 10 tokens
                    sentences_tokenized.append(tokens) 
                    if label == '\n__label__rap__' or label == '__label__rap__':
                        labels.append(0)
                    elif label == '\n__label__rock__' or label == '__label__rock__':
                        labels.append(1)

    return sentences_tokenized, labels


