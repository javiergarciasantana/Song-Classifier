import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from colorama import Back, Style
from BengioNN import BengioNN, collate_fn_PAD
from ElmanRNN import ElmanRNN, collate_fn_RNN
from auxfunctions import Vocabulary, SentenceDataset, read_and_tokenize_with_labels
import csv

def calculate_accuracy(predictions, labels):
    """
    Calcula la precisión de las predicciones.
    Args:
        predictions (list): Lista de predicciones del modelo.
        labels (list): Lista de etiquetas verdaderas.
    Returns:
        float: Precisión en porcentaje.
    """
    correct = sum(p == l for p, l in zip(predictions, labels))
    return 100 * correct / len(labels)

# ----------------------------- Funciones de Predicción y CSV -----------------------------
def predict_with_model(model, processed_corpus, vocab, device):
    model.eval()  # Poner el modelo en modo evaluación
    predictions = []
    
    for sentence in processed_corpus:
        # Convertir la oración en índices de vocabulario
        sentence_indices = [vocab.token_to_idx.get(token, vocab.token_to_idx['<UNK>']) for token in sentence]

        # Aquí adaptamos la longitud al context_size
        if len(sentence_indices) < context_size:
            sentence_indices += [vocab.token_to_idx['<PAD>']] * (context_size - len(sentence_indices))
        else:
            sentence_indices = sentence_indices[:context_size]

        sentence_tensor = torch.tensor(sentence_indices).unsqueeze(0).to(device)  # Añadir dimensión extra para el batch
        
        # Obtener la predicción (logits)
        with torch.no_grad():
            output = model(sentence_tensor)

        #  # --- handle tuple output (RNN case) ---
        # if isinstance(output, tuple):
        #     output = output[0]

        # # Take last time step output if shape is (batch, seq_len, hidden)
        # if output.dim() == 3:
        #     output = output[:, -1, :]
        
        # # If output is still > 1 value (e.g. vocab_size logits), reduce to scalar logit
        # if output.size(-1) > 1:
        #     # Apply linear classifier if it's not part of model
        #     output = output.mean(dim=-1, keepdim=True)  # crude fallback
        #     # Ideally you have a final layer that returns [batch_size, 1]

        # Apply sigmoid to get probability
        prob = torch.sigmoid(output)
    
        # Asumir que cualquier probabilidad > 0.5 es 'rap' (1), sino 'rock' (0)
        prediction = 1 if prob.item() > 0.5 else 0
        predictions.append(prediction)
    
    return predictions

def generate_csv(pad_predictions, elman_predictions, output_filename="aluNUM.csv"):
    # Definir las categorías de clasificación según los modelos
    pad = ["K" if pred == 1 else "P" for pred in pad_predictions]  # Rap (K), Rock (P)
    recurrente = ["K" if pred == 1 else "P" for pred in elman_predictions]  # Rap (K), Rock (P)
    transformer = ["Z"] * len(pad)  # Implementacion pendiente
    ngramas = ["Z"] * len(pad)  # Implementacion pendiente

    # Guardar los resultados en un archivo CSV
    with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["num_cancion", "ngramas", "pad", "recurrente", "transformer"])
        
        for i, (ng, p, r, t) in enumerate(zip(ngramas, pad, recurrente, transformer), start=1):
            writer.writerow([i, ng, p, r, t])
    
    print(f"Archivo CSV generado: {output_filename}")


# ----------------------------- Script Principal (song_classifier) -----------------------------
if __name__ == "__main__":
    # --- Configuración ---
    # Using apple Silicon GPU, metal performance
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Usando dispositivo: {device}")
    # Paso 1: Leer y tokenizar oraciones de dos archivos
    file1 = r"../corpus/cleaned_train.txt"
    sentences, labels = read_and_tokenize_with_labels(file1)
 
    # Paso 2: Construir vocabulario solo desde el conjunto de entrenamiento
    vocab = Vocabulary(min_freq=5)  # Usar min_freq para entrenar con <UNK>
    train_sentences = sentences[int(len(sentences) * 0.8):]  # Usar el 80% de las oraciones para el vocabulario
    vocab.build_vocab(train_sentences)  # Construir vocabulario con min_freq

    # Dividir en conjuntos de entrenamiento y prueba(Prueba es el 20% de las oraciones empezando por el principio del textfile)
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
    print(Back.YELLOW + "Inicializando modelo BengioNN...")
    print(Style.RESET_ALL)
    model = BengioNN(
        vocab_size=vocab.size,
        context_size=context_size,
        embed_size=128,
        hidden_dim=64
    ).to(device)
    # Cargar pesos guardados
    load_path = "models/bengioNN.pth"
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.eval()  # Muy importante: poner en modo evaluación
    print(f"Modelo cargado desde {load_path}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # --- Predicción ---
    pad_predictions = predict_with_model(model, test_sentences, vocab, device)

    # --- Inicialización del Modelo (ElmanRNN) ---
    print(Back.YELLOW + "Inicializando modelo ElmanRNN...")
    print(Style.RESET_ALL)
    model = ElmanRNN(
        vocab_size=vocab.size,
        embed_size=128,
        hidden_dim=64
    ).to(device)
    # Cargar pesos guardados
    load_path = "models/ElmanRNN.pth"
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Modelo cargado desde {load_path}")
    print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # --- Predicción ---
    elman_predictions = predict_with_model(model, test_sentences, vocab, device)

    # --- Evaluación Final ---
    # Generar el CSV con las predicciones del conjunto de test
    print("\nGenerando predicciones para el conjunto de test...")
    #print(test_sentences[0])
    print(f"Precisión de PAD: {calculate_accuracy(pad_predictions, test_labels):.2f}%")
    print(f"Precisión de ElmanRNN: {calculate_accuracy(elman_predictions, test_labels):.2f}%")
    generate_csv(pad_predictions, elman_predictions, output_filename="alu0101391663.csv")
 