import random
import math
import os
import pickle
from collections import defaultdict
from auxfunctions import read_and_tokenize_with_labels

def read_and_tokenize(filename) -> list:
    """
    Lee un archivo, y crea una lista donde cada elemento es una lista que contiene todas las palabras de una canción.
    Sustituye los finales de línea de cada canción por el token "EOL".
    Por ejemplo, si el corpus es:

    Hola mundo
    Mi casa

    El perro
    El gato

    La salida debe ser:
    [['Hola', 'mundo', 'EOL', 'Mi', 'casa'], ['El', 'perro', 'EOL', 'El', 'gato']]

    Args:
        filename (str): Path to the input file.

    Returns:
        list: A list of sentences, where each sentence is a list of words.
    """
    sentences = []

    with open(filename, "r", encoding="utf-8-sig") as f:
        # Read the entire file content
        content = f.read()
        # Split the content into sentences using two consecutive newlines (\n\n)
        raw_sentences = content.split("###")
        # Process each sentence
        for raw_sentence in raw_sentences:
            # Replace remaining newlines within the sentence with "EOL"
            processed_sentence = raw_sentence.replace("\n", " EOL ").strip()
            # Split the sentence into words
            tokens = processed_sentence.split()
            # Remove the first "EOL" if it exists
            if tokens and tokens[0] == "EOL":
                tokens.pop(0)
            # Remove the last "EOL" if it exists
            if tokens and tokens[-1] == "EOL":
                tokens.pop()
            # Append the processed sentence if it's not empty
            if tokens:  # Ignore empty sentences
                sentences.append(tokens)

    return sentences


def prepare_corpus(corpus, n, unk_threshold=-1):
    """
    Prepara el corpus agregando tokens <s> al inicio y </s> al final de cada oración.
    Reemplaza palabras poco frecuentes con <UNK> y construye el vocabulario.

    Args:
        corpus (list of list of str): El corpus tokenizado.
        n (int): El tamaño del modelo de n-gramas.
        unk_threshold (int): Palabras con frecuencia <= este umbral se reemplazan con <UNK>.

    Returns:
        tuple: Una tupla que contiene el corpus procesado (lista de listas de str) y el vocabulario (set).
    """
    word_counts = {}
    for sentence in corpus:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Construir el vocabulario excluyendo palabras poco frecuentes
    vocab = {word for word, count in word_counts.items() if count > unk_threshold}
    # Agregar tokens especiales al vocabulario
    if unk_threshold >= 0:
        vocab.update({"<s>", "</s>", "<UNK>"})
    else:
        vocab.update({"<s>", "</s>"})

    processed_corpus = []
    for sentence in corpus:
        # Agregar (n-1) tokens <s> al inicio y </s> al final
        processed_sentence = (
            ["<s>"] * (n - 1)
            + [word if word in vocab else "<UNK>" for word in sentence]
            + ["</s>"]
        )
        processed_corpus.append(processed_sentence)
    return processed_corpus, vocab


def generate_ngrams(corpus, n):
    """
    Genera todos los n-gramas contiguos del corpus preparado.

    Args:
        corpus (list of list of str): El corpus preparado.
        n (int): El tamaño de los n-gramas.

    Returns:
        list: Una lista de n-gramas, donde cada n-grama es una tupla de longitud n.
    """
    ngrams = []
    for sentence in corpus:
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i + n])
            ngrams.append(ngram)
    return ngrams


from collections import defaultdict

def compute_ngram_probabilities(ngrams, vocab, n):
    """
    Calcula las probabilidades suavizadas de Laplace para todos los n-gramas observados.
    Muestra mensajes de progreso durante el cálculo.

    Args:
        ngrams (list of tuple): Lista de n-gramas generados del corpus.
        vocab (set): Conjunto de palabras en el vocabulario.
        n (int): El tamaño de los n-gramas.

    Returns:
        dict: Un diccionario donde las claves son contextos (tuplas de longitud n-1)
              y los valores son distribuciones de probabilidad suavizadas.
    """
    # Usamos un diccionario común para contar las frecuencias
    cfd = {}  # Distribución condicional de frecuencias
    print(f"Procesando {len(ngrams)} n-gramas...")

    # Paso 1: Contar frecuencias de los n-gramas
    for i, ngram in enumerate(ngrams):
        context, target = tuple(ngram[:-1]), ngram[-1]
        if target == "<s>":  # Ignorar <s> como objetivo
            continue
        if context not in cfd:
            cfd[context] = {}
        if target not in cfd[context]:
            cfd[context][target] = 0
        cfd[context][target] += 1
        
        # Mostrar progreso cada 10,000 n-gramas
        if (i + 1) % 10000 == 0:
            print(f"Procesados {i + 1}/{len(ngrams)} n-gramas...")

    # Paso 2: Crear el vocabulario de objetivos (sin "<s>")
    target_vocab = list(vocab - {"<s>"})
    vocab_size = len(target_vocab)
    print(f"Calculando probabilidades para {len(cfd)} contextos observados...")
    
    # Paso 3: Calcular las probabilidades suavizadas de Laplace
    cpd = {}
    for j, (context, observed_counts) in enumerate(cfd.items()):
        total_count = sum(observed_counts.values())
        smoothed_probs = {}
        for word in target_vocab:
            smoothed_probs[word] = (observed_counts.get(word, 0) + 1) / (total_count + vocab_size)
        cpd[context] = smoothed_probs
        
        # Mostrar progreso cada 100 contextos
        if (j + 1) % 1000 == 0:
            print(f"Procesados {j + 1}/{len(cfd)} contextos...")

    print("Cálculo de probabilidades completado.")
    return cpd



def sentence_logprobability(sentence, cpd, n, vocab):
    """
    Calcula la probabilidad logarítmica de una oración usando el modelo de n-gramas con suavizado de Laplace.
    Reemplaza palabras desconocidas con <UNK> y maneja contextos no observados de manera robusta.

    Args:
        sentence (list of str): La oración como una lista de palabras.
        cpd (dict): Diccionario de distribuciones de probabilidad condicional.
        n (int): El tamaño de los n-gramas.
        vocab (set): Conjunto de palabras en el vocabulario.

    Returns:
        float: La probabilidad logarítmica de la oración.
    """
    # Reemplazar palabras fuera del vocabulario con <UNK>
    processed_sentence = [word if word in vocab else "<UNK>" for word in sentence]
    log_prob = 0.0
    for i in range(len(processed_sentence) - n + 1):
        context = tuple(processed_sentence[i : i + n - 1])
        target = processed_sentence[i + n - 1]
        if context in cpd:
            # Usar <UNK> si el objetivo no fue observado en este contexto
            token = target if target in cpd[context] else "<UNK>"
            prob = cpd[context].get(
                token, 1 / len(vocab)
            )  # Probabilidad uniforme para <UNK>
        else:
            # Aplicar suavizado de Laplace para contextos no observados
            prob = 1 / len(list(vocab - {"<s>"}))
        log_prob += math.log(prob)
    return log_prob

def split_corpus(corpus, train_ratio=0.8):
    """
    Divide el corpus en conjuntos de entrenamiento y prueba.

    Args:
        corpus (list of list of str): El corpus preparado.
        train_ratio (float): Proporción del corpus para entrenamiento (default: 0.8).

    Returns:
        tuple: Una tupla que contiene el corpus de entrenamiento y el corpus de prueba.
    """
    split_index = int(len(corpus) * train_ratio)
    train_corpus = corpus[:split_index]
    test_corpus = corpus[split_index:]

    return train_corpus, test_corpus


def compute_perplexity(test_sentences, cpd, n, vocab):
    """
    Calcula la perplejidad del modelo sobre un conjunto de oraciones de prueba.

    Args:
        test_sentences (list of list of str): Lista de oraciones de prueba.
        cpd (dict): Diccionario de distribuciones de probabilidad condicional.
        n (int): El tamaño de los n-gramas.
        vocab (set): Conjunto de palabras en el vocabulario.

    Returns:
        float: La perplejidad del modelo.
    """
    total_log_prob = 0.0
    for sentence in test_sentences:
        total_log_prob += sentence_logprobability(sentence, cpd, n, vocab)

    # Calcular la perplejidad como la probabilidad media inversa de las oraciones
    perplexity = math.exp(-total_log_prob / sum(len(s) for s in test_sentences))
    return perplexity

def classify_sentence(sentence, cpd_rap, cpd_rock, vocab, n):
    """
    Clasifica una oración como 'rap' o 'rock' basada en los modelos de n-gramas para cada clase.

    Args:
        sentence (list of str): La oración tokenizada.
        cpd_rap (dict): Probabilidades de n-gramas para la clase rap.
        cpd_rock (dict): Probabilidades de n-gramas para la clase rock.
        vocab (set): Conjunto de palabras en el vocabulario.
        n (int): El tamaño de los n-gramas.

    Returns:
        str: La etiqueta predicha ('rap' o 'rock').
    """
    log_prob_rap = sentence_logprobability(sentence, cpd_rap, n, vocab)
    log_prob_rock = sentence_logprobability(sentence, cpd_rock, n, vocab)

    if log_prob_rap > log_prob_rock:
        return 'rap'
    else:
        return 'rock'

def train_classifier(train_corpus, labels, n, vocab):
    """
    Entrena un clasificador basado en n-gramas para las clases 'rap' y 'rock'.

    Args:
        train_corpus (list of list of str): El corpus de entrenamiento tokenizado.
        labels (list of int): Las etiquetas correspondientes (0 para rap, 1 para rock).
        n (int): El tamaño de los n-gramas.
        vocab (set): Conjunto de palabras en el vocabulario.

    Returns:
        tuple: Un par de diccionarios con las distribuciones de probabilidad de n-gramas para cada clase.
    """
    # Filtrar corpus por etiquetas de clase
    rap_corpus = [train_corpus[i] for i in range(len(labels)) if labels[i] == 0]
    rock_corpus = [train_corpus[i] for i in range(len(labels)) if labels[i] == 1]

    #print(rap_corpus[0])

    # Generar n-gramas para cada clase
    ngrams_rap = generate_ngrams(rap_corpus, n)
    ngrams_rock = generate_ngrams(rock_corpus, n)

    # Calcular probabilidades de n-gramas para cada clase
    cpd_rap = compute_ngram_probabilities(ngrams_rap, vocab, n)
    cpd_rock = compute_ngram_probabilities(ngrams_rock, vocab, n)

    return cpd_rap, cpd_rock

def evaluate_classifier(test_corpus, test_labels, cpd_rap, cpd_rock, vocab, n):
    """
    Evalúa el rendimiento del clasificador en el conjunto de prueba.

    Args:
        test_corpus (list of list of str): El corpus de prueba tokenizado.
        test_labels (list of int): Las etiquetas correspondientes del conjunto de prueba.
        cpd_rap (dict): Probabilidades de n-gramas para la clase rap.
        cpd_rock (dict): Probabilidades de n-gramas para la clase rock.
        vocab (set): Conjunto de palabras en el vocabulario.
        n (int): El tamaño de los n-gramas.

    Returns:
        float: La precisión del clasificador.
    """
    correct_predictions = 0

    # Clasificar cada oración en el conjunto de prueba
    for i, sentence in enumerate(test_corpus):
        predicted_label = classify_sentence(sentence, cpd_rap, cpd_rock, vocab, n)
        if (predicted_label == 'rap' and test_labels[i] == 0) or (predicted_label == 'rock' and test_labels[i] == 1):
            correct_predictions += 1

    # Calcular la precisión
    accuracy = correct_predictions / len(test_labels)
    return accuracy

# Ejecución Principal
if __name__ == "__main__":

    # Paso 1: Leer y tokenizar el corpus
    file1 = r"../corpus/cleaned_train.txt"
    corpus, labels = read_and_tokenize_with_labels(file1)

    # Paso 2: Preparar el corpus y el vocabulario
    n = 2  # Orden del modelo de n-gramas
    unk_threshold = 5  # Umbral para palabras poco frecuentes
    print("Preparando el corpus...")
    prepared_corpus, vocab = prepare_corpus(corpus, n, unk_threshold)

    # Paso 3: Dividir el corpus en entrenamiento y prueba
    do_split_corpus = True
    if do_split_corpus:
        print("Dividiendo el corpus en entrenamiento y prueba...")
        train_corpus, test_corpus = split_corpus(prepared_corpus, train_ratio=0.8)
        train_labels, test_labels = split_corpus(labels, train_ratio=0.8)
    else:
        train_corpus = test_corpus = prepared_corpus
        train_labels = test_labels = labels

    subset_corpus = train_corpus[:len(train_corpus)//20]  # Usa solo el 20% del corpus
    subset_labels = train_labels[:len(train_labels)//20]  # Usa las etiquetas correspondientes

    # Paso 4: Entrenar el clasificador
    print("Entrenando el clasificador...")
    cpd_rap, cpd_rock = train_classifier(subset_corpus, subset_labels, n, vocab)

    # Paso 5: Evaluar el clasificador
    print("Evaluando el clasificador...")
    accuracy = evaluate_classifier(test_corpus, test_labels, cpd_rap, cpd_rock, vocab, n)

    # Paso 6: Mostrar resultados
    print(f"Precisión del clasificador: {accuracy * 100:.2f}%")

    print("Guardando el modelo...")
    # Guardar
    with open("ngram_model.pkl", "wb") as f:
      pickle.dump((cpd_rap, cpd_rock, vocab), f)