import os
import re
import string

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
        raw_sentences = content.split('\n\n')  # Dividir por doble salto de línea
        for raw_sentence in raw_sentences:
            processed_sentence = raw_sentence.replace('\n', ' EOL ').strip()
            tokens = processed_sentence.split()
            if tokens and tokens[0] == "EOL":
                tokens.pop(0)  # Eliminar primer "EOL" si existe
            if tokens and tokens[-1] == "EOL":
                tokens.pop()  # Eliminar último "EOL" si existe
            if tokens:  # Ignorar oraciones vacías
                sentences.append(tokens)
    return sentences

def save_to_file(sentences, filename):
    """
    Guarda las oraciones tokenizadas en un archivo.
    Args:
        sentences (list): Lista de oraciones tokenizadas.
        filename (str): Ruta al archivo de salida.
    """
    with open(filename, 'w', encoding='utf-8-sig') as f:
        for sentence in sentences:
            f.write(''.join(sentence) + '\n')


def declutter():
  file = open("../corpus/train.txt", "r")
  lines = file.readlines()
  #lines = lines[:3]  # Only keep the first three lines
  cleaned_lines = []
  for line in lines:
      line = line.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
      line = line.replace('EOL', '<<<EOL>>>') # Protect EOL
      line = line.lower()  # Convert to lowercase
      line = line.replace('labelrap', '__label__rap__')
      line = line.replace('labelrock', '__label__rock__')
 
      line = re.sub(r'[\d?¿¡!"“”\-–—]', '', line) # Remove digits, dashes, quotes, etc.
      line = line.replace('…', '')  # Remove three consecutive dots
      line = line.replace('<<<eol>>>', 'EOL') # Protect EOL
      line += '\n'
      cleaned_lines.append(line)
  
  file.close()
  return cleaned_lines
  
def classify_text(text):
  
  rap_songs = []
  rock_songs = []

  for line in text:
    if line.startswith('__label__rap__'):
      line = line.replace('__label__rap__', '')
      rap_songs.append(line)
    elif line.startswith('__label__rock__'):
      line = line.replace('__label__rock__', '')
      rock_songs.append(line)

  return rap_songs, rock_songs

if __name__=="__main__":
    cleaned_lines = declutter()

    save_to_file(cleaned_lines, "../corpus/cleaned_train.txt")


    # tokenized_sentences = read_and_tokenize("../corpus/cleaned_rap.txt")
    # with open("../corpus/tokenized_corpus.txt", "w") as output_file_2:
    #   for sentence in tokenized_sentences:
    #     output_file_2.write(' '.join(sentence) + '\n')