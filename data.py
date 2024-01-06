pip install num2words
pip install num2words
pip install nbconvert
!jupyter nbconvert --to html Untitled11.ipynb
pip install scispacy
pip install spacy

import en_core_web_sm
from num2words import num2words
import pandas as pd
import re

df = pd.read_csv('rotuladow.csv', sep=',', quotechar='"', encoding='utf8') # arquivo de 10/04/2023

def str_replace(x):
    number = x.group(1)
    extract_num = re.findall(r'\d+', number)[0]
    return num2words((extract_num), lang='pt_BR')

# criando nova coluna com descrição por extenso
extenso = df.Descrição.str.replace(r'(\d+)', str_replace)
df['extenso'] = extenso

df.to_csv("rotulado_final_v1.7.csv", sep=',', quotechar='"', index=False)

pd.set_option('display.max_colwidth', 400)

line = re.sub(r"""
  (?x) # Use free-spacing mode.
  <    # Match a literal '<'
  /?   # Optionally match a '/'
  \[   # Match a literal '['
  \d+  # Match one or more digits
  >    # Match a literal '>'
  """, "", line)

df_tratamento = pd.read_csv('rotulado_final_v1.7.csv', sep=',', quotechar='"', encoding='utf8')

# Criação de acrônimos
class Solution:
   def solve(self, s):
      tokens=s.split()
      string=""
      for word in tokens:
         if word != "and":
            string += str(word[0])
      return string.upper()
ob = Solution()
print(ob.solve("National Aeronautics and Space Administration"))

nlp = en_core_web_sm.load()

nlp = spacy.load("en_core_web_sm")

import spacy
from scispacy.abbreviation import AbbreviationDetector

nlp = spacy.load("en_core_web_sm")

abbreviation_pipe = AbbreviationDetector(nlp)
nlp.add_pipe(abbreviation_pipe)

text = "StackOverflow (SO) is a question and answer site for professional and enthusiast programmers. SO rocks!"

def replace_acronyms(text):
    doc = nlp(text)
    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)

    return(" ".join(altered_tok))

replace_acronyms(text)

df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'/', ' / ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'~', ' ~ ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'-', ' - ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'\n', ' \n ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'"', ' " ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'\.', ' . ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r' \(', ' ( ', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'\)', ' )', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'#', ' # ', x))


for i, desc in enumerate(df_tratamento['extenso']):

    words = desc.split()
 
    spelled_words = []
    for word in words:
     
        spelled_word = ' '.join([letter for letter in word[:3]])
        spelled_words.append(spelled_word)

    df_tratamento.at[i, 'extenso'] = ' '.join(spelled_words)

import re

def processamento_coluna(df_tratamento, extenso):

    
   
    padrao_sigla = r"\b([A-Z]{3})\b"
    
   
    for i in range(len(df_tratamento[extenso])):
        value = df_tratamento[extenso][i]
        matches = regex.findall(value)
        for match in matches:
            
            group1 = match[0]
            group2 = match[1]
            group3 = match[2]
           
            value = value.replace(match, f"{group1} {group2} {group3}")
        
        df_tratamento[extenso][i] = value
    
    return df_tratamento

    padrao_sigla = r"\b([A-Z]{3})\b"
    
    def replace(match):
        return match.group(1)
    
    df_tratamento_1 = df_tratamento.copy()
    
    column = df_tratamento_1[extenso].astype(str)
    
    df_tratamento_1[extenso] = column.apply(lambda x: re.sub(pattern, replace, x))
    
    return df_tratamento_1

df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'BHA', 'B H A', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'SMS', 'S M S', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'BOP', 'B O P', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'CVU', 'C V U', x))
df_tratamento['extenso'] = df_tratamento['extenso'].apply(lambda x: re.sub(r'duzentosgpm', 'duzentos gpm', x))


from collections import Counter


text_list = df_tratamento['extenso'].tolist()

text = ' '.join(text_list)


words = re.findall(r'\b\w+\b', text)


word_counts = Counter(words)


df_word_counts = pd.DataFrame.from_dict(word_counts, orient='index', columns=['count'])

print(df_word_counts)

def find_words_and_acronyms(text):
    words = []
    acronyms = []

    current_word = ""
    for letter in text:
        if letter.isalpha():
            current_word += letter
        else:
            if len(current_word) > 1:
                if current_word.isupper():
                    acronyms.append(current_word)
                else:
                    words.append(current_word)
            current_word = ""

    if len(current_word) > 1:
        if current_word.isupper():
            acronyms.append(current_word)
        else:
            words.append(current_word)

    return (words, acronyms)

texto = "IBM é uma grande empresa, mas sua sede não está no BR ."
palavras, siglas = find_words_and_acronyms(texto)
print("Palavras: ", palavras)
print("Siglas: ", siglas)