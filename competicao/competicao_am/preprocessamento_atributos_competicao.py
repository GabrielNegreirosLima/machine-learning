import pandas as pd
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def gerar_atributos_letra_musica(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame, max_df:float) -> pd.DataFrame:
    bow_amostra = BagOfWordsLyrics(max_df)
    df_bow_treino = bow_amostra.cria_bow(df_treino,"lyrics")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"lyrics")

    return df_bow_treino,df_bow_data_to_predict

lem = WordNetLemmatizer()
def lem_tokenizer(lyrics, to_lower = True, only_alphanum = True):
    tokens = lyrics
    
    # tokenize only if not tokenized
    if isinstance(lyrics, str):
        tokens = word_tokenize(lyrics)
    
    # process tokens: lower case > clamp to alphanum > lemmatize
    processed_tokens = []
    for i, tk in enumerate(tokens):
        if to_lower:
            tk = tk.lower()

        if only_alphanum:
            tk = re.sub(r'\W+', '', tk)

        if tk == '':
            continue

        processed_tokens.append(lem.lemmatize(tk))
    pos_tags = nltk.pos_tag(tokens)

    return processed_tokens, pos_tags

def preprocessar_dataframe(df: pd.DataFrame):
    df_preprocessado = []

    for row in df.values:
        lyrics = row[-1]

        # pular instancia caso nao tenha letra
        if not isinstance(lyrics, str):
            continue

        tokens, pos_tags = lem_tokenizer(lyrics)

        # reconstruir letra
        processed_lyrics = ''
        for tk in tokens:
            processed_lyrics += tk + ' '
        
        # reconstruir dataframe
        df_preprocessado.append(list(row[:-1]) + [str(processed_lyrics)])
    
    return pd.DataFrame(df_preprocessado, columns=list(df.columns))

stop_list = lem_tokenizer(["i","he","she","it","a","the","almost","do","does"])[0]
#stop_list = lem_tokenizer(stopwords.words('english'))[0]
#stop_list = lem_tokenizer(stopwords.words('english') + stopwords.words('german') + stopwords.words('spanish'))[0]
class BagOfWordsLyrics(BagOfWords):
    def __init__(self, max_df:float):
        #O TfidfVectorizer que é resposavel por gerar a representação BOW
        #você pode mudar a parametrização do mesmo (inclusive, na fase de avaliação)
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        #max_df: remove palavras que ocorrem em mais que 90% dos documentos
        #stop_words: lista das stopwords a serem removidas
        self.vectorizer = TfidfVectorizer(norm="l2",max_df=max_df, stop_words=stop_list)
        pass


