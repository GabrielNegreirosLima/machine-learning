import pandas as pd
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems
from sklearn.feature_extraction.text import TfidfVectorizer

def gerar_atributos_letra_musica(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame, max_df:float) -> pd.DataFrame:
    bow_amostra = BagOfWordsLyrics(max_df)
    df_bow_treino = bow_amostra.cria_bow(df_treino,"lyrics")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"lyrics")

    return df_bow_treino,df_bow_data_to_predict

# extraido da lib nltk (stopwords)
#stop_list = {'almost', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
stop_list = {"i","he","she","it","a","the","almost","do","does"}
class BagOfWordsLyrics(BagOfWords):
    def __init__(self, max_df:float):
        #O TfidfVectorizer que é resposavel por gerar a representação BOW
        #você pode mudar a parametrização do mesmo (inclusive, na fase de avaliação)
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        #max_df: remove palavras que ocorrem em mais que 90% dos documentos
        #stop_words: lista das stopwords a serem removidas
        self.vectorizer = TfidfVectorizer(norm="l2",max_df=max_df, stop_words=stop_list)
        pass


