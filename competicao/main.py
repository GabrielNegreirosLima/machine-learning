import pandas as pd
from base_am.resultado import Fold
from base_am.avaliacao import Experimento
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from competicao_am.metodo_competicao import MetodoCompeticaoHierarquico
from competicao_am.preprocessamento_atributos_competicao import gerar_atributos_letra_musica
from competicao_am.avaliacao_competicao import OtimizacaoObjetivoSVMCompeticao, OtimizacaoObjetivoRandomForestCompeticao

df_lyrics = pd.read_csv("datasets/lyrics_amostra.csv")
col_classe = "grouped_genre"

#remove id
df_lyrics.drop("id",axis=1)


# separa em treino e validacao manualmente
df_treino = df_lyrics.sample(frac=0.7,random_state=2)
df_validacao = df_lyrics.drop(df_treino.index)
fold = Fold(df_treino, df_validacao, col_classe=col_classe)

df_treino_valid = df_treino.sample(frac=0.7,random_state=2)
df_validacao_valid = df_treino.drop(df_treino_valid.index)
fold.arr_folds_validacao = [Fold(df_treino_valid, df_validacao_valid, col_classe=col_classe)]


#cria o metodo de ap de maquina
scikit_rndforest_method = RandomForestClassifier(random_state=2,class_weight='balanced',n_estimators=64)
scikit_svm_method = LinearSVC(random_state=2)

#no método de competição hierarquico, temos que passar como parametro qual é a classe do primeiro nivel tb
ml_rndforest_method = MetodoCompeticaoHierarquico(scikit_rndforest_method,"grouped_genre")
ml_svm_method = MetodoCompeticaoHierarquico(scikit_svm_method,"grouped_genre")

experimento_rndforest = Experimento([fold], ml_method=ml_rndforest_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoRandomForestCompeticao,
                    num_trials=5)

experimento_svm = Experimento([fold], ml_method=ml_svm_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoSVMCompeticao,
                    num_trials=5)

macro_f1_randomforest, macro_f1_svm = experimento_rndforest.macro_f1_avg, experimento_svm.macro_f1_avg
print(f"Melhor Macro F1 RandomForest: {macro_f1_randomforest}")
print(f"Melhor Macro F1 SVM: {macro_f1_svm}")

# gerar relatório do random forest
best_rndf_params = experimento_rndforest.studies_per_fold[0].best_params
best_rndf = MetodoCompeticaoHierarquico(
    RandomForestClassifier(random_state=2, class_weight='balanced', n_estimators=best_rndf_params['n_estimators']),
    "grouped_genre")

result = best_rndf.eval(df_treino,df_validacao,"genre",seed=2)

print("=========== RandomForest ===========")
print("====== Resultado primeiro Nivel ====")
result_prim_nivel = best_rndf.result_prim_nivel
print(f"Macro F1: {result_prim_nivel.macro_f1}")
print(result_prim_nivel.mat_confusao)
print(best_rndf.obj_class_prim_nivel.dic_int_to_nom_classe)
print(classification_report(result_prim_nivel.y, result_prim_nivel.predict_y))

print("\n\n====== Resultado segundo nivel =====")
print(f"Macro F1: {result.macro_f1}")
print(result.mat_confusao)
print(best_rndf.obj_class_final.dic_int_to_nom_classe)

print(classification_report(result.y, result.predict_y))

# gerar relatório do svm
best_svm_params = experimento_svm.studies_per_fold[0].best_params
best_rndf = MetodoCompeticaoHierarquico(
    LinearSVC(random_state=2, class_weight='balanced', C=2**best_svm_params['exp_cost']),
    "grouped_genre")

result = best_rndf.eval(df_treino,df_validacao,"genre",seed=2)

print("=============== SVM ================")
print("====== Resultado primeiro Nivel ====")
result_prim_nivel = best_rndf.result_prim_nivel
print(f"Macro F1: {result_prim_nivel.macro_f1}")
print(result_prim_nivel.mat_confusao)
print(best_rndf.obj_class_prim_nivel.dic_int_to_nom_classe)
print(classification_report(result_prim_nivel.y, result_prim_nivel.predict_y))

print("\n\n====== Resultado segundo nivel =====")
print(f"Macro F1: {result.macro_f1}")
print(result.mat_confusao)
print(best_rndf.obj_class_final.dic_int_to_nom_classe)

print(classification_report(result.y, result.predict_y))