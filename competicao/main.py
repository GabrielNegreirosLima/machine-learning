import pandas as pd
from base_am.resultado import Fold
from base_am.avaliacao import Experimento
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from competicao_am.metodo_competicao import MetodoCompeticaoHierarquico
from competicao_am.preprocessamento_atributos_competicao import gerar_atributos_letra_musica
from competicao_am.avaliacao_competicao import OtimizacaoObjetivoSVMCompeticao, OtimizacaoObjetivoRandomForestCompeticao, OtimizacaoObjetivoGradientBoostCompeticao

# ====== LEITURA DO DATASET E DIVISÃO EM FOLDS
df_lyrics = pd.read_csv("datasets/lyrics_amostra.csv")
col_classe = "grouped_genre"

#remove id
df_lyrics.drop("id",axis=1)


# separa em treino e validacao
arr_folds = Fold.gerar_k_folds(df_lyrics, val_k=2, col_classe=col_classe, num_repeticoes=1, num_folds_validacao=2, num_repeticoes_validacao=1)
arr_folds = [arr_folds[0]] # queremos estudar a princípio somente um fold, para ser mais rápido



# ====== PESQUISA VIA OPTUNA PARA OBTENÇÃO DOS MELHORES PARÂMETROS
#cria o metodo de ap de maquina
scikit_rndforest_method = RandomForestClassifier(random_state=2,class_weight='balanced',n_estimators=64)
scikit_svm_method = LinearSVC(random_state=2)
scikit_gaussianNB = GaussianNB()
scikit_multinomialNB = MultinomialNB()
scikit_gradient_boost = GradientBoostingClassifier(random_state=2, min_samples_split=50, n_estimators=70)

#no método de competição hierarquico, temos que passar como parametro qual é a classe do primeiro nivel tb
ml_rndforest_method = MetodoCompeticaoHierarquico(scikit_rndforest_method,"grouped_genre")
ml_svm_method = MetodoCompeticaoHierarquico(scikit_svm_method,"grouped_genre")
ml_nb_method = MetodoCompeticaoHierarquico(scikit_gaussianNB, "grouped_genre")
ml_mnb_method = MetodoCompeticaoHierarquico(scikit_multinomialNB, "grouped_genre")
ml_gb_method = MetodoCompeticaoHierarquico(scikit_gradient_boost, "grouped_genre")

experimento_rndforest = Experimento(arr_folds, ml_method=ml_rndforest_method,
                 ClasseObjetivoOtimizacao=OtimizacaoObjetivoRandomForestCompeticao,
                    num_trials=5)

experimento_svm = Experimento(arr_folds, ml_method=ml_svm_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoSVMCompeticao,
                    num_trials=5)

experimento_nb = Experimento(arr_folds, ml_method=ml_nb_method, num_trials=1)

experimento_mnb = Experimento(arr_folds, ml_method=ml_mnb_method, num_trials=1)

experimento_gb = Experimento(arr_folds, ml_method=ml_gb_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoGradientBoostCompeticao,
                    num_trials=5)

macro_f1_randomforest, macro_f1_svm, macro_f1_nb, macro_f1_mnb, macro_f1_gb = experimento_rndforest.macro_f1_avg, experimento_svm.macro_f1_avg, experimento_nb.macro_f1_avg, experimento_mnb.macro_f1_avg, experimento_gb.macro_f1_avg

print(f"Melhor Macro F1 RandomForest: {macro_f1_randomforest}")
print(f"Melhor Macro F1 SVM: {macro_f1_svm}")
print(f"Melhor Macro F1 NB: {macro_f1_nb}")
print(f"Melhor Macro F1 MNB: {macro_f1_mnb}")
print(f"Melhor Macro F1 GB: {macro_f1_gb}")



# ====== GERAÇÃO DE RELATÓRIO COM BASE NOS MELHORES PARÂMETROS ENCONTRADOS
# usar primeiro fold como fonte do relatório
fold = arr_folds[0]

# gerar relatório do random forest
best_rndf_params = experimento_rndforest.studies_per_fold[0].best_params
best_rndf = MetodoCompeticaoHierarquico(
    RandomForestClassifier(random_state=2, class_weight='balanced', n_estimators=best_rndf_params['n_estimators']),
    "grouped_genre")
result = best_rndf.eval(fold.df_treino,fold.df_data_to_predict,"genre",seed=2)

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
best_svm = MetodoCompeticaoHierarquico(
    LinearSVC(random_state=2, class_weight='balanced', C=2**best_svm_params['exp_cost']),
    "grouped_genre")
result = best_svm.eval(fold.df_treino,fold.df_data_to_predict,"genre",seed=2)

print("=============== SVM ================")
print("====== Resultado primeiro Nivel ====")
result_prim_nivel = best_svm.result_prim_nivel
print(f"Macro F1: {result_prim_nivel.macro_f1}")
print(result_prim_nivel.mat_confusao)
print(best_svm.obj_class_prim_nivel.dic_int_to_nom_classe)
print(classification_report(result_prim_nivel.y, result_prim_nivel.predict_y))

print("\n\n====== Resultado segundo nivel =====")
print(f"Macro F1: {result.macro_f1}")
print(result.mat_confusao)
print(best_svm.obj_class_final.dic_int_to_nom_classe)

print(classification_report(result.y, result.predict_y))

# gerar relatório do naive bayes
best_nb = MetodoCompeticaoHierarquico(GaussianNB(), col_classe_prim_nivel="grouped_genre")
result = best_nb.eval(fold.df_treino,fold.df_data_to_predict,"genre",seed=2)

print("=============== NaiveBayes ================")
print("====== Resultado primeiro Nivel ====")
result_prim_nivel = best_nb.result_prim_nivel
print(f"Macro F1: {result_prim_nivel.macro_f1}")
print(result_prim_nivel.mat_confusao)
print(best_nb.obj_class_prim_nivel.dic_int_to_nom_classe)
print(classification_report(result_prim_nivel.y, result_prim_nivel.predict_y))

print("\n\n====== Resultado segundo nivel =====")
print(f"Macro F1: {result.macro_f1}")
print(result.mat_confusao)
print(best_nb.obj_class_final.dic_int_to_nom_classe)

print(classification_report(result.y, result.predict_y))

# gerar relatório do gradient boost
best_gb_params = experimento_gb.studies_per_fold[0].best_params
best_gb = MetodoCompeticaoHierarquico(
    GradientBoostingClassifier(random_state=2, min_samples_split=best_gb_params['min_samples_split'], n_estimators=best_gb_params['n_estimators']),
    col_classe_prim_nivel="grouped_genre")
result = best_gb.eval(fold.df_treino,fold.df_data_to_predict,"genre",seed=2)

print("========== GradientBoost ===========")
print("====== Resultado primeiro Nivel ====")
result_prim_nivel = best_gb.result_prim_nivel
print(f"Macro F1: {result_prim_nivel.macro_f1}")
print(result_prim_nivel.mat_confusao)
print(best_gb.obj_class_prim_nivel.dic_int_to_nom_classe)
print(classification_report(result_prim_nivel.y, result_prim_nivel.predict_y))

print("\n\n====== Resultado segundo nivel =====")
print(f"Macro F1: {result.macro_f1}")
print(result.mat_confusao)
print(best_gb.obj_class_final.dic_int_to_nom_classe)

print(classification_report(result.y, result.predict_y))