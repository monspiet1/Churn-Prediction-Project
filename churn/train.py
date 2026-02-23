# %%

import pandas as pd
from sklearn import model_selection
from sklearn import tree
import matplotlib.pyplot as plt
from feature_engine import discretisation, encoding
from sklearn import pipeline
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
import mlflow
from sklearn import metrics

mlflow.set_experiment(experiment_name="churn_experiment")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

df = pd.read_csv("abt_churn.csv")
df.head()

# %%
#---------------ETAPA DE SAMPLE----------------------

# definição do out of time

oot = df[df['dtRef'] == df['dtRef'].max()].copy()

# %%
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
features = df_train.columns[2:-1]
target = 'flagChurn'



# %%
# defino o conjunto de dados como sendo todas_linhas x 40 colunas escolhidas
# y vai ser somente a coluna flagChurn
X, y = df_train[features], df_train[target]


# %%


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    random_state=42, 
                                                                    test_size=0.2,
                                                                    stratify=y
                                                                    ) 

# %%
print("Taxa variável resposta Treino: ", y_train.mean())
print("Taxa variável resposta Teste: ", y_test.mean())

#----------------------------------------------------

# %%
# -----------------ETAPA DE EXPLORE-----------------------

# missing values
X_train.isna().sum().sort_values(ascending=False)


# %%
df_analise = X_train.copy()
df_analise[target] = y_train

sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T
sumario

# %%
# analisando as diferenças entre as médias e medianas para cada target
# para ter uma ideia de quais as variáveis mais importantes
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)

# %%


arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# a árvore nos ajuda a entender quais as variáveis melhores que devemos considerar
# isso porque as que estao mais acima, dividem melhor o grupo

# %%
# transformamos o vetor de importancia em serie e relacionamos cada valor
# ao seu respectivo index dentro do conjunto de treinamento
feature_importance = (pd.Series(arvore.feature_importances_, 
                               index=X_train.columns)
                               .sort_values(ascending=False)
                               .reset_index()
                               )

feature_importance['acumulada'] = feature_importance[0].cumsum()

# pode se fazer um filtro para usar somente as variáveis com soma acumulada
# abaixo de 0.95 ou que sejam maiores que 0.01, ou ambos
feature_importance[feature_importance['acumulada'] < 0.96]
#      ou
# feature_importance[feature_importance[0] > 0.01]

# -----------------------ETAPA DE MODIFY ------------------------------
# %%
# utilizando apenas as melhores variáveis
best_features = (feature_importance[feature_importance['acumulada'] < 0.96]['index']
                 .tolist())

best_features

# %% modificando as variáveis

# nó de discretização do pipeline
tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                             bin_output="bin_number",
                                                             regression= False,
                                                             cv=3
                                                             )
#nó de one-hot encoding do pipeline
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%
# ---------------- ETAPA DE MODEL ------------------------ 

# nó do modelo do pipeline
#model = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
#model = naive_bayes.BernoulliNB()

model = ensemble.RandomForestClassifier(random_state=42,
                                        n_jobs=2, # quantos nucleos do pc quer usar
                                        )
# definindo o grid --> hiperparametros que quero testar
params = {
    "min_samples_leaf": [ 15, 20, 25, 30, 50],
    "n_estimators": [100, 200, 500, 1000],
    "criterion": ['gini', 'entropy', 'log_loss'],
}

grid = model_selection.GridSearchCV(model, 
                                    params, 
                                    cv=3, 
                                    scoring='roc_auc',
                                    verbose=4
                                    )

# desta maneira a discretizacao e o onehot sao feitos somente uma vez
# o grid que é feito varias vezes treinando o modelo ate achar o melhor
model_pipeline = pipeline.Pipeline(
    steps=[
        ('Discretizar', tree_discretisation),
        ('Onehot', onehot),
        ('Grid', grid)
    ]
)

# %%

with mlflow.start_run(run_name=model.__str__()):
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train[best_features], y_train)

    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

# -------------  ETAPA DE ASSESS ---------------------------
    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acurácia Treino: ", acc_train)
    print("AUC treino: ", auc_train)

    
    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Acurácia Teste: ", acc_test)
    print("AUC teste: ", auc_test)

    oot_predict = model_pipeline.predict(oot[best_features])
    oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]
    roc_oot = metrics.roc_curve(oot[target], oot_proba)

    acc_oot = metrics.accuracy_score(oot[target], oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], oot_proba)

    print("Acurácia oot: ", acc_oot)
    print("AUC oot: ", auc_oot)

    mlflow.log_metrics({
        "acc_train":acc_train,
        "auc_train":auc_train,
        "acc_test":acc_test,
        "auc_test":auc_test,
        "acc_oot":acc_oot,
        "auc_oot":auc_oot,
    })

# %%
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title("Curva ROC")
plt.legend([
    f"Treino : {100 * auc_train:.2f}",
    f"Teste : {100 * auc_test:.2f}",
    f"Out Of Time : {100 * auc_oot:.2f}"
])

plt.show()


# %%
# --------------------- PREDIÇÕES COM O MODELO - 2 FORMAS --------------------  
model_df = pd.Series({
            "model": model_pipeline,
            "features": best_features
            })

model_df.to_pickle('model_2.pkl')

# %%

