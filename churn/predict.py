# %%
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model = mlflow.sklearn.load_model("models:/churn_model/2")

# %%
features = model.feature_names_in_
features

# %%
# novos dados

df = pd.read_csv("abt_churn.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra = amostra.drop('flagChurn', axis=1)
amostra

# %%
pred = model.predict_proba(amostra[features])[:, 1]
amostra['proba'] = pred
amostra
