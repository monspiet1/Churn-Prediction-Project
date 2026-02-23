# %%
import pandas as pd

df = pd.read_excel("dados_cerveja_nota.xlsx")
df.head()
# %%
from sklearn import linear_model

X = df[['cerveja']] # o X é um dataframe
y = df['nota'] # o y é uma série

reg = linear_model.LinearRegression()
reg.fit(X, y=y)

a, b = reg.intercept_, reg.coef_[0]
print("Os valores que mais minimizam a soma do erro quadrático são: \n\n", a, b)

predict_reg = reg.predict(X.drop_duplicates())

# %%
from sklearn import tree

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X,y=y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

# neste caso nós mudamos a profundidade para a árvore nao superajustar aos dados
arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2) 
arvore_d2.fit(X,y=y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %%
import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2)
plt.legend(['Observado', 
            f'y = {a:.3f} + {b:.3f}x',
            'Árvore super ajustada',
            'Árvore Depth = 2'
            ])

# %%
