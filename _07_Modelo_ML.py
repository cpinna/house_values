#%%
#Importar libraries necessárias
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import joblib as jb

# %%
#Carregar base de features
features = pd.read_csv('features95.csv')
# %%
#Fazer o split da base 
X, y = features.drop('preco', axis = 1), features['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# %%
# Criar modelo conforme parâmetros desenvolvidos no pré-test
ohe = make_column_transformer((OneHotEncoder(sparse  = False), ['cidade', 'bairro', 'tipo']),
                           remainder = 'passthrough')
rand_forest = RandomForestRegressor(n_estimators = 140,
                                    max_depth = 27,
                                    min_samples_split = 3,
                                    min_samples_leaf = 1,
                                    max_features = 'log2',
                                    bootstrap = False,
                                    n_jobs = 12)

# Montar o pipeline usando o One Hot Encoder e o modelo base criado
pipe = make_pipeline(ohe, rand_forest)

#Ajustar o modelo ao pipe
mdl = pipe.fit(X_train, y_train)
# %%
# Salvar modelo
jb.dump(mdl, 'random_forest.pkl.z')