#%%
#Importar libraries necessárias
import numpy as np
import pandas as pd

# %%
#Carregar base de dados criada
features = pd.read_csv('features95.csv')
features.head()
# %%
#Testar apenas com os dados numéricos
feat_num = features.drop(labels = ['cidade', 'bairro', 'tipo'], axis = 1)
# %%
#Separar o X do y das nossas amostras 
from sklearn.model_selection import train_test_split

X, y = feat_num.drop('preco', axis = 1), feat_num['preco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%
#Importar e aplicar a Linear Regresion e  medir o RMSE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error  

model = LinearRegression()

model.fit(X = X_train, y = y_train)
y_pred = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test, y_pred)))

# %%
#Visualiar a dispersão da previsão dos dados X os dados reais
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.show()
#%%
#Fazer um novo split com a base completa
#Separar o X do y das nossas amostras 
X, y = features.drop('preco', axis = 1), feat_num['preco']

X_train, X_test, y_train, y_test = train_test_split 

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# %%
#Criar um modelo geral usando Dummy,  One Hot e  standartizado
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

#Aplicar o OneHot apenas às colunas categóricas da nossa base
ohe = make_column_transformer((OneHotEncoder(sparse  = False), ['cidade', 'bairro', 'tipo']),
                           remainder = 'passthrough')   
std = StandardScaler()
reg = LinearRegression()

#Montar Pipeline com todos os processos
pipe = make_pipeline(ohe, std, reg)

#Ajustar o modelo ao pipe
pipe.fit(X_train, y_train)

#Prever nos dados de validação
y_pred = pipe.predict(X_test)

#Criar função para RMSE
def rmse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

print(rmse(y_test, y_pred))
# %%
plt.scatter(y_test, y_pred)
plt.xlabel('Valor de controle')
plt.ylabel('Valor previsto')
plt.title('Imagem do controle X previsto')
plt.show()
# %%
#investigar o Outlier que apareceu como valor previsto negativo
test_neg =X_test.copy()
test_neg['preco'] = y_test
test_neg['preco_prev'] = y_pred

test_neg[y_pred <0]
# No teste vemos que há valores previstos como negativos no modelo de regressão linear simples
#Isso implica que no modelo final precisaremos utilizar uma forma diferente na previsão
# %%
# Testar modelo utilizando os principais modelos de árvores
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


models = {
    'dtr': DecisionTreeRegressor(),
    'rfr': RandomForestRegressor(),
    'xgb': XGBRegressor(),
    'lgbm': LGBMRegressor(),
    'ada': AdaBoostRegressor(),
}

#%%
for model in models.keys():
    #fazer pipe
    pipe = make_pipeline(ohe, models[model])
    #Ajustar o modelo ao pipe
    pipe.fit(X_train, y_train)

    #Prever nos dados de validação
    y_pred = pipe.predict(X_test)
    print('O modelo {} teve RMSE de {} e MAE de {}'.format(model, rmse(y_test, y_pred), mean_absolute_error(y_test, y_pred)))
    #plotar gráfico
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valor de controle')
    plt.ylabel('Valor previsto')
    plt.title('Imagem do controle X previsto - {}'.format(model))
    plt.show()


# %%
# O melhor modelo foi o Random Forest, com MAE de 65.239 e RMSE de  107.511
# Vamos então tunar os hyperparâmetros do RF para tentar chegar no melhor modelo possível

# Bayesian Optimazation
from skopt import forest_minimize

parametros_possiveis = [(10,300),# n_estimators
                        (5,100), #max_depth
                        (2,10), #min_sample_split
                        (1,20), #min_sample_leaf
                        ('auto', 'sqrt', 'log2'), #max_features,
                        (True, False)] #bootstrap

def tunar_parametros(param):
    print('-----------')
    # Selecionar parâmetros
    n_estimators= param[0]
    max_depth = param[1]
    min_sample_split = param[2]
    min_sample_leaf = param[3]
    max_features = param[4]
    bootstrap = param[5]
    
    # Montar modelo
    rand_forest = RandomForestRegressor( n_estimators = n_estimators,
                                        max_depth = max_depth,
                                        min_samples_split = min_sample_split,
                                        min_samples_leaf = min_sample_leaf,
                                        max_features = max_features,
                                        bootstrap = bootstrap,
                                        n_jobs = 12)
    
    # Montar o pipeline usando o One Hot Encoder e o modelo base criado
    pipe = make_pipeline(ohe, rand_forest)
    
     #Ajustar o modelo ao pipe
    pipe.fit(X_train, y_train)

    #Prever nos dados de validação
    y_pred = pipe.predict(X_test)
    
    #Pegar o resutado de RMSE
    performance = rmse(y_test, y_pred)
    
    return performance

#Aplicar ao forest_minimize
resultado = forest_minimize(tunar_parametros,
                            parametros_possiveis,
                            random_state = 123456,
                            n_random_starts = 50,
                            n_calls = 80,
                            verbose = True)

# %%
# Testar o XGBoost que obteve resultado bastante próximo

parametros_possiveis_xgb = [(0.05,0.9),# eta / Learning_rate
                        (0,15000), #gamma
                        (1,30), #max_depth
                        (0,20), #min_child_weight
                        (0,20), #max_delta_step,
                        (0,1), #lambda
                        (0,1), #alpha 
                        (10,1000)] #n_estimators


def tunar_parametros_xgb(param):
    print('-----------')
    # Selecionar parâmetros
    learn_rate = param[0]
    gamma = param[1]
    max_depth = param[2]
    min_child_weight = param[3]
    max_delta_step = param[4]
    lamb = param[5]
    alpha = param[6]
    
    # Montar modelo
    xgb_reg = XGBRegressor(learning_rate = learn_rate,
                           gamma = gamma, 
                           max_depth = max_depth,
                           min_child_weight = min_child_weight,
                           max_delta_step = max_delta_step,
                           reg_lambda = lamb,
                           reg_alpha = alpha,
                           n_jobs = 12)
    
    # Montar o pipeline usando o One Hot Encoder e o modelo base criado
    pipe_xgb = make_pipeline(ohe, xgb_reg)
    
     #Ajustar o modelo ao pipe
    pipe_xgb.fit(X_train, y_train)

    #Prever nos dados de validação
    y_pred_xgb = pipe_xgb.predict(X_test)
    
    #Pegar o resutado de RMSE
    performance_xgb = rmse(y_test, y_pred_xgb)
    
    return performance_xgb

#Aplicar ao forest_minimize
resultado_xgb = forest_minimize(tunar_parametros_xgb,
                            parametros_possiveis_xgb,
                            n_random_starts = 50,
                            n_calls = 80,
                            verbose = True)
# %%
#Melhores resultados
# rand_forest_rmse = 103071
# xgb_rmse = 103807
# %%
# Captar os melhores parâmetros
resultado.x
# %%
