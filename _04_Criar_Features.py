#%%
#Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import re
from tqdm import tqdm_notebook
# %%
#Carregar dados limpos
dados_limpos = pd.read_csv('dados_tratados.csv',
                           engine = 'python',
                           header = 0,
                           encoding = 'UTF-8').fillna(0)
dados_limpos.info()
# %%
dados_limpos.head()
# %%
dados_limpos.groupby('tipo').count()

#Remover o tipo 'Aluguel' da base
dados_limpos = dados_limpos[~dados_limpos['tipo'].str.contains('Aluguel')]
#%%
#Criar o DF de Features
dados_features = ['preco',
                  'valor_condominio',
                  'area',
                  'quartos',
                  'banheiros',
                  'vagas',
                  'cidade',
                  'bairro']
features = dados_limpos[dados_features]
# %%
#Criar a feature do tipo do apartamento
'''
Ap duplex/triplex, kitchenette, loft e padrão serão agrupados juntos ao 'apartamento'
Casa em vila será agrupado para grupo de 'casa em rua pública'
-----------
Utilizar mesmo loop para coletar informações dos detalhes do imóvel:
Churrasqueira, armário de cozinha, piscina, salão de festa e varanda
'''

#Criar as funções de tratamento para cada etapa
def def_tipo(t):
    if bool(re.search('cobertura', t)):
        return 'cobertura'

    elif bool(re.search(r'apartamento|loft', t)):
        return 'apartamento'
    
    #Categorias de 'casa
    if bool(re.search('condominio fechado', t)):
        return 'casa_condominio'
    elif bool(re.search('casa', t)):
        return 'casa_rua'
    
def loc_churrasqueira(ap, condominio):
    '''
    Localizar se o imóvel possui churrasqueiras e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('churrasqueira', str(ap).lower())):
        return 1
    elif bool(re.search('churrasqueira', str(condominio).lower())):
        return 1
    else:
        return 0 

def loc_armarios(ap):
    '''
    Localizar se o imóvel possui armários na cozinha e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('armários na cozinha', str(ap).lower())):
        return 1
    else:
        return 0 
    
def loc_varanda(ap):
    '''
    Localizar se o imóvel possui varanda e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('varanda', str(ap).lower())):
        return 1
    else:
        return 0 

def loc_piscina(ap, condominio):
    '''
    Localizar se o imóvel possui piscina e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('piscina', str(ap).lower())):
        return 1
    elif bool(re.search('piscina', str(condominio).lower())):
        return 1
    else:
        return 0 
    
def loc_salao(ap, condominio):
    '''
    Localizar se o imóvel possui salão de festas e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('salão de festas', str(ap).lower())):
        return 1
    elif bool(re.search('salão de festas', str(condominio).lower())):
        return 1
    else:
        return 0 

def loc_seguranca(condominio):
    '''
    Localizar se o imóvel possui segurnaça24h e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('segurança 24h', str(condominio).lower())):
        return 1
    else:
        return 0 

def loc_portaria(condominio):
    '''
    Localizar se o imóvel possui portaria e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('portaria', str(condominio).lower())):
        return 1
    else:
        return 0 
    
def loc_elevador(condominio):
    '''
    Localizar se o imóvel possui elevador e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('elevador', str(condominio).lower())):
        return 1
    else:
        return 0 

def loc_academia(condominio):
    '''
    Localizar se o imóvel possui academia e retornar 0 ou 1 (Neg ou Pos)
    '''
    if bool(re.search('academia', str(condominio).lower())):
        return 1
    else:
        return 0 

#%%
#Criar as listas das features que serão tratadas
tipo = []
churrasqueira = []
armarios_cozinha = []
varanda = []
piscina = []
salao = []
seguranca = []
portaria = []
elevador = []
academia = []

#Tratamento das features
for i in tqdm_notebook(dados_limpos.index):

    #Definir tipo
    t = dados_limpos.loc[i,'tipo']
    tipo.append(def_tipo(t))
    
    #Criar str dos dados
    det_imovel = dados_limpos.loc[i,'detalhes_imovel']
    det_condominio = dados_limpos.loc[i,'detalhes_condominio']
    
    #Executar funções de localizar (loc)
    churrasqueira.append(loc_churrasqueira(det_imovel, det_condominio))
    armarios_cozinha.append(loc_armarios(det_imovel))
    varanda.append(loc_varanda(det_imovel))
    piscina.append(loc_piscina(det_imovel, det_condominio))
    salao.append(loc_salao(det_imovel, det_condominio))
    seguranca.append(loc_seguranca(det_condominio))
    portaria.append(loc_portaria(det_condominio))
    elevador.append(loc_elevador(det_condominio))
    academia.append(loc_academia(det_condominio))
   
#Adicionar os dados ao DF de Features
features['tipo'] = tipo
features['churrasqueira'] = churrasqueira
features['armarios_cozinha'] = armarios_cozinha
features['varanda'] = varanda
features['piscina'] = piscina
features['salao'] = salao
features['seguranca'] = seguranca
features['portaria'] = portaria
features['elevador'] = elevador
features['academia'] = academia
#%%
#Criação das Dummy Variables para poder lidar com as variáveis categóricas
features_dummies = pd.get_dummies(features)
features_dummies.head()
# %%
features_dummies.to_csv('features_dummies.csv', encoding = 'UTF-8', index = False)
features.to_csv('features.csv', encoding = 'UTF-8', index = False)
# %%
