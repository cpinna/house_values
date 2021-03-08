#%%
#Importar Bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
#Carregar bases
features = pd.read_csv('features.csv')
# %%
features.head()
# %%
#Verificar campos que possuem valor Zero
np.sum(features == 0)
# %%
#Detectamos que vários imóveis possuem valor de área = 0 e banheiros igual = 0, isso precisa ser excluído dos dados
features_limpas = features[(features['area'] != 0) & (features['banheiros'] != 0)]
np.sum(features_limpas == 0)

# %%
#Criar mapa de correlações entre as variáveis numéricas
corr = features_limpas[['preco', 'valor_condominio', 'area', 'quartos', 'banheiros', 'vagas', 'churrasqueira', 'varanda', 'piscina', 'elevador', 'salao', 'seguranca', 'portaria', 'academia']].corr()

sns.heatmap(corr)
plt.title('Mapa de Correlações das Principais Features')
plt.show()
# %%
#Boxplot do preço por cidade e tipo para buscar comportamentos e outliers
sns.set_style('darkgrid')
sns.boxplot(y = 'preco',
            x = 'cidade',
            data = features_limpas,
            hue = 'tipo',
            palette = 'tab10')
sns.despine(offset=10, trim=True)
plt.title('Boxplot de Cidade e Tipo por Valor')
plt.ticklabel_format(axis = 'y', style = 'plain')
plt.show()
# %%
#Obserar um scatterplot pela área e preço dos imóveis
sns.scatterplot(y = 'preco',
                x = 'area',
                data = features_limpas,
                hue = 'cidade',
                palette = 'tab10')
plt.title('Scatterplot de área por quartos')
plt.ticklabel_format(axis = 'y', style = 'plain')
plt.tight_layout()
plt.show()
# %%
#Como a presença de Outliers é muito forte,  vamos aplicar o intervalo de confiança de 95% no preço
p_025 = features_limpas['preco'].quantile(.025)
p_975 = features_limpas['preco'].quantile(.975)
int_conf_p = features_limpas['preco'].between(p_025, p_975)
features_95 = features_limpas[int_conf_p]

#Também aplicar IC de 95% nas areas
a_025 = features_95['area'].quantile(.025)
a_975 = features_95['area'].quantile(.975)
int_conf_a = features_95['area'].between(a_025, a_975)
features_95 = features_95[int_conf_a]
# %%
features_95.info()
# %%
#Checar a distribuição dos dados no IC 95
#Criar mapa de correlações entre as variáveis numéricas
corr_95 = features_95[['preco', 'valor_condominio', 'area', 'quartos', 'banheiros', 'vagas', 'churrasqueira', 'varanda', 'piscina', 'elevador', 'salao', 'seguranca', 'portaria', 'academia']].corr()

sns.heatmap(corr_95)
plt.title('Mapa de Correlações das Principais Features')
plt.show()
# %%
#Boxplot do preço por cidade e tipo para buscar comportamentos e outliers
sns.set_style('darkgrid')
sns.boxplot(y = 'preco',
            x = 'cidade',
            data = features_95,
            hue = 'tipo',
            palette = 'tab10')
sns.despine(offset=10, trim=True)
plt.title('Boxplot de Cidade e Tipo por Valor')
plt.ticklabel_format(axis = 'y', style = 'plain')
plt.show()
# %%
#Obserar um scatterplot pela área e preço dos imóveis
sns.scatterplot(y = 'preco',
                x = 'area',
                data = features_95,
                hue = 'cidade',
                palette = 'tab10')
plt.title('Scatterplot de área por quartos')
plt.ticklabel_format(axis = 'y', style = 'plain')
plt.tight_layout()
plt.show()


# %%
#Olhando pelo gráfico, percebemos que a quantidade de dados liberados para Viana, impossibilitam a análise
features_95 = features_95[features_95['cidade'] != 'Viana']

#%%
#Identificar a quantidade de casas por bairro e fazer tratamentos
bairros = features_95.groupby('bairro').agg({'preco':'count'}).sort_values('preco').reset_index()
lista_bairros = bairros[bairros['preco'] > 5]['bairro'].tolist()

#Filtrar apenas os bairros que contém 5 ou mais anúncios
features_95 = features_95[features_95['bairro'].isin(lista_bairros)]

# %%
#Exportar aquivo
features_95.to_csv('features95.csv',
                        index = False,
                        encoding = 'UTF-8')