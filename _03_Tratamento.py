# Importar libraries necessárias
import numpy as np
import pandas as pd
# Carregar dados brutos resultantes das tratativas
dados_brutos = pd.read_json('informacao_anuncios.json', lines=True)

# Iniciar processo de limpeza dos dados criando o DF  limpo
# Limpar algumas inconsistências dos dados para uma correta estruturação dos dados
# Algumas colunas numéricas vieram com '.' e outras com texto '5 ou mais' no item de seleção.
def remover_valor(x, valor):
    string = x.str.replace(valor, '')
    return string


dados_brutos['preco'] = remover_valor(dados_brutos['preco'], '.') 
dados_brutos['valor_condominio'] = remover_valor(dados_brutos['valor_condominio'], '.')
dados_brutos['cep'] = dados_brutos['cep'].astype('str').str.replace('.0','')
dados_brutos['quartos'] = remover_valor(dados_brutos['quartos'], ' ou mais')
dados_brutos['banheiros'] = remover_valor(dados_brutos['banheiros'], ' ou mais')
dados_brutos['vagas'] = remover_valor(dados_brutos['vagas'], ' ou mais')
dados_limpos = dados_brutos.replace('NaN', np.NaN)

# Ajustar o tipo de valor das colunas
colunas_numericas = ['preco',
                     'valor_condominio',
                     'quartos',
                     'banheiros',
                     'vagas']

dados_limpos[colunas_numericas] = dados_limpos[colunas_numericas].astype('float')

# Limpar a coluna 'Preço' que contém dados vazios e iguais a Zero.
# Como a coluna 'Preço' é meu Y, não posso ter dados vazios ou Zero nele
dados_limpos = dados_limpos[dados_limpos['preco'] > 0].dropna(subset=['preco'])

# Na amostra baixada, o número de imóveis comerciais é muito pequeno (apenas 8)
# Então também vamos removêlo e setar o sistema para apenas imóveis residenciais
dados_limpos = dados_limpos[dados_limpos['categoria'] != 'Comércio e indústria']

# Substituir, em todas as colunas numéricas o 'NaN' por 0
for col in dados_limpos.columns:
    if dados_limpos[col].dtype == 'float64':
        dados_limpos[col] = dados_limpos[col].fillna(0)
dados_limpos.info()

# Exportar dados limpos
dados_limpos.to_csv('dados_tratados.csv', index=False, encoding='UTF-8')
