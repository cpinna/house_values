# Sistema para Captar informações de todos os imóveis anunciados na OLX na Grande Vitória
# Depois criar um sistema de Machine Learning para ajudar a avaliar o preço dos imóveis
# %%
# Importar bibliotecas necessárias
import numpy as np  # Processaemento numérico e matrizes
import \
    pandas as pd  # 'adaptação' do R dentro python -> Manipulação de tabelas / Dataframes -> Feito em cima da estrutura do Numpy
import re  # ReGex -> Regular expressions -> Localização de padrões textuais
import time  # Manipulação de tempo
import json  # Manipulação de arquivos tipo .json

import bs4  # Manipulação de .HTML
from selenium import webdriver  # Controle de Navegador através de código

# %%
# Coleta OLX - URL
cidades_consulta = ['vitoria', 'vila-velha', 'outras-cidades']
url = 'https://es.olx.com.br/norte-do-espirito-santo/{cidade}/imoveis/venda?o={pagina}'
# %%
# Coletar os dados
driver = webdriver.Chrome()  # Abrir Chrome
for cidade in cidades_consulta:  # Para cada cidade, verificar as 100 primeiras páginas
    for pagina in range(1, 101):
        urll = url.format(cidade=cidade, pagina=pagina)
        print(urll)
        driver.get(urll)  # Acessar a página no navegador
        if driver.current_url == 'https://es.olx.com.br/norte-do-espirito-santo/{cidade}/imoveis/venda'.format(
                cidade=cidade):  # Se a ágina voltar para o básico, chegamos no limite, logo vamos para a próxima cidade de busca
            break

        with open('./Dados_brutos_search/{}_{}.html'.format(cidade, pagina), 'w+') as output:
            output.write(driver.page_source)  # Salvar o HTML da página
        time.sleep(2)
driver.quit()
# %%
# Processamento dos dados brutos
# Olhar individualmente os dados de cada pagina salva e coletar os dados específicos pelas tags
for cidade in cidades_consulta:
    for pagina in range(1, 101):
        with open('./Dados_brutos_search/{}_{}.html'.format(cidade, pagina), 'r+') as inp:
            pagina_html = inp.read()
            # Transformar o arquivo HTML em uma estrutura legível pelo Python usando  o BS4
            parsed = bs4.BeautifulSoup(pagina_html)
            # Captar dados de título e link das consultas
            tags_a = parsed.findAll('a')
            for tag in tags_a:
                if tag['class'] == ['fnmrjs-0', 'fyjObc']:
                    titulo = tag['title']
                    link = tag['href']
                    with open('dados_extraidos_paginas.json', 'a+') as output:
                        dados = {'Titulo': titulo, 'link': link, 'cidade': cidade, 'pagina': pagina}
                        output.write('{}\n'.format(json.dumps(dados)))
# %%
