#Sistema para Captar informações de todos os imóveis anunciados na OLX, VivaReal e ZapImóveis na Grande Vitória
#Depois criar um sistema de Machine Learning para ajudar a avaliar o preço dos imóveis
#%%
#Importar bibliotecas necessárias 
import numpy as np
import pandas as pd
import re
import time 
import json
from selenium import webdriver
import bs4
import tqdm
#%%
#Ler os dados salvos no coletor de dados das páginas de busca
dados_brutos = pd.read_json('dados_extraidos_paginas.json', lines = True)
dados_brutos = dados_brutos.drop_duplicates(subset=['link'], keep='first')
dados_brutos.head()
# %%
#Salvar os dados individuais da página de cada anúncio do imóvel
driver = webdriver.Chrome() #Abrir Chrome
for i in tqdm.tqdm_notebook(dados_brutos.index):
    url = dados_brutos.loc[i,'link']
    cidade = dados_brutos.loc[i,'cidade']
    driver.get(url) #Acessar a página no navegador
    print('link {}'.format(i))
    
    #Cria um arquivo individual para cada 
    #with open('./Dados_brutos_imoveis/link_{}.html'.format( i), 'w+') as output:
    #    output.write(driver.page_source) #Salvat o HTML da página
    time.sleep(1.5)
driver.quit()
# %%
#Acessar cada uma das páginas baixadas e captar os dados que forem importantes
with open('informacao_anuncios.json', 'w+') as output: 
    for pagina in tqdm.tqdm_notebook(dados_brutos.index):
        #print('-----------------------------------')
        #print('loop_{}'.format(pagina))
        #Abrir o arquivo, setar apenas leitura 'r+' e colocar como inp (input)
        with open('./Dados_brutos_imoveis/link_{}.html'.format( pagina), 'r+') as inp:
            #Ler arquivo e transformar o HTML para formato legível (bs4)
            page_html = inp.read()
            parsed = bs4.BeautifulSoup(page_html)
            #Identificar as tags importantes para captar as informações necessárias
            tags_h2 = parsed.findAll('h2') #Tags que contém informação de Preço
            tags_div = parsed.findAll('div') #Tags que contém inforações gerais do anúncio
            dados = dict()

            #Testar se a página do imóvel foi localizada ou não:
            for t_h2 in tags_h2:
                if t_h2['class'] == ['sc-ifAKCX', 'ixpykR']:
                    '''
                    Quando existe essa classe dentro da tag h2, significa que  página não existe
                    então passamos para o próximo loop
                    '''
                    continue
                #Pegar informações de preço
                if t_h2['class'] == ['sc-ifAKCX', 'sc-1wimjbb-0', 'eRiLRs']:
                    tag_preco = str(t_h2)
                    try:
                        preco = re.search(r'>R\$.(.*?)<', tag_preco).group(1)
                        #print('Preço = {}'.format(preco))
                        dados['preco'] = preco
                    except:
                        continue
                    
            #Pegar valores de Categoria e quarto
            try: 
                '''
                O melhor caminho é por essas tags div. 
                Cada um desses t_div dessas classes são identificáveis pelo item  e quantidade dentro do padrão
                '''

                for t_div in tags_div:
                    try:
                        #Identificar somente as tags que contenham a informação dada
                        if t_div['class'] == ['sc-jTzLTM', 'sc-hmzhuo', 'sc-1f2ug0x-3', 'jcodVG']:
                            #Transformar o t_div em string para poder utilizar o re (originalmente o t_div é um bs4 object)
                            info = str(t_div)
                            #print(info)
                            
                            #Identificar individualmente cada informação e separar
                            #Buscar categoria
                            if bool(re.search('>Categoria<', info)):
                                categoria = re.search(r'>(.{1,40}?)</a', info).group(1) #limitei a 40 caracteres ao máximo para não pegar a tag inteira
                                #print('Categoria = {}'.format(categoria))
                                dados['categoria'] = categoria
                            
                            #Buscar tipo
                            if bool(re.search('>Tipo<', info)):
                                tipo = re.search(r'>(.{1,40}?)</a', info).group(1) 
                                #print('Tipo = {}'.format(tipo))
                                dados['tipo'] = tipo
                            
                            #Buscar Área
                            if bool(re.search('>Área útil<', info)):
                                area = re.search(r'>(.{1,40}?)m²</dd', info).group(1) 
                                #print('Área = {}'.format(area))
                                dados['area'] = area
                            
                            #Buscar Quartos
                            if bool(re.search('>Quartos<', info)):
                                quartos = re.search(r'>(.{1,40}?)</a', info).group(1) 
                                #print('Quartos = {}'.format(quartos))
                                dados['quartos'] = quartos
                            
                            #Buscar Banheiros
                            if bool(re.search('>Banheiros<', info)):
                                banheiros = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('Banheiros = {}'.format(banheiros))
                                dados['banheiros'] = banheiros
                                
                            #Buscar Vagas de Garagem
                            if bool(re.search('>Vagas na garagem<', info)):
                                vagas = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('Vagas = {}'.format(vagas))
                                dados['vagas'] = vagas
                            
                            #Buscar Detalhes do imóvel
                            if bool(re.search('>Detalhes do imóvel<', info)):
                                detalhes = re.search(r'>(.{1,100}?)</dd', info).group(1) 
                                #print('Detalhes = {}'.format(detalhes))
                                dados['detalhes_imovel'] = detalhes
                            
                            #Buscar Valor do Condomínio
                            if bool(re.search('>Condomínio<', info)):
                                valor_condominio = re.search(r'>R\$.(.*?)<', info).group(1) 
                                #print('Valor condomínio = {}'.format(valor_condominio))
                                dados['valor_condominio'] = valor_condominio
                                
                            #Buscar Detalhes do condomínio
                            if bool(re.search('>Detalhes do condominio<', info)):
                                detalhes_condominio = re.search(r'>(.{1,100}?)</dd', info).group(1) 
                                #print('Detalhes Condomínio = {}'.format(detalhes_condominio))
                                dados['detalhes_condominio'] = detalhes_condominio
                                
                            #Buscar CEP
                            if bool(re.search('>CEP<', info)):
                                cep = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('CEP = {}'.format(cep))
                                dados['cep'] = cep
                            
                            #Buscar Cidade
                            if bool(re.search('>Município<', info)):
                                cidade = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('Cidade = {}'.format(cidade))
                                dados['cidade'] = cidade
                            
                            #Buscar Bairro
                            if bool(re.search('>Bairro<', info)):
                                bairro = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('Bairro = {}'.format(bairro))
                                dados['bairro'] = bairro
                            
                            #Buscar Rua
                            if bool(re.search('>Logradouro<', info)):
                                rua = re.search(r'>(.{1,40}?)</dd', info).group(1) 
                                #print('Rua = {}'.format(rua))
                                dados['rua'] = rua
                    except:
                        continue
            except:
                print('erro na localização das tags')
                continue
            #output.write('{}\n'.format(json.dumps(dados)))
                
# %%
df = pd.read_json('informacao_anuncios.json', lines = True)
df.head(10)
# %%
