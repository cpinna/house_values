# importing Flask and other modules
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib as jb

# Flask constructor
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def dados_ap():
    if request.method == "POST":
        # Número de quartos
        n_quartos = request.form.get('n_quartos')

        # Número de Banheiros
        n_banheiros = request.form.get('n_banheiros')

        # Área
        area = request.form.get('area')

        # Vagas de Garagem
        vagas = request.form.get('garagem')

        # Vagas de Garagem
        cidade = request.form.get('cidade')

        # Bairro
        bairro = request.form.get('bairro')

        # Tipo de imóvel
        tipo = request.form.get('tipo')

        # Condomínio
        condominio = request.form.get('condominio')

        # Churrasqueira
        churrasqueira = request.form.get('churrasqueira')
        churrasqueira = np.where(churrasqueira == 'on', 1, 0)

        # Armários na Cozinha
        armarios = request.form.get('armarios')
        armarios = np.where(armarios == 'on', 1, 0)

        # Varanda
        varanda = request.form.get('varanda')
        varanda = np.where(varanda == 'on', 1, 0)

        # Piscina
        piscina = request.form.get('piscina')
        piscina = np.where(piscina == 'on', 1, 0)

        # Salão
        salao = request.form.get('salao')
        salao = np.where(salao == 'on', 1, 0)

        # Segurança
        seguranca = request.form.get('seguranca')
        seguranca = np.where(seguranca == 'on', 1, 0)

        # Portaria
        portaria = request.form.get('portaria')
        portaria = np.where(portaria == 'on', 1, 0)

        # Elevador
        elevador = request.form.get('elevador')
        elevador = np.where(elevador == 'on', 1, 0)

        # Academia
        academia = request.form.get('academia')
        academia = np.where(academia == 'on', 1, 0)

        # Montar dados para transformar em DF:
        data = {
            'valor_condominio': [condominio],
            'area': [area],
            'quartos': [n_quartos],
            'banheiros': [n_banheiros],
            'vagas': [vagas],
            'cidade': [cidade],
            'bairro': [bairro],
            'tipo': [tipo],
            'churrasqueira': [churrasqueira],
            'armarios_cozinha': [armarios],
            'varanda': [varanda],
            'piscina': [piscina],
            'salao': [salao],
            'seguranca': [seguranca],
            'portaria': [portaria],
            'elevador': [elevador],
            'academia': [academia]
        }

        # Montar Dataframe
        df = pd.DataFrame(data=data)
        # Carregar .pkl
        random_forest = jb.load('random_forest.pkl.z')

        predict_value = random_forest.predict(df)
        valor = np.round(predict_value[0], 2)

        return f'O valor previsto para o imóvel é de R${valor:,}'
    return render_template("form.html")


if __name__ == '__main__':
    app.run(debug=True)
