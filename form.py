# importing Flask and other modules
from flask import Flask, request, render_template

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
        return f"{n_quartos} quartos, {n_banheiros} banheiros, {area} de área, {vagas} vagas, em {cidade}"
    return render_template("form.html")


if __name__ == '__main__':
    app.run(debug=True)
