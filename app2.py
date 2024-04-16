from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

app = Flask(__name__)

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1, -1)
    preds = model.predict(x)
    return preds

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Se carga el modelo
    with open('modelo2.pkl', 'rb') as file:
        model = pickle.load(file)

    # Lecctura de datos
    es = float(request.form['es'])
    ma = float(request.form['ma'])
    os = float(request.form['os'])
    c = float(request.form['c'])
    adn = float(request.form['adn'])
    tp = float(request.form['tp'])
    tg = float(request.form['tg'])
    N = float(request.form['N'])
    cm = float(request.form['cm'])
    cp = float(request.form['cp'])
    pm = float(request.form['pm'])
    tap = float(request.form['tap'])
    noa = float(request.form['noa'])
    d = float(request.form['d'])
    nec = float(request.form['nec'])
    deu = float(request.form['deu'])
    matri = float(request.form['matri'])
    ge = float(request.form['ge'])
    beca = float(request.form['beca'])
    edad = float(request.form['edad'])
    inte = float(request.form['inte'])
    unoacre = float(request.form['unoacre'])
    unomatr = float(request.form['unomatr'])
    unoeva = float(request.form['unoeva'])
    unoapro = float(request.form['unoapro'])
    unogrd = float(request.form['unogrd'])
    unosin = float(request.form['unosin'])
    dosacre = float(request.form['dosacre'])
    dosmatr = float(request.form['dosmatr'])
    doseva = float(request.form['doseva'])
    dosapro = float(request.form['dosapro'])
    dosgrd = float(request.form['dosgrd'])
    dossin = float(request.form['dossin'])
    des = float(request.form['des'])
    infl = float(request.form['infl'])
    PIB = float(request.form['PIB'])

    # Realizar la predicción
    x_in = np.asarray([es, ma, os, c, adn, tp, tg, N, cm, cp, pm, tap, noa, d, 
            nec, deu, matri, ge, beca, edad, 
            inte, unoacre, unomatr, unoeva, unoapro, unogrd, unosin, 
            dosacre, dosmatr, doseva, dosapro, dosgrd, dossin, 
            des, infl, PIB])    
    prediction = model_prediction(x_in, model)

    if prediction == 0:
        status_message = "El estudiante está en estado de Abandono Escolar."
    elif prediction == 1:
        status_message = "El estudiante está Inscrito."
    elif prediction == 2:
        status_message = "El estudiante está Graduado."
    else:
        status_message = "Estado del estudiante desconocido."

    return status_message
    

if __name__ == '__main__':
    app.run(debug=True)
