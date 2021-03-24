import flask
import pickle

app = Flask(__name__)
model = pickle.load(open('final_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = ???

    prediction = model.predict(features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Estimated price is $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
