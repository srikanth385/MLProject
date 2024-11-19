from flask import Flask, request, render_template
import pickle
import numpy as np

# Create Flask app i.e. the instance of our app
app = Flask(__name__)

# Load the Pickle model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')         # mapping the URLs to a specific function that will handle the logic for that URL.
def hello_world():      # the URL (“/”) is associated with the root URL.
    return render_template('forest.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('forest.html',
                               pred='Your Forest is in Danger.\nProbability of fire occurring is {}'.format(output))
    else:
        return render_template('forest.html',
                               pred='Your Forest is safe.\n Probability of fire occurring is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
