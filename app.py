from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import math
from sklearn.metrics import mean_squared_error
import os
from model import prepare_data, create_model

app = Flask(__name__)

# Load the data
df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=["Date"])

# Preset configurations
PRESETS = {
    'Better Performance': [50, 25],
    'Balanced': [100, 50, 25],
    'Better Accuracy': [200, 100, 50, 25]
}

@app.route('/')
def index():
    return render_template('index.html', presets=PRESETS.keys())

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model']
    preset = request.form['preset']
    epochs = int(request.form['epochs'])
    
    neurons = PRESETS[preset]
    
    X_train, y_train, test_set, sc = prepare_data(df)
    
    model = create_model(model_type, neurons, (X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=epochs, batch_size=32)

    # Prepare test data
    dataset_total = pd.concat((df["High"][:'2016'], df["High"]['2017':]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Predict stock prices
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(test_set, color='red', label='Real AAPL Stock Price')
    plt.plot(predicted_stock_price, color='blue', label='Predicted AAPL Stock Price')
    plt.title('AAPL Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AAPL Stock Price')
    plt.legend()
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Calculate performance metrics
    rmse = math.sqrt(mean_squared_error(test_set, predicted_stock_price))
    r2 = r2_score(test_set, predicted_stock_price)
    
    return jsonify({
        'rmse': rmse,
        'r2': r2,
        'plot_path': plot_path
    })

if __name__ == '__main__':
    app.run(debug=True)
