from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle

app = Flask(__name__)

rf, lr, rf_acc, lr_acc = pickle.load(open("weather_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", rf_acc=rf_acc, lr_acc=lr_acc)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        df = pd.read_csv(file)

        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month

        X = df[['precipitation', 'temp_min', 'wind', 'month']]

        df['RF_Prediction'] = rf.predict(X)
        df['LR_Prediction'] = lr.predict(X)

        df['Condition'] = df['precipitation'].apply(
            lambda x: "Rainy 🌧️" if x > 0 else "Sunny ☀️"
        )

        df.to_csv("output.csv", index=False)

        table = df.head(20).to_html(classes='table', index=False)

        return render_template("index.html",
                               table=table,
                               rf_acc=rf_acc,
                               lr_acc=lr_acc,
                               download=True)

@app.route("/download")
def download():
    return send_file("output.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)