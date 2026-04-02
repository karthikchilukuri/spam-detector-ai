from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Store history (temporary memory)
history = []

@app.route("/", methods=["GET", "POST"])
def home():
    global history
    prediction = None

    if request.method == "POST":
        message = request.form["message"]
        msg_vec = vectorizer.transform([message])

        result = model.predict(msg_vec)[0]
        prob = model.predict_proba(msg_vec)[0][1]

        if result == 1:
            prediction = f"🚨 Spam ({prob:.2f})"
        else:
            prediction = f"✅ Not Spam ({1 - prob:.2f})"

        # Save history
        history.insert(0, {"message": message, "prediction": prediction})

    return render_template("index.html", prediction=prediction, history=history)

if __name__ == "__main__":
    app.run(debug=True)