from flask import Flask, render_template, request
from PIL import Image
from model import predict

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_disease():

   
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

   
    if file.filename == "":
        return "No image selected"

 
    image = Image.open(file.stream)

   
    results = predict(image)

   
    return render_template(
        "result.html",
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)
