from flask import Flask, render_template, request
from PIL import Image
from model import predict

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_disease():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    # Check whether a file was actually selected
    if file.filename == "":
        return "No image selected"

    # Open the uploaded image
    image = Image.open(file.stream)

    # Send image to our AI model
    results = predict(image)

    # Send prediction back to webpage
    return render_template(
        "result.html",
        results=results
    )


if __name__ == "__main__":
    app.run(debug=True)