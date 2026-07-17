from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/output"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    if "video" not in request.files:
        return "No video uploaded"

    file = request.files["video"]

    if file.filename == "":
        return "Please choose a video"

    input_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(input_path)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "output.mp4"
    )

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    return render_template(
        "index.html",
        output_video="output/output.mp4"
    )


if __name__ == "__main__":
    app.run(debug=True)