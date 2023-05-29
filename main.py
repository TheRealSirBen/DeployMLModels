import os
import uuid

import cv2
import torch
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse
from numpy import squeeze

# Instantiate Application
app = FastAPI()

# Instantiate global variables
MODEL_FILE_TYPES = [".pt"]
IMAGE_FILE_TYPES = [".jpg", ".jpeg"]
VIDEO_FILE_TYPES = [".mp4"]
SUPPORTED_FILE_TYPES = tuple(IMAGE_FILE_TYPES + VIDEO_FILE_TYPES)
MODELS_FOLDER = 'models'
UPLOADED_FILES_FOLDER = 'uploaded-files'
PREDICTIONS_FOLDER = 'predictions'


async def process_video(uploaded_file_path, uploaded_file_with_extension, model):
    video = cv2.VideoCapture(uploaded_file_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(PREDICTIONS_FOLDER, uploaded_file_with_extension)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model(frame)
        # Write the processed frame to the output video file
        writer.write(squeeze(results.render()))

    video.release()
    writer.release()

    return output_path


async def process_image(model, uploaded_file_path, uploaded_file_with_extension):
    results = model(uploaded_file_path)

    results_path = os.path.join(PREDICTIONS_FOLDER, uploaded_file_with_extension)
    results.save(save_dir=PREDICTIONS_FOLDER, exist_ok=True)

    return results_path


@app.get("/")
async def index():
    return "Hello, you are welcome"


@app.post('/detect/{model_id}')
async def object_detection_on_image_or_video(model_id: str, file: UploadFile = File()):
    # Check file size
    max_size = 32 * 1024 * 1024  # 32 MB
    if file.size > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 32 MB")

    # Get model filename
    model_file = str()
    for file_type in MODEL_FILE_TYPES:
        file_path = os.path.join(MODELS_FOLDER, model_id + file_type)
        if os.path.exists(file_path):
            model_file = file_path

    if model_file == str():
        raise HTTPException(status_code=400, detail='Model provided does not exist, you can upload model first')

    # Check File Type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_FILE_TYPES)} files are allowed")

    # Check if the uploaded files folder exists, and create it if it doesn't
    if not os.path.exists(UPLOADED_FILES_FOLDER):
        os.makedirs(UPLOADED_FILES_FOLDER)

    # Generate a unique filename using UUID4 with the same extension as the uploaded file
    uploaded_file_name = str(uuid.uuid4())
    uploaded_file_with_extension = uploaded_file_name + file_ext

    # Save the file to the uploaded files folder
    uploaded_file_path = os.path.join(UPLOADED_FILES_FOLDER, uploaded_file_with_extension)
    with open(uploaded_file_path, "wb") as buffer:
        buffer.write(await file.read())

        # Check if the uploaded files folder exists, and create it if it doesn't
        if not os.path.exists(PREDICTIONS_FOLDER):
            os.makedirs(PREDICTIONS_FOLDER)

    # Make Predictions
    results_path, model = str(), torch.hub.load('ultralytics/yolov5', 'custom', path=model_file, verbose=False)

    if file_ext in IMAGE_FILE_TYPES:
        results_path = await process_image(model, uploaded_file_path, uploaded_file_with_extension)

    if file_ext in VIDEO_FILE_TYPES:
        results_path = await process_video(uploaded_file_path, uploaded_file_with_extension, model)

    return FileResponse(results_path, media_type="application/octet-stream", filename=file.filename)


@app.post("/upload-model")
async def upload_yolo_model(file: UploadFile = File()):
    # Check File Type
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in MODEL_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(MODEL_FILE_TYPES)} files are allowed")

    # Check if the models folder exists, and create it if it doesn't
    if not os.path.exists(MODELS_FOLDER):
        os.makedirs(MODELS_FOLDER)

    # Generate a unique filename using UUID4 with the same extension as the uploaded file
    file_name = str(uuid.uuid4())
    file_name_with_extension = file_name + file_ext

    # Save the file to the models folder
    save_location = os.path.join(MODELS_FOLDER, file_name_with_extension)
    with open(save_location, "wb") as buffer:
        buffer.write(await file.read())

    return {"message": "File Successfully Uploaded", "data": file_name}
