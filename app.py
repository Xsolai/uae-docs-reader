import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import time
from datetime import datetime
import json
import logging

logging.basicConfig(filename="gfg-log.log", filemode="w", format="%(name)s → %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler('logs.log')
logger.addHandler(FileOutputHandler)

app = FastAPI()

model = YOLO(r'D:/Saincube task/emirates project/front.pt')
model_back = YOLO(r'D:/Saincube task/emirates project/back.pt')
new_model = YOLO(r'D:/Saincube task/emirates project/certificate.pt')
vehicle_model = YOLO("D:/Saincube task/emirates project/models/vehicle.pt")
reader = easyocr.Reader(['en'])

def read_image(file_stream: BytesIO) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def certificate(img):
    class_names = {
        'inspection date': 'inspection date',
        'test certificate': 'test certificate'
    }
    detected_cert = {
        "inspection date": None,
        "test certificate": None
    }
    results = new_model.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            try:
                pass_results = reader.readtext(crop_img)
                if pass_results:
                    text = pass_results[0][1].strip().lower()
                    class_name = result.names[int(cls)].lower()
                    if class_name in class_names:
                        key = class_names[class_name]
                        detected_cert[key] = text
            except Exception as e:
                print(f"Error processing box: {e}")
    return detected_cert

def driving(img):
    class_names = {
        'issue date': 'issue date',
        'date of birth': 'date of birth',
        'exp date': 'exp date',
        'nationality': 'nationality',
        'name': 'name',
        'licence-no': 'licence-no'
    }
    detected_info = {
        "issue date": None,
        "date of birth": None,
        "exp date": None,
        "nationality": None,
        "name": None,
        "licence-no": None
    }
    results = model.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            dr_results = reader.readtext(crop_img)
            if dr_results:
                text = dr_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

def back_driving(img):
    class_names = {
        'traffic code': 'traffic Code'
    }
    detected_info = {
        "traffic Code": None
    }
    results = model_back.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            back_driving_results = reader.readtext(crop_img)
            if back_driving_results:
                text = back_driving_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

def id(img):
    class_names = {
        'name': 'Name',
        'emirates id': 'emirates ID',
        'date of birth': 'date of birth',
        'exp date': 'exp date'
    }
    detected_info = {
        "Name": None,
        "emirates ID": None,
        "date of birth": None,
        "exp date": None,
    }
    results = model.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            ID_results = reader.readtext(crop_img)
            if ID_results:
                text = ID_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

def id_back(img):
    class_names = {
        'employer': 'employer',
        'occupation': 'occupation',
        'card-number': 'card-number',
        'place of issue': 'Place of issue'
    }
    detected_info = {
        "Employer": None,
        "Occupation": None,
        "card-number": None,
        "Place of issue": None
    }
    results = model_back.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            back_id_results = reader.readtext(crop_img)
            if back_id_results:
                text = back_id_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

def vehicle(img):
    class_names = {
        'place of issue': 'place of issue',
        'vehicle license': 'vehicle license',
        'insurance company': 'insurance company',
        'reg date': 'reg date',
        'TC': 'TC',
        'ins date': 'ins date',
        'owner': 'owner'
    }
    detected_info = {
        "place of issue": None,
        "vehicle license": None,
        "insurance Company": None,
        "reg date": None,
        "TC": None,
        "ins date": None,
        "owner": None
    }
    results = vehicle_model.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            veh_results = reader.readtext(crop_img)
            if veh_results:
                text = veh_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

def back_vehic(img):
    class_names = {
        'model': 'model',
        'chassis no': 'chassis no',
        'origin': 'origin',
        'eng no': 'eng no',
        'veh type': 'veh type'
    }
    detected_info = {
        "model": None,
        "chassis no": None,
        "origin": None,
        "eng no": None,
        "veh type": None
    }
    results = model_back.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            back_vehicle_results = reader.readtext(crop_img)
            if back_vehicle_results:
                text = back_vehicle_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
            else:
                print(f"No text detected in box with class {result.names[int(cls)]}")
    return detected_info

def trade(img):
    class_names = {
        'trade name': 'trade name',
        'issue date': 'issue date',
        'expiry date': 'expiry date'
    }
    detected_info = {
        "trade name": None,
        "issue date": None,
        "expiry date": None
    }
    results = vehicle_model.predict(conf=0.7, source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            trade_results = reader.readtext(crop_img)
            if trade_results:
                text = trade_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_image(BytesIO(contents))
    start_time = time.time()

    predictions = {
        "certificate": certificate(img),
        "driving": driving(img),
        "back_driving": back_driving(img),
        "id": id(img),
        "id_back": id_back(img),
        "vehicle": vehicle(img),
        "back_vehic": back_vehic(img),
        "trade": trade(img)
    }

    end_time = time.time()
    metadata = {
        "processing_time": end_time - start_time,
        "timestamp": datetime.now().isoformat()
    }
    logger.info(f"Predictions: {predictions}, Metadata: {metadata}")
    return JSONResponse(content={"predictions": predictions, "metadata": metadata})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
