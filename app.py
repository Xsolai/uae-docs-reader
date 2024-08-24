import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
# from models import front.pt, back.pt, certificate.pt, vehicle.pt
from fastapi.responses import JSONResponse
from io import BytesIO
import time
from datetime import datetime
import json

app = FastAPI()


model = YOLO(r'models/front.pt')
model_back = YOLO(r'models/back.pt')
new_model = YOLO(r'models/certificate.pt')
reader = easyocr.Reader(['en'])

def read_image(file_stream: BytesIO) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def certificate(img):
    class_names = {
        'inspection date': 'inspection date',
        'test certificate': 'test certificate'}
    detected_info = {
        "inspection date": None,
        "test certificate": None}
    results = new_model.predict(source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            pass_results = reader.readtext(crop_img)
            if pass_results:
                text = pass_results[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return detected_info


def driving(img):
    class_names = {
        'issue date': 'issue date',
        'date of birth': 'date of birth',
        'exp date': 'exp date',
        'nationality': 'nationality',
        'name': 'name',
        'licence-no': 'licence-no'}
    detected_info = {
        "issue date": None,
        "date of birth": None,
        "exp date": None,
        "nationality": None,
        "name": None,
        "licence-no": None}
    results = model.predict(source=img)
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
        'traffic code': 'traffic Code'}
    detected_info = {
        "traffic Code": None}
    results = model_back.predict(source=img)
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
        'name': 'name',
        'emirates ID': 'emirates ID',
        'exp date': 'exp date',
        'date of birth': 'date of birth'}
    detected_info = {
        'name': None,
        'emirates ID': None,
        'exp date': None,
        'date of birth': None}
    results = model.predict(source=img)
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
        'place of issue': 'place of issue'}
    detected_info = {
        "employer": None,
        "occupation": None,
        "card-number": None,
        "place of issue": None}
    results = model_back.predict(source=img)
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
        'vehicle license':"vehicle license",
        'reg date' : 'reg date',
        'exp date':'exp date',
        'ins date': 'ins date',
        'owner': 'owner'}
    detected_info = {
        "vehicle license" : None,
        "reg date": None,
        "exp date":None,
        "ins date": None,
        "owner": None}
    results = model.predict(source=img)
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
        'veh type': 'veh type'}
    detected_info = {
        "model": None,
        "chassis no": None,
        "origin": None,
        "eng no": None,
        "veh type": None}
    results = model_back.predict(source=img)
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
    return detected_info

def trade(img):
    class_names = {
        'trade name': 'trade name',
        'issue date': 'issue date',
        'exp date': 'exp date',
        "commercial license": "commercial license"}
    detected_info = {
        "trade name": None,
        "issue date": None,
        "exp date": None,
        "commercial license": None}
    results = new_model.predict(source=img)
    for result in results:
        boxes = result.boxes.xyxy
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            trades = reader.readtext(crop_img)
            if trades:
                detected_text = trades[0][1].strip().lower()
                class_name = result.names[int(cls)].lower()
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = detected_text
                else:
                    print("no text box in class")
    return detected_info

def detect_document_type(img):
    results = model.predict(source=img) 
    detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
   
    back_res = model_back.predict(source=img)   
    detected_back_classes = [back_res[0].names[int(cls)] for cls in back_res[0].boxes.cls.cpu().numpy()]    
    
    certificate_doc=new_model.predict(source=img)
    new_classes = [certificate_doc[0].names[int(cls)] for cls in certificate_doc[0].boxes.cls.cpu().numpy()]

    if any("emirates ID" in cls for cls in detected_classes):
        return "front",id(img)
    elif any("licence-no" in cls for cls in detected_classes):
        return "front",driving(img)
    elif any("vehicle license" in cls  in cls for cls in detected_classes):
        return "front",vehicle(img)
    
    if any("veh type" in cls for cls in detected_back_classes):
        return "back",back_vehic(img)
    elif any("traffic code" in cls for cls in detected_back_classes):
        return "back",back_driving(img)
    elif any("card-number" in cls for cls in detected_back_classes):
        return "back",id_back(img)
    
    if any("commercial license" in cls for cls in new_classes):
        return "front", trade(img)
    elif any("test certificate" in cls for cls in new_classes):
        return "front", certificate(img)
    
    return {"message": "Document type not recognized"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        img = read_image(BytesIO(await file.read()))
        image_height, image_width = img.shape[:2]
        tokens_used = (image_height * image_width) // 1000

        detected_info = detect_document_type(img)

        processing_time = time.time() - start_time
        response_data = {
            "metadata": {
                "Side": "front" if any("front" in cls for cls in detect_document_type(img)) else "back",
                "PTime": f"{processing_time:.2f} seconds",
                "Tokens_Used": tokens_used,
                "Timestamp": datetime.now().isoformat()
            },
            "data": detected_info
        }
        return JSONResponse(content=response_data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
