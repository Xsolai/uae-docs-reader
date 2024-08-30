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
import os
from PIL import Image
import uvicorn
import fitz  # PyMuPDF
import shutil
import warnings
warnings.filterwarnings("ignore")




app = FastAPI(
    title="Emirates ID Reader",
    description="This API can extract data from Emirates ID, Driving Licence, Commercial, and Vehicle Registration."
)


model = YOLO(r'models/front.pt')
model_back = YOLO(r'models/back.pt')
new_model = YOLO(r'models/certificate.pt')
reader = easyocr.Reader(['en'])
reader_vehicle=easyocr.Reader(['en','ar'])






# Rotation map to correct orientations
rotation_map = {
    '0': 0,
    '90': 270,  # Rotating 270 degrees is equivalent to rotating -90 degrees
    '180': 180,
    '270': 90,  # Rotating 90 degrees is equivalent to rotating -270 degrees
}

def process_file(file_path: str, model_path: str = 'models/classification.pt'):
    """
    Processes the given file (PDF or image), detects objects using the YOLO model,
    crops the detected regions, and corrects the orientation of the cropped images.

    Args:
        file_path (str): The path to the PDF or image file.
        model_path (str): The path to the YOLO model (default: 'classification.pt').

    Returns:
        List of paths to the processed images.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Create directories for saving cropped and oriented images
    cropped_dir = 'cropped_images'
    oriented_dir = 'oriented_images'
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(oriented_dir, exist_ok=True)

    def process_pdf(pdf_path):
        """Convert each page of the PDF into an image using PyMuPDF."""
        doc = fitz.open(pdf_path)
        image_paths = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            img_path = f"{os.path.splitext(pdf_path)[0]}_page_{i + 1}.png"
            pix.save(img_path)
            image_paths.append(img_path)
        return image_paths

    def process_image(image_path):
        """Process a single image for detection, cropping, and orientation correction."""
        results = model(source=image_path, save=True, conf=0.5)
        processed_images = []
        for i, result in enumerate(results):
            img = Image.open(result.path)
            for j, box in enumerate(result.boxes.xyxy):
                class_idx = int(result.boxes.cls[j].item())
                class_name = result.names[class_idx]
                orient = class_name.split('_')[-1]
                xmin, ymin, xmax, ymax = map(int, box)
                cropped_img = img.crop((xmin, ymin, xmax, ymax))

                cropped_img_name = f'{class_name}_{i}_{j}_cropped.jpg'
                cropped_img_path = os.path.join(cropped_dir, cropped_img_name)
                cropped_img.save(cropped_img_path)
                processed_images.append(cropped_img_path)

                if orient in rotation_map:
                    rotation_angle = rotation_map[orient]
                    if rotation_angle != 0:
                        cropped_img = cropped_img.rotate(rotation_angle, expand=True)

                oriented_img_name = f'{class_name}_{i}_{j}_oriented.jpg'
                oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                cropped_img.save(oriented_img_path)
                processed_images.append(oriented_img_path)

        return processed_images

    processed_files = []
    if file_path.endswith('.pdf'):
        image_paths = process_pdf(file_path)
        for img_path in image_paths:
            processed_files.extend(process_image(img_path))
    else:
        processed_files.extend(process_image(file_path))

    return processed_files











def read_image(file_stream: BytesIO) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def certificate(img):
    class_names = {
        'inspection date': 'inspection date'}
    detected_info = {
        "inspection date": None}
    results = new_model.predict(source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            results = reader.readtext(crop_img)
            if results:
                text = results[0][1].strip().lower()
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
        'emirates id': 'emirates ID',
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
        "tc":  "TC",
        "insurance company": "Insurance company",
        'vehicle license':"vehicle license",
        'reg date' : 'reg date',
        'exp date':'exp date',
        'ins date': 'ins date',
        'owner': 'owner',
        "place of issue":'place of issue',
        "nationality":'nationality'}
    detected_info = {
        "vehicle license" : None,
        "reg date": None,
        "exp date":None,
        "ins date": None,
        "owner": None,
        "Insurance company": None,
        "TC" : None,
        "place of issue":None,
        "nationality":None}
    results = model.predict(source=img)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]
            veh_results = reader_vehicle.readtext(crop_img)
            print("veh results:",veh_results)
            if veh_results:
                text = veh_results[0][1].strip()
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
        'exp date': 'exp date'
        }
    detected_info = {
        "trade name": None,
        "issue date": None,
        "exp date": None}
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
    print("new_classes:",new_classes)

    if any("emirates ID" in cls for cls in detected_classes):
        return "front",id(img)
    elif any("licence-no" in cls for cls in detected_classes):
        return "front",driving(img)
    elif any("vehicle license" in cls  in cls for cls in detected_classes):
        return "front",vehicle(img)
    
    if any("model" in cls for cls in detected_back_classes):
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
        # Save the uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the file using the process_file method
        processed_files = process_file(temp_file_path)

        # Initialize a list to hold the detected information for each image
        image_results = []

        # Filter out and process only the oriented images
        for img_path in processed_files:
            if "oriented_images" in img_path:  # Ensure only oriented images are processed
                img = Image.open(img_path)
                img = img.convert('RGB')  # Ensure the image is in RGB format
                
                # Convert the PIL image to numpy array if detect_document_type expects it
                img_np = np.array(img)

                # Get the image dimensions for token calculation
                image_height, image_width = img_np.shape[:2]
                tokens_used = (image_height * image_width) // 1000

                # Detect document type
                detected_info = detect_document_type(img_np)

                # Compile the result for the current image
                image_result = {
                    "image_metadata": {
                        "Image_Path": img_path,
                        "Side": "front" if any("front" in cls for cls in detected_info) else "back",
                        "Tokens_Used": tokens_used
                    },
                    "detected_data": detected_info
                }

                # Append the result to the list of image results
                image_results.append(image_result)

        # Calculate overall processing time
        processing_time = time.time() - start_time

        # Compile the final response data
        response_data = {
            "overall_metadata": {
                "Total_PTime": f"{processing_time:.2f} seconds",
                "Total_Tokens_Used": sum(result["image_metadata"]["Tokens_Used"] for result in image_results),
                "Timestamp": datetime.now().isoformat()
            },
            "images_results": image_results
        }

        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        # Cleanup: Remove the temporary file
        os.remove(temp_file_path)
        # Optionally, clean up the cropped and oriented image directories if needed
        shutil.rmtree('cropped_images')
        shutil.rmtree('oriented_images')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
