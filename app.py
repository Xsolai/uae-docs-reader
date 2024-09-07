#importing the required libraries
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
import warnings
warnings.filterwarnings("ignore")


# FastAPI instance
app = FastAPI(
    title="Emirates ID Reader",
    description="This API can extract data from Emirates ID, Driving Licence, Vehicle Licence, Trade, and Pass documents."
)


# Load the YOLO models from models directory
driving_model = YOLO("models/driving_front_back.pt")
id_model = YOLO("models/ID_front_back.pt")
vehicle_model = YOLO("models/vehicle_front_back.pt")
pass_model = YOLO("models/pass.pt")
trade_model = YOLO("models/trade.pt")


# Load the OCR reader
reader = easyocr.Reader(['en'])
ar_en_reader = easyocr.Reader(['ar', 'en'])


# Function to classify the document, than cropped and rotated the document and send it to it's respective model
def process_file(file_path: str, model_path: str = 'models/classify.pt', cropped_dir: str = 'cropped_images', oriented_dir: str = 'oriented_images'):
    """
    Processes the given file (PDF or image), detects objects using the YOLO model,
    crops the detected regions, and corrects the orientation of the cropped images.

    Args:
        file_path (str): The path to the PDF or image file.
        model_path (str): The path to the YOLO model (default: 'classify.pt').
        cropped_dir (str): The directory to save cropped images (default: 'cropped_images').
        oriented_dir (str): The directory to save oriented images (default: 'oriented_images').

    Returns:
        List of paths to the processed images.
    """

    # Load the trained YOLO model
    model_classify = YOLO(model_path)

    # Create directories for saving cropped and oriented images
    os.makedirs(cropped_dir, exist_ok=True)
    os.makedirs(oriented_dir, exist_ok=True)

    # Rotation map to correct orientations
    rotation_map = {
        '0': 0,
        '90': 270,  # Rotating 270 degrees is equivalent to rotating -90 degrees
        '180': 180,
        '270': 90,  # Rotating 90 degrees is equivalent to rotating -270 degrees
    }

    def process_pdf(pdf_path, dpi=300):
        """
        Convert each page of the PDF into a high-quality image using PyMuPDF.
        
        Args:
        pdf_path (str): Path to the PDF file
        dpi (int): Dots per inch for image resolution (default: 300)
        
        Returns:
        list: Paths to the extracted images
        """
        doc = fitz.open(pdf_path)
        image_paths = []
        
        for i in range(len(doc)):
            page = doc[i]
            
            # Set the matrix for higher resolution
            zoom = dpi / 72  # 72 is the default PDF resolution
            mat = fitz.Matrix(zoom, zoom)
            
            # Get the pixmap using the matrix for higher resolution
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save the image with high quality
            img_path = f"{os.path.splitext(pdf_path)[0]}_page_{i + 1}.png"
            img.save(img_path, format="PNG", dpi=(dpi, dpi), quality=95)
            
            image_paths.append(img_path)
        
        doc.close()
        return image_paths


    def process_image(image_path):
        """Process a single image for detection, cropping, and orientation correction."""
        results = model_classify(source=image_path, save=True, conf=0.5)
        processed_images = []
        for i, result in enumerate(results):
            img = Image.open(result.path)
            for j, box in enumerate(result.boxes.xyxy):
                class_idx = int(result.boxes.cls[j].item())
                class_name = result.names[class_idx]

                # Extract document type, side, and orientation from the class name
                parts = class_name.split('_')
                
                if len(parts) == 3:
                    doc_type, side, orient = parts
                    # Save cropped and oriented images with proper naming
                    xmin, ymin, xmax, ymax = map(int, box)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_cropped.jpg'
                    cropped_img_path = os.path.join(cropped_dir, cropped_img_name)
                    cropped_img.save(cropped_img_path)
                    processed_images.append(cropped_img_path)

                    if orient in rotation_map:
                        rotation_angle = rotation_map[orient]
                        if rotation_angle != 0:
                            cropped_img = cropped_img.rotate(rotation_angle, expand=True)

                    oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_oriented.jpg'
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    cropped_img.save(oriented_img_path)
                    processed_images.append(oriented_img_path)

                else:
                    doc_type, orient = parts[0], parts[1]
                    side = 'front'  # No side information for certificates

                    # Save the image as it is in cropped_dir
                    non_cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_non_cropped.jpg'
                    non_cropped_img_path = os.path.join(cropped_dir, non_cropped_img_name)
                    img.save(non_cropped_img_path)
                    processed_images.append(non_cropped_img_path)
                    
                    
                    # Save the image as it is in oriented_dir (no rotation)
                    oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_oriented.jpg'
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    img.save(oriented_img_path)
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




# Function for ID card detection and extraction
def id(img):
    class_names = {
        'Name': 'Name',
        'ID Number': 'ID Number',
        'Expiry Date': 'Expiry Date',
        'Date of birth': 'Date of birth',
        'Nationality': 'Nationality',
        'Card Number': 'Card Number',
        'Employer': 'Employer',
        'Occupation': 'Occupation',
        'Place of issue': 'Place of issue',
        'Issue Date' : 'Issue Date'
        }
    detected_info = {
        'Name': None,
        'ID Number': None,
        'Expiry Date': None,
        'Date of birth': None,
        'Nationality': None,
        'Card Number': None,
        'Employer': None,
        'Occupation': None,
        'Place of issue': None,
        'Issue Date': None
    }
    results = id_model.predict(img, line_width=2)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ID_results = reader.readtext(crop_img)       
            if ID_results:
                text = ID_results[0][1].strip()
                class_name = result.names[int(cls)]
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    # Remove any null values from the detected_info dictionary
    return {k: v for k, v in detected_info.items() if v is not None}



# Function for driving license detection and extraction
def driving(img):
    class_names = {
        'Customer Name': 'Customer Name',
        'DOB': 'DOB',
        'Expiry date': 'Expiry date',
        'Issue Date': 'Issue Date',
        'License No': 'License No',
        'Nationality': 'Nationality',
        'Place of Issue': 'Place of Issue',
        'Traffic Code No': 'Traffic Code No'
    }

    detected_info = {
        'Customer Name': None,
        'DOB': None,
        'Expiry date': None,
        'Issue Date': None,
        'License No': None,
        'Nationality': None,
        'Place of Issue': None,
        'Traffic Code No': None
    }

    results = driving_model.predict(img, line_width=2)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ID_results = reader.readtext(crop_img)       
            if ID_results:
                text = ID_results[0][1].strip()
                class_name = result.names[int(cls)]
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    # Remove any null values from the detected_info dictionary
    return {k: v for k, v in detected_info.items() if v is not None}



# Function for vehicle registration detection and extraction
def vehicle(img):
    class_names = {
        "TC no":  "TC no",
        "Insurance company": "Insurance company",
        'Reg date' : 'Reg date',
        'Exp date':'Exp date',
        'Ins Exp': 'Ins Exp',
        'Owner': 'Owner',
        "place of issue":'place of issue',
        "nationality":'nationality',
        "Model":'Model',
        "Origin":'Origin',
        "veh type":'veh type',
        "Eng no":'Eng no',
        "chassis no":'chassis no'}
    
    detected_info = {
        "TC no": None,
        "Insurance company": None,
        'Reg date': None,
        'Exp date': None,
        'Ins Exp': None,
        'Owner': None,
        "place of issue": None,
        "nationality": None,
        "Model": None,
        "Origin": None,
        "veh type": None,
        "Eng no": None,
        "chassis no": None}
    

    results = vehicle_model.predict(img, line_width=1)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            veh_results = ar_en_reader.readtext(crop_img)
            if veh_results:
                text = veh_results[0][1].strip()
                class_name = result.names[int(cls)]
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    return {k: v for k, v in detected_info.items() if v is not None}


# Function for Pass
def pass_certificate(img):
    class_names = {
        'inspection date': 'inspection date'
        }
    detected_info = {
        'inspection date': None
    }
    results = pass_model.predict(img, line_width=2)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ID_results = reader.readtext(crop_img)       
            if ID_results:
                text = ID_results[0][1].strip()
                class_name = result.names[int(cls)]
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    # Remove any null values from the detected_info dictionary
    return {k: v for k, v in detected_info.items() if v is not None}
      


def trade_certificate(img):
    class_names = {
        'Trade Name': 'trade name',
        'Issue Date': 'issue date',
        'Exp Date': 'exp date',
        'activity': 'activity'
    }
    detected_info = {
        'trade name': None,
        'issue date': None,
        'exp date': None,
        'activity': None
    }

    results = trade_model.predict(img, line_width=2)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box, cls in zip(boxes, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            crop_img = img[y1:y2, x1:x2]
            crop_img = cv2.resize(crop_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ID_results = reader.readtext(crop_img)       
            if ID_results:
                text = ID_results[0][1].strip()
                class_name = result.names[int(cls)]
                if class_name in class_names:
                    key = class_names[class_name]
                    detected_info[key] = text
    # Remove any null values from the detected_info dictionary
    return {k: v for k, v in detected_info.items() if v is not None}


# Upload endpoint for processing the uploaded file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (PDF or image) and processes it to detect objects using the YOLO model,
    crops the detected regions, and corrects the orientation of the cropped images.

    Args:
        file (UploadFile): The file to upload.

    Returns:
        JSONResponse: The response containing the paths to the processed images.
    """
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
        oriented_files = [file for file in processed_files if 'oriented' in file]

        # Process each oriented image
        for oriented_file in oriented_files:
            img = cv2.imread(oriented_file)
            img_np = np.array(img)

            image_height, image_width = img_np.shape[:2]
            tokens_used = (image_height * image_width) // 1000

            # Extract document type from the file name
            file_name = os.path.basename(oriented_file)
            doc_type = file_name.split('_')[0]

            # enhanced the image for better results
            # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if 'ID' in oriented_file:
                detected_info = id(img)
            elif 'Driving' in oriented_file:
                detected_info = driving(img)
            elif 'vehicle' in oriented_file:
                detected_info = vehicle(img)
            elif 'pass' in oriented_file:
                detected_info = pass_certificate(img)
            elif 'trade' in oriented_file:
                detected_info = trade_certificate(img)
            else:
                detected_info = {}

            # Compile the result for the current image
            image_result = {
                "image_metadata": {
                    "Image_Path": oriented_file,
                    "Document_Type": doc_type,
                    "side": "front" if "front" in oriented_file else "back",
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
                "Total_Tokens_Used": sum([result['image_metadata']['Tokens_Used'] for result in image_results]),
                "Timestamp": datetime.now().isoformat()
            },
            "images_results": image_results
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Remove the temporary file
        os.remove(temp_file_path)

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
