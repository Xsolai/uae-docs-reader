#importing the required libraries
import cv2
import easyocr
import numpy as np
import base64
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from pydantic import BaseModel
from typing import Optional
import time
import tempfile
from datetime import datetime
import json
import os
from PIL import Image
import uvicorn
import fitz  # PyMuPDF
import re
import together
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")


# Define the model for incoming JSON data
class FileData(BaseModel):
    data: str
    ext: str

# Pydantic model for the request body
class Base64Request(BaseModel):
    data: str
    extension: str

# FastAPI instance
app = FastAPI(
    title="Emirates ID Reader",
    description="This API can extract data from Emirates ID, Driving Licence, Vehicle Licence, Trade, and Pass documents."
)

load_dotenv()


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
            # Convert the image to RGB mode if it has an alpha channel
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            for j, box in enumerate(result.boxes.xyxy):
                class_idx = int(result.boxes.cls[j].item())
                class_name = result.names[class_idx]
                confidence = result.boxes.conf[j].item()
                confidence = round(confidence, 2)

                # Extract document type, side, and orientation from the class name
                parts = class_name.split('_')
                
                if len(parts) == 3:
                    doc_type, side, orient = parts
                    # Save cropped and oriented images with proper naming
                    xmin, ymin, xmax, ymax = map(int, box)
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_cropped.jpg'
                    cropped_img_path = os.path.join(cropped_dir, cropped_img_name)
                    cropped_img.save(cropped_img_path)
                    processed_images.append(cropped_img_path)

                    if orient in rotation_map:
                        rotation_angle = rotation_map[orient]
                        if rotation_angle != 0:
                            cropped_img = cropped_img.rotate(rotation_angle, expand=True)

                    oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_oriented.jpg'
                    oriented_img_path = os.path.join(oriented_dir, oriented_img_name)
                    cropped_img.save(oriented_img_path)
                    processed_images.append(oriented_img_path)

                else:
                    doc_type, orient = parts[0], parts[1]
                    side = 'front'  # No side information for certificates

                    # Save the image as it is in cropped_dir
                    non_cropped_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_non_cropped.jpg'
                    non_cropped_img_path = os.path.join(cropped_dir, non_cropped_img_name)
                    img.save(non_cropped_img_path)
                    processed_images.append(non_cropped_img_path)
                    
                    
                    # Save the image as it is in oriented_dir (no rotation)
                    oriented_img_name = f'{doc_type}_{side}_{orient}_{i}_{j}_{confidence}_oriented.jpg'
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




class DocumentOCRProcessor:
    def __init__(self, 
                 model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                 api_key=None):
        """
        Initialize the Document OCR Processor
        
        Args:
            model (str, optional): Vision model to use. 
            api_key (str, optional): Together.ai API key
        """
        # Load environment variables from .env file if available
        load_dotenv()

        # Use API key from parameter or environment
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')

        # Validate API key
        if not self.api_key:
            raise ValueError("Together.ai API key is required. Set TOGETHER_API_KEY in the .env file or pass it explicitly.")
        
        # Set the API key in the environment for Together client
        os.environ["TOGETHER_API_KEY"] = self.api_key

        # Initialize Together client
        try:
            self.client = together.Together()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Together client: {str(e)}")

        # Set the model
        self.model = model
        print(f"DocumentOCRProcessor initialized with model: {self.model}")



    def encode_image(self, image_path):
        """
        Encode image to base64
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')




    def _extract_json_from_response(self, response):
        # Look for standalone JSON objects in the response
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)

        if json_match:
            try:
                # Try to parse the matched JSON string
                parsed_json = json.loads(json_match.group(0).strip())
                # Ensure the result is a dictionary (valid JSON object)
                if isinstance(parsed_json, dict):
                    return parsed_json
            except json.JSONDecodeError:
                pass

        # If no valid JSON object is found, return an empty dictionary
        return {}



    def process_emirates_id_front(self, image_path):
        """
        Process Emirates ID image
        
        Args:
            image_path (str): Path to the Emirates ID image
        
        Returns:
            dict: Extracted Emirates ID information
        """
        prompt = """
    Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
    - Name
    - ID Number
    - Date of Birth
    - Expiry Date
    - Nationality
    Note: Only return these fields in a JSON format.
        *Example Output:*  
        {
        "Name": "John Doe",
        "ID Number": "1234567890123456",
        "Date of Birth": "01/01/1980",
        "Expiry Date": "01/01/2030",
        "Nationality": "USA"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        
        except Exception as e:
            print(f"Error processing Emirates ID: {str(e)}")
            return {}


    def process_emirates_id_back(self, image_path):
        """
        Process Emirates ID image
        
        Args:
            image_path (str): Path to the Emirates ID image
        
        Returns:
            dict: Extracted Emirates ID information
        """
        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
        - Card Number (Card no is of 9 digits and will be found at the top left corner of the card)
        - Occupation
        - Issuing Place

        Note: Only return these fields in a JSON format.
        *Example Output:*
        {
        "Card Number": "123456789
        "Occupation": "Engineer",
        "Issuing Place": "Dubai"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
        """

        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response

        except Exception as e:
            print(f"Error processing Emirates ID: {str(e)}")
            return {}



    def process_driving_licence_front(self, image_path):

        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  

        - Name
        - Date of Birth  
        - Expiry Date  
        - Issue Date
        - License Number
        - Nationality
        - Place of Issue  
        Note: Only return these fields in a JSON format.
        *Example Output:*  
        {
        "Name": "John Doe",
        "Date of Birth": "01/01/1980",
        "Expiry Date": "01/01/2030",
        "Issue Date": "01/01/2020",
        "License Number": "ABC12345",
        "Nationality": "USA",
        "Place of Issue": "California"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        except Exception as e:
            print(f"Error processing Driving Licence: {str(e)}")
            return {}



    def process_driving_licence_back(self, image_path):
        """
        Process Driving Licence image
        
        Args:
            image_path (str): Path to the Driving Licence image
        
        Returns:
            dict: Extracted Driving Licence information
        """
        prompt = """
        Extract the following field from the provided image and return them *only* in JSON format as key-value pairs. 
        *Do not include extra fields, details, comments, or any additional information.* 
        Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  

        - Traffic Code No
        
        Note: Only return above fields in a JSON format.
        
        *Example Output:*
        {
        "Traffic Code No": "567890"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
        """

        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response

        except Exception as e:
            print(f"Error processing Driving Licence: {str(e)}")
            return {}




    def process_vehicle_licence_front(self, image_path):
        """
        Process Vehicle Licence image
        
        Args:
            image_path (str): Path to the Vehicle Licence image
        
        Returns:
            dict: Extracted Vehicle Licence information
        """
        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
        - T. C. No.
        - Owner
        - Nationality
        - Exp. Date
        - Reg. Date
        - Ins. Exp.
    
        Note: Only return these fields in a JSON format.
        *Example Output:*
        {
        "T. C. No.": "123456",
        "Owner": "John Doe",
        "Nationality": "USA",
        "Exp. Date": "01/01/2030",
        "Reg. Date": "01/01/2020",
        "Ins. Exp.": "01/01/2025"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.

        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        except Exception as e:
            print(f"Error processing Vehicle Licence: {str(e)}")
            return {}



    def process_vehicle_licence_back(self, image_path):

        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
        - Model
        - Origin
        - Veh. Type
        - Engine No. (Note: It will be written as NIL most of the time. It will not contain any unit like k.g or cm)
        - Chassis No.
        

        Note: Only return these fields in a JSON format.
        *Example Output:*
        {
        "Model": "Toyota",
        "Origin": "Japan",
        "Veh. Type": "Sedan",
        "Engine No.": "NIL",
        "Chassis No.": "ABC12345"
        }


        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        except Exception as e:
            print(f"Error processing Vehicle Licence: {str(e)}")
            return {}

    def process_pass_certificate(self, image_path):
        """
        Process Pass Certificate image
        
        Args:
            image_path (str): Path to the Pass Certificate image
        
        Returns:
            dict: Extracted Pass Certificate information
        """
        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
        - Inspection Date
        - Pass/Not Pass
        
        Note: Only return these fields in a JSON format.
        *Example Output:*
        {
        "Inspection Date": "01/01/2022",
        "Pass/Not Pass": "Pass"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.
    

        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        except Exception as e:
            print(f"Error processing Pass Certificate: {str(e)}")
            return {}

    def process_trade_certificate(self, image_path):
        """
        Process Trade Certificate image
        
        Args:
            image_path (str): Path to the Trade Certificate image
        
        Returns:
            dict: Extracted Trade Certificate information
        """
        prompt = """
        Extract the following fields from the provided image and return them *only* in JSON format as key-value pairs. *Do not include extra details, comments, or any additional information.* Ensure the output is strictly in plain JSON format without any additional formatting or explanations or markdown:  
        - Trade Name
        - Issue Date
        - Expiry Date
        - Activity
        
        Note: Only return these fields in a JSON format.
        *Example Output:*
        {
        "Trade Name": "ABC Trading",
        "Issue Date": "01/01/2022",
        "Expiry Date": "01/01/2023",
        "Activity": "Retail"
        }

        - It's strictly metioned to not add aany additional data. If something is missing then just return None.

        """
        
        try:
            base64_image = self.encode_image(image_path)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000
            )
            
            full_response = response.choices[0].message.content
            print(f"Model Response /n{full_response}")
            json_response = self._extract_json_from_response(full_response)
            print(f"Extracted JSON /n{json_response}")
            return json_response
        
        except Exception as e:
            print(f"Error processing Trade Certificate: {str(e)}")
            return {}

# Initialize the processor
ocr_processor = DocumentOCRProcessor(api_key="72a0d121ba7634e38a566d277bb41e143b8d75d743697163188f5364740ef258")
# ocr_processor.test_connection()
# Update your existing endpoint methods
def id_front(img_path):
    return ocr_processor.process_emirates_id_front(img_path)

def id_back(img_path):
    return ocr_processor.process_emirates_id_back(img_path)

def driving_front(img_path):
    return ocr_processor.process_driving_licence_front(img_path)

def driving_back(img_path):
    return ocr_processor.process_driving_licence_back(img_path)

def vehicle_front(img_path):
    return ocr_processor.process_vehicle_licence_front(img_path)

def vehicle_back(img_path):
    return ocr_processor.process_vehicle_licence_back(img_path)

def pass_certificate(img_path):
    return ocr_processor.process_pass_certificate(img_path)

def trade_certificate(img_path):
    return ocr_processor.process_trade_certificate(img_path)







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
            confidence = file_name.split('_')[-2]

            # enhanced the image for better results
            # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if 'ID' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = id_front(oriented_file)
                else:
                    detected_info = id_back(oriented_file)
                
            elif 'Driving' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = driving_front(oriented_file)
                else:
                    detected_info = driving_back(oriented_file)
                
            elif 'vehicle' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = vehicle_front(oriented_file)
                else:
                    detected_info = vehicle_back(oriented_file)
             
            elif 'pass' in oriented_file:
                detected_info = pass_certificate(oriented_file)

            elif 'trade' in oriented_file:
                detected_info = trade_certificate(oriented_file)
            else:
                detected_info = {}

            # Compile the result for the current image
            image_result = {
                "image_metadata": {
                    "Image_Path": oriented_file,
                    "Document_Type": doc_type,
                    "Confidence_score": confidence,
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




@app.post("/upload/base64/")
async def upload_base64(request: Base64Request):
    """
    Accepts base64 encoded file data and processes it using the YOLO model.
    
    Args:
        request (Base64Request): Contains base64 encoded data and file extension
            - data: Base64 encoded string of the file
            - extension: File extension (e.g., 'pdf', 'jpg', 'png')
            
    Returns:
        JSONResponse: The response containing the paths to the processed images
    """
    start_time = time.time()
    try:
        # Save the base64 encoded data to a temporary file
        temp_file_path = f"temp_base64.{request.extension}"
        with open(temp_file_path, "wb") as buffer:
            buffer.write(base64.b64decode(request.data))

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
            confidence = file_name.split('_')[-2]

            # enhanced the image for better results
            # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if 'ID' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = id_front(oriented_file)
                else:
                    detected_info = id_back(oriented_file)
                
            elif 'Driving' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = driving_front(oriented_file)
                else:
                    detected_info = driving_back(oriented_file)
                
            elif 'vehicle' in oriented_file:
                if 'front' in oriented_file:
                    detected_info = vehicle_front(oriented_file)
                else:
                    detected_info = vehicle_back(oriented_file)
             
            elif 'pass' in oriented_file:
                detected_info = pass_certificate(oriented_file)

            elif 'trade' in oriented_file:
                detected_info = trade_certificate(oriented_file)
            else:
                detected_info = {}

            # Compile the result for the current image
            image_result = {
                "image_metadata": {
                    "Image_Path": oriented_file,
                    "Document_Type": doc_type,
                    "Confidence_score": confidence,
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
