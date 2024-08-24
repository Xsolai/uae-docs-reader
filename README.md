# emrites-id-reader


# Overview

This project involves the development and deployment of a document detection and text extraction system capable of handling five different types of documents:

- ID Cards 
- Driving Licenses
- Vehicle license
- Pass Certificates
- Commercial Licenses

The system utilizes advanced machine learning techniques to extract relevant information from these documents and provides the extracted data in a structured JSON format.

# Workflow
1. Data Annotation
Tool Used: Roboflow
Description: Annotated the document images by drawing bounding boxes around key information areas such as names, dates, and license numbers. This annotation process enables precise detection and extraction during model training.

3. Model Training
Model: YOLOv8 (You Only Look Once)
Description: Trained the YOLOv8 object detection model on the annotated dataset. The model learns to identify and localize regions of interest within the documents based on the bounding boxes created during annotation.

5. Bounding Box Creation
Description: Utilized the trained YOLOv8 model to create bounding boxes around detected regions of interest in document images. These boxes help in isolating the areas where text extraction needs to be performed.

7. Text Extraction
Tool Used: EasyOCR
Description: Extracted text from the regions identified by the YOLOv8 model. EasyOCR was employed to recognize and extract text from the specified bounding boxes with high accuracy.

9. JSON Conversion
Description: Converted the extracted text data into a structured JSON format. This format organizes the extracted information into a standardized structure, making it easy to handle and integrate into other applications.

11. Deployment
Platform: FastAPI
Description: Deployed the entire document processing pipeline using FastAPI. The API allows users to upload document images and receive the extracted information in JSON format. This deployment facilitates scalable and efficient processing of document images.

API Usage
Endpoint: POST /upload/
Request: Submit an image file of the document to the API.
Response: The API returns a JSON response containing the extracted information, including metadata such as processing time,side of image and tokens used.

# Installation and Setup

Install Dependencies:

```bash*
pip install -r requirements.txt
```

Run the Application:
```bash*
uvicorn app:app --reload
```

# Directory Structure
/models: Contains the YOLOv8 models.

/app: Contains the FastAPI application code.

/images: Contains input and output images.
