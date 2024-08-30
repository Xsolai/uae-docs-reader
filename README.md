# emrites-id-reader

## Overview

This project involves the development and deployment of a document detection and text extraction system capable of handling four different types of documents:

* ID Cards
* Driving Licenses
* Vehicle Licenses
* Certificates

The system utilizes advanced deep learning techniques to extract relevant information from these documents and provides the extracted data in a structured JSON format. The system now supports both image and PDF inputs, processing them to detect and extract data from multiple documents within a single file.

## Workflow

Start
  |
  v
User uploads an image or PDF via the FastAPI endpoint
  |
  v
Document Detection
  |
  v
Cropping detected documents (saved in cropped_images folder)
  |
  v
Orientation Adjustment (saved in oriented_images folder)
  |
  v
Text Extraction using EasyOCR
  |
  v
JSON Conversion of extracted data
  |
  v
Response sent back to the user with extracted data and metadata
  |
  v
End


1. **Data Annotation**
   * **Tool Used** : Roboflow
   * **Description** : Annotated the document images by drawing bounding boxes around key information areas such as names, dates, and license numbers. This annotation process enables precise detection and extraction during model training.
2. **Model Training**
   * **Model** : YOLOv8 (You Only Look Once)
   * **Description** : Trained the YOLOv8 object detection model on the annotated dataset. The model learns to identify and localize regions of interest within the documents based on the bounding boxes created during annotation.
3. **Document Detection and Cropping**
   * **Description** : The system detects documents within the input image or PDF file, crops the detected regions, and saves them. Each detected document is saved as a separate image, ensuring that further processing is focused only on the relevant document areas.
4. **Orientation Adjustment**
   * **Description** : Automatically adjusts the orientation of the cropped document images to ensure they are correctly aligned for text extraction. This step is crucial for improving the accuracy of text recognition.
5. **Text Extraction**
   * **Tool Used** : EasyOCR
   * **Description** : Extracts text from the oriented and cropped document images. EasyOCR is employed to recognize and extract text from the specified regions with high accuracy.
6. **JSON Conversion**
   * **Description** : Converts the extracted text data into a structured JSON format. This format organizes the extracted information into a standardized structure, making it easy to handle and integrate into other applications.
7. **Deployment**
   * **Platform** : FastAPI
   * **Description** : Deployed the entire document processing pipeline using FastAPI. The API allows users to upload document images or PDFs and receive the extracted information in JSON format. This deployment facilitates scalable and efficient processing of document images.

## API Usage

* **Endpoint** : `POST /upload/`
* **Request** : Submit an image or PDF file of the document to the API.
* **Response** : The API returns a JSON response containing the extracted information for each detected document, including metadata such as processing time, side of the image, and tokens used.

## Installation and Setup

1. **Install Dependencies** :

```besh
pip install -r requirements.txt
```

1. **Run the Application** :

```besh
uvicorn app:app --reload
```

## Directory Structure

* **/models** : Contains the YOLOv8 models.
* **/app** : Contains the FastAPI application code.
* **/test** : Contains testing images and pdf.
* **/cropped_images** : Stores the cropped images of detected documents.
* **/oriented_images** : Stores the oriented images ready for text extraction.

## Workflow Example

1. **Image or PDF Upload** : The user uploads an image or PDF containing multiple documents.
2. **Document Detection** : The system detects each document within the file.
3. **Cropping** : Detected documents are cropped from the original file and saved.
4. **Orientation Adjustment** : Each cropped document is adjusted to the correct orientation.
5. **Text Extraction** : Text is extracted from each oriented document.
6. **JSON Output** : The extracted text is returned in a structured JSON format, with metadata for each documen
