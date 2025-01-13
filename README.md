# Emirates ID Reader API

A FastAPI application that extracts information from various UAE documents including Emirates ID, Driving License, Vehicle License, Trade Certificates, and Pass Certificates using computer vision and OCR.

## Features

* Document classification and orientation detection using YOLO
* Support for multiple document types:
  * Emirates ID (front and back)
  * Driving License (front and back)
  * Vehicle License (front and back)
  * Pass Certificates
  * Trade Certificates
* PDF support with high-quality image extraction
* Automatic document cropping and orientation correction
* JSON response with detailed metadata
* Built-in token usage tracking
* Support for both image and PDF uploads

## Requirements

Create a virtual environment and install the required packages:

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with the following:

```
TOGETHER_API_KEY=your_together_ai_api_key_here
```

## Usage

1. Start the server:

```
uvicorn app:app --reload
```

2. The API will be available at `http://localhost:8000/docs`
3. Use the `/upload/` endpoint to process documents:

## Model Information

* Document Classification: YOLO model trained for UAE document classification
* OCR: Together.ai's Vision model for text extraction
* Supported Image Formats: JPG, PNG, PDF
