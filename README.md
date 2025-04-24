# PDF Converter API

A FastAPI backend for PDF conversion operations.

## Features

- Merge multiple PDF files
- Split PDF files by page ranges
- Download processed PDF files

## Setup

1. Create a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python run.py
```

The API will be available at http://localhost:8000

## API Documentation

Once the server is running, you can access the API documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Endpoints

- `GET /`: Check if API is running
- `POST /api/merge-pdf`: Merge multiple PDF files
- `POST /api/split-pdf`: Split a PDF file
- `GET /api/download/{session_id}/{filename}`: Download a processed PDF file 