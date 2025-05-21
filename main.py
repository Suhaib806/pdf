from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import os
import shutil
from uuid import uuid4
from typing import List
import pypdf
import time
import logging
from pathlib import Path
import fitz  # PyMuPDF
import io
import tempfile
import zipfile
from PIL import Image, ImageDraw, ImageFont
import subprocess
import asyncio  # Added asyncio import
from PyPDF2 import PdfReader, PdfWriter

# Try to import PIL but don't fail if it's not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: PIL/Pillow not available. Some image processing features may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pdf_converter")

app = FastAPI(title="PDF Converter API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.up.railway.app",  # Railway public domains
        "https://pdf.railway.internal",  # Railway internal domain
        "https://pdfgadgets.webfalcons.pk",
        "https://pdf-production-ef1b.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total
ALLOWED_EXTENSIONS = ['.pdf']

# Get the current working directory
BASE_DIR = Path.cwd()

# Define upload and result directories relative to the base directory
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

# Create necessary directories
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

@app.get("/")
async def read_root():
    return {"message": "PDF Converter API is running"}

async def validate_pdf_file(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate PDF file size and type."""
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type not allowed. Only {', '.join(ALLOWED_EXTENSIONS)} files are supported.")
    
    # Validate file size
    file_size = 0
    content = await file.read(1024 * 1024)  # Read 1MB chunks
    while content:
        file_size += len(content)
        if file_size > max_size:
            await file.seek(0)  # Reset file pointer
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {max_size / (1024 * 1024)}MB.")
        content = await file.read(1024 * 1024)
    
    await file.seek(0)  # Reset file pointer for subsequent operations

@app.post("/api/merge-pdf")
async def merge_pdf(files: List[UploadFile] = File(...)):
    start_time = time.time()
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed for merging")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_paths = []
    total_size = 0
    file_details = []
    
    try:
        for file in files:
            # Validate file
            await validate_pdf_file(file)
            
            # Get file size
            file.file.seek(0, os.SEEK_END)
            file_size = file.file.tell()
            file.file.seek(0)
            
            # Check total size
            total_size += file_size
            if total_size > MAX_TOTAL_SIZE:
                raise HTTPException(status_code=400, 
                                detail=f"Total file size exceeds maximum limit of {MAX_TOTAL_SIZE / (1024 * 1024)}MB.")
            
            # Save file
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Verify the file is a valid PDF
            try:
                with open(file_path, "rb") as f:
                    # Check for PDF header
                    header = f.read(5)
                    if header != b"%PDF-":
                        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a valid PDF")
                    
                    # Try to read the PDF
                    f.seek(0)
                    pdf = pypdf.PdfReader(f)
                    if len(pdf.pages) == 0:
                        raise HTTPException(status_code=400, detail=f"File {file.filename} contains no pages")
            except Exception as e:
                # Clean up the invalid file
                try:
                    os.unlink(file_path)
                except:
                    pass
                raise HTTPException(status_code=400, detail=f"Invalid PDF file: {file.filename}")
            
            file_paths.append(str(file_path))
            file_details.append({
                "name": file.filename,
                "size": file_size,
                "path": str(file_path)
            })
        
        logger.info(f"Merging {len(files)} PDFs, total size: {total_size / (1024 * 1024):.2f}MB")
        
        # Merge PDFs
        merger = pypdf.PdfMerger()
        for pdf in file_paths:
            merger.append(pdf)
        
        output_path = result_dir / "merged.pdf"
        merger.write(str(output_path))
        merger.close()
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Merge completed in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
        
        return {
            "message": "PDFs merged successfully",
            "file_path": str(output_path), 
            "session_id": session_id,
            "file_count": len(files),
            "total_input_size": total_size,
            "output_size": output_size,
            "processing_time": elapsed_time
        }
        
    except HTTPException:
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        raise
        
    except Exception as e:
        logger.error(f"Error merging PDFs: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error merging PDFs: {str(e)}")

@app.post("/api/split-pdf")
async def split_pdf(file: UploadFile = File(...), page_ranges: str = Form(None)):
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Validate and save uploaded file
        await validate_pdf_file(file)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Split PDF
        pdf = pypdf.PdfReader(str(file_path))
        total_pages = len(pdf.pages)
        
        # Check if PDF is encrypted
        if pdf.is_encrypted:
            raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported")
        
        result_files = []
        
        if page_ranges:
            # Custom page ranges (format: "1-3,5-7")
            ranges = page_ranges.split(',')
            for i, page_range in enumerate(ranges):
                try:
                    output = pypdf.PdfWriter()
                    start, end = map(int, page_range.split('-'))
                    
                    # Adjust for 0-based indexing
                    start = max(1, start) - 1
                    end = min(total_pages, end) - 1
                    
                    # Copy metadata from original PDF
                    if pdf.metadata:
                        output.add_metadata(pdf.metadata)
                    
                    for p in range(start, end + 1):
                        output.add_page(pdf.pages[p])
                    
                    output_path = result_dir / f"split_{i+1}.pdf"
                    with open(output_path, "wb") as output_file:
                        output.write(output_file)
                    
                    result_files.append(str(output_path))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid page range: {page_range}")
        else:
            # Split into individual pages
            for i in range(total_pages):
                output = pypdf.PdfWriter()
                
                # Copy metadata from original PDF
                if pdf.metadata:
                    output.add_metadata(pdf.metadata)
                
                output.add_page(pdf.pages[i])
                
                output_path = result_dir / f"page_{i+1}.pdf"
                with open(output_path, "wb") as output_file:
                    output.write(output_file)
                
                result_files.append(str(output_path))
        
        elapsed_time = time.time() - start_time
        logger.info(f"Split completed in {elapsed_time:.2f}s, created {len(result_files)} files")
        
        return {
            "message": "PDF split successfully", 
            "total_pages": total_pages,
            "result_files": result_files,
            "session_id": session_id,
            "processing_time": elapsed_time
        }
    
    except HTTPException:
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        raise
        
    except Exception as e:
        logger.error(f"Error splitting PDF: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error splitting PDF: {str(e)}")

@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    file_path = RESULT_DIR / session_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine the appropriate media type based on file extension
    media_type = "application/pdf"  # Default
    
    if filename.lower().endswith(".docx"):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif filename.lower().endswith(".doc"):
        media_type = "application/msword"
    elif filename.lower().endswith(".pdf"):
        media_type = "application/pdf"
    
    logger.info(f"Serving file {filename} with media type {media_type}")
    
    return FileResponse(path=str(file_path), filename=filename, media_type=media_type)

@app.post("/api/word-to-pdf")
async def convert_word_to_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.filename.lower().endswith(('.doc', '.docx')):
        raise HTTPException(status_code=400, detail="Only Word files (.doc, .docx) are allowed")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Validate and save uploaded file
        await validate_word_file(file)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Convert Word to PDF using LibreOffice
        output_path = result_dir / "converted.pdf"
        
        try:
            import platform
            
            # Determine the LibreOffice command based on the platform
            if platform.system() == "Windows":
                # For Windows, try to find LibreOffice in common installation paths
                possible_paths = [
                    r"C:\Program Files\LibreOffice\program\soffice.exe",
                    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                    r"C:\Program Files\LibreOffice\App\libreoffice\program\soffice.exe",
                    r"C:\Program Files (x86)\LibreOffice\App\libreoffice\program\soffice.exe"
                ]
                soffice_cmd = None
                for path in possible_paths:
                    if os.path.exists(path):
                        soffice_cmd = path
                        break
                if not soffice_cmd:
                    raise Exception(
                        "LibreOffice not found. Please install LibreOffice from https://www.libreoffice.org/download/download/ "
                        "and make sure it's installed in one of the standard locations."
                    )
            else:
                soffice_cmd = "soffice"
            
            # Run LibreOffice conversion
            cmd = [
                soffice_cmd,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(result_dir),
                str(file_path)
            ]
            
            # Run the conversion command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Rename the output file to match our expected name
            converted_file = result_dir / f"{file_path.stem}.pdf"
            if converted_file.exists():
                converted_file.rename(output_path)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Conversion failed: {e.stderr}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except FileNotFoundError as e:
            error_msg = (
                "LibreOffice not found. Please install LibreOffice from https://www.libreoffice.org/download/download/ "
                "and make sure it's installed in one of the standard locations."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Word to PDF conversion completed in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
        
        return {
            "message": "Word file converted to PDF successfully",
            "file_path": str(output_path),
            "session_id": session_id,
            "output_size": output_size,
            "processing_time": elapsed_time
        }
        
    except HTTPException:
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        raise
        
    except Exception as e:
        logger.error(f"Error converting Word to PDF: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error converting Word to PDF: {str(e)}")

async def validate_word_file(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate Word file size and type."""
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.doc', '.docx']:
        raise HTTPException(status_code=400, detail="File type not allowed. Only .doc and .docx files are supported.")
    
    # Validate file size
    file_size = 0
    content = await file.read(1024 * 1024)  # Read 1MB chunks
    while content:
        file_size += len(content)
        if file_size > max_size:
            await file.seek(0)  # Reset file pointer
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {max_size / (1024 * 1024)}MB.")
        content = await file.read(1024 * 1024)
    
    await file.seek(0)  # Reset file pointer for subsequent operations

@app.post("/api/image-to-pdf")
async def convert_images_to_pdf(
    files: List[UploadFile] = File(...),
    quality: str = Form("high"),
    pageSize: str = Form("a4"),
    orientation: str = Form("portrait")
):
    if not HAS_PIL:
        raise HTTPException(status_code=500, detail="PIL/Pillow is not available")
    
    session_id = str(uuid4())
    session_dir = Path("uploads") / session_id
    result_dir = Path("results") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded images
        image_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            
            file_path = session_dir / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(file_path)
        
        # Convert images to PDF
        output_filename = f"converted_{session_id}.pdf"
        output_path = result_dir / output_filename
        
        # Create PDF from images
        images = []
        for image_path in image_paths:
            img = Image.open(image_path)
            if orientation == "landscape" and img.width < img.height:
                img = img.rotate(90, expand=True)
            elif orientation == "portrait" and img.width > img.height:
                img = img.rotate(-90, expand=True)
            images.append(img)
        
        # Save first image as PDF
        if images:
            # Convert to RGB if necessary
            if images[0].mode in ('RGBA', 'LA'):
                background = Image.new('RGB', images[0].size, (255, 255, 255))
                background.paste(images[0], mask=images[0].split()[-1])
                images[0] = background
            
            # Set quality based on user selection
            quality_map = {
                "high": 100,
                "medium": 75,
                "low": 50
            }
            quality_value = quality_map.get(quality, 100)
            
            # Save as PDF
            if len(images) == 1:
                # If there's only one image, don't use append_images
                images[0].save(
                    output_path,
                    "PDF",
                    resolution=quality_value
                )
            else:
                # If there are multiple images, use append_images
                images[0].save(
                    output_path,
                    "PDF",
                    resolution=quality_value,
                    save_all=True,
                    append_images=images[1:]
                )
        
        return {
            "session_id": session_id,
            "filename": output_filename
        }
        
    except Exception as e:
        logger.error(f"Error converting images to PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded files
        for file in files:
            file.file.close()

@app.post("/api/pdf-to-jpg")
async def convert_pdf_to_jpg(
    file: UploadFile = File(...),
    quality: str = Form("high"),
    dpi: int = Form(300)
):
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Validate and save uploaded file
        await validate_pdf_file(file)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Convert PDF to images
        pdf_document = fitz.open(str(file_path))
        total_pages = len(pdf_document)
        
        if total_pages == 0:
            raise HTTPException(status_code=400, detail="PDF file contains no pages")
        
        # Handle single page PDF
        if total_pages == 1:
            page = pdf_document[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert to PIL Image for quality control
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save image with quality settings
            img_quality = 95 if quality == "high" else 75
            img_path = result_dir / "converted.jpg"
            img.save(img_path, "JPEG", quality=img_quality)
            
            # Get output file size
            output_size = os.path.getsize(img_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"PDF to JPG conversion completed in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
            
            return {
                "message": "PDF converted to JPG successfully",
                "filename": "converted.jpg",
                "session_id": session_id,
                "total_pages": total_pages,
                "output_size": output_size,
                "processing_time": elapsed_time
            }
        
        # Handle multi-page PDF
        # Create a zip file to store all images
        zip_path = result_dir / "converted_images.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for page_num in range(total_pages):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                
                # Convert to PIL Image for quality control
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save image with quality settings
                img_quality = 95 if quality == "high" else 75
                img_path = result_dir / f"page_{page_num + 1}.jpg"
                img.save(img_path, "JPEG", quality=img_quality)
                
                # Add to zip
                zipf.write(img_path, f"page_{page_num + 1}.jpg")
                
                # Clean up individual image file
                os.remove(img_path)
        
        pdf_document.close()
        
        # Get output file size
        output_size = os.path.getsize(zip_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"PDF to JPG conversion completed in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
        
        return {
            "message": "PDF converted to JPG successfully",
            "filename": "converted_images.zip",
            "session_id": session_id,
            "total_pages": total_pages,
            "output_size": output_size,
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error converting PDF to JPG: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error converting PDF to JPG: {str(e)}")

@app.post("/api/powerpoint-to-pdf")
async def convert_powerpoint_to_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.filename.lower().endswith(('.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Only PowerPoint files (.ppt, .pptx) are allowed")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    output_filename = f"{Path(file.filename).stem}.pdf"
    output_path = result_dir / output_filename
    
    try:
        # Validate and save uploaded file
        await validate_powerpoint_file(file)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            import platform
            
            # Determine the LibreOffice command based on the platform
            if platform.system() == "Windows":
                # For Windows, try to find LibreOffice in common installation paths
                possible_paths = [
                    r"C:\Program Files\LibreOffice\program\soffice.exe",
                    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                    r"C:\Program Files\LibreOffice\App\libreoffice\program\soffice.exe",
                    r"C:\Program Files (x86)\LibreOffice\App\libreoffice\program\soffice.exe"
                ]
                soffice_cmd = None
                for path in possible_paths:
                    if os.path.exists(path):
                        soffice_cmd = path
                        break
                if not soffice_cmd:
                    raise Exception(
                        "LibreOffice not found. Please install LibreOffice from https://www.libreoffice.org/download/download/ "
                        "and make sure it's installed in one of the standard locations."
                    )
            else:
                soffice_cmd = "soffice"
            
            # Run LibreOffice conversion
            cmd = [
                soffice_cmd,
                "--headless",
                "--convert-to", "pdf",
                "--outdir", str(result_dir),
                str(file_path)
            ]
            
            # Run the conversion command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Get output file size
            output_size = os.path.getsize(output_path)
            
            elapsed_time = time.time() - start_time
            logger.info(f"PowerPoint to PDF conversion completed in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
            
            return {
                "message": "PowerPoint converted to PDF successfully",
                "session_id": session_id,
                "filename": output_filename,
                "processing_time": elapsed_time,
                "output_size": output_size
            }
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Conversion failed: {e.stderr}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except FileNotFoundError as e:
            error_msg = (
                "LibreOffice not found. Please install LibreOffice from https://www.libreoffice.org/download/download/ "
                "and make sure it's installed in one of the standard locations."
            )
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        
    except Exception as e:
        logger.error(f"Error converting PowerPoint to PDF: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error converting PowerPoint to PDF: {str(e)}")

async def validate_powerpoint_file(file: UploadFile, max_size: int = MAX_FILE_SIZE) -> None:
    """Validate PowerPoint file size and type."""
    # Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.ppt', '.pptx']:
        raise HTTPException(status_code=400, detail="File type not allowed. Only .ppt and .pptx files are supported.")
    
    # Validate file size
    file_size = 0
    content = await file.read(1024 * 1024)  # Read 1MB chunks
    while content:
        file_size += len(content)
        if file_size > max_size:
            await file.seek(0)  # Reset file pointer
            raise HTTPException(status_code=400, detail=f"File size exceeds maximum limit of {max_size / (1024 * 1024)}MB.")
        content = await file.read(1024 * 1024)
    
    await file.seek(0)  # Reset file pointer for subsequent operations

@app.post("/api/add-watermark")
async def add_watermark(
    file: UploadFile = File(...),
    watermark_image: UploadFile = File(...),
    opacity: float = Form(0.5),
    position: str = Form("center"),
    scale: float = Form(1.0)
):
    start_time = time.time()
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    try:
        # Validate PDF file
        await validate_pdf_file(file)
        
        # Save PDF file
        pdf_path = upload_dir / file.filename
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Save watermark image
        watermark_path = upload_dir / watermark_image.filename
        with open(watermark_path, "wb") as buffer:
            shutil.copyfileobj(watermark_image.file, buffer)
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Open and process the watermark image
        watermark_img = Image.open(watermark_path)
        
        # Convert watermark to RGBA if it's not already
        if watermark_img.mode != 'RGBA':
            watermark_img = watermark_img.convert('RGBA')
        
        # Apply opacity
        watermark_data = watermark_img.getdata()
        new_data = []
        for item in watermark_data:
            # Adjust alpha channel based on opacity
            new_data.append((item[0], item[1], item[2], int(item[3] * opacity)))
        watermark_img.putdata(new_data)
        
        # Process each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get page dimensions
            page_width = page.rect.width
            page_height = page.rect.height
            
            # Calculate watermark dimensions
            wm_width = int(watermark_img.width * scale)
            wm_height = int(watermark_img.height * scale)
            
            # Calculate position
            if position == "center":
                x = (page_width - wm_width) / 2
                y = (page_height - wm_height) / 2
            elif position == "top-left":
                x = 50
                y = 50
            elif position == "top-right":
                x = page_width - wm_width - 50
                y = 50
            elif position == "bottom-left":
                x = 50
                y = page_height - wm_height - 50
            elif position == "bottom-right":
                x = page_width - wm_width - 50
                y = page_height - wm_height - 50
            else:  # default to center
                x = (page_width - wm_width) / 2
                y = (page_height - wm_height) / 2
            
            # Convert watermark to bytes
            watermark_bytes = io.BytesIO()
            watermark_img.save(watermark_bytes, format='PNG')
            watermark_bytes.seek(0)
            
            # Insert watermark
            page.insert_image(fitz.Rect(x, y, x + wm_width, y + wm_height), stream=watermark_bytes.getvalue())
        
        # Save the result
        output_path = result_dir / "watermarked.pdf"
        pdf_document.save(str(output_path))
        pdf_document.close()
        
        # Get output file size
        output_size = os.path.getsize(output_path)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Watermark added in {elapsed_time:.2f}s, output size: {output_size / (1024 * 1024):.2f}MB")
        
        return {
            "message": "Watermark added successfully",
            "file_path": str(output_path),
            "session_id": session_id,
            "output_size": output_size,
            "processing_time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Error adding watermark: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")

@app.post("/api/compress-pdf")
async def compress_pdf(
    file: UploadFile = File(...),
    quality: str = Form("medium")
):
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Create a unique folder for this request
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Validate and save uploaded file
        await validate_pdf_file(file)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Enhanced compression quality settings with better ratios
        quality_map = {
            "low": {
                "image_quality": 15,
                "image_dpi": 72,
                "downsample_threshold": 150,
                "compress_images": True,
                "compress_fonts": True,
                "cleanup": True
            },
            "medium": {
                "image_quality": 30,
                "image_dpi": 100,
                "downsample_threshold": 200,
                "compress_images": True,
                "compress_fonts": True,
                "cleanup": True
            },
            "high": {
                "image_quality": 60,
                "image_dpi": 150,
                "downsample_threshold": 300,
                "compress_images": True,
                "compress_fonts": True,
                "cleanup": True
            }
        }
        compression_settings = quality_map.get(quality, quality_map["medium"])
        
        # Create output PDF with a different name
        output_filename = f"compressed_{session_id}.pdf"
        output_path = result_dir / output_filename
        
        # Open the PDF with PyMuPDF
        doc = fitz.open(file_path)
        
        # Process each page
        for page in doc:
            # Get all images on the page
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Open image with PIL
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    
                    # Get image DPI if possible
                    img_dpi = 300  # Default assumption
                    if hasattr(img_pil, "info") and "dpi" in img_pil.info:
                        img_dpi = img_pil.info["dpi"][0]
                    
                    # Skip if image is already small
                    if len(image_bytes) < 10 * 1024:  # Skip images smaller than 10KB
                        continue
                    
                    # Convert to RGB if necessary
                    if img_pil.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img_pil.size, (255, 255, 255))
                        background.paste(img_pil, mask=img_pil.split()[-1])
                        img_pil = background
                    elif img_pil.mode != 'RGB' and img_pil.mode != 'L':
                        img_pil = img_pil.convert('RGB')
                    
                    # Calculate new dimensions based on DPI target
                    if img_dpi > compression_settings["downsample_threshold"]:
                        scale_factor = compression_settings["image_dpi"] / img_dpi
                        if scale_factor < 0.9:  # Only resize if reduction is meaningful
                            new_width = max(1, int(img_pil.width * scale_factor))
                            new_height = max(1, int(img_pil.height * scale_factor))
                            img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save with compression
                    output_buffer = io.BytesIO()
                    
                    # Choose optimal format based on image type
                    if img_pil.mode == 'L':  # Grayscale image
                        img_pil.save(
                            output_buffer,
                            format='PNG',
                            optimize=True
                        )
                    else:
                        # Use better compression for photographic images
                        img_pil.save(
                            output_buffer,
                            format='JPEG',
                            quality=compression_settings["image_quality"],
                            optimize=True,
                            progressive=True
                        )
                    
                    # Only replace if actually smaller
                    new_image_bytes = output_buffer.getvalue()
                    if len(new_image_bytes) < len(image_bytes):
                        # Replace the image in the PDF
                        doc.update_stream(xref, new_image_bytes)
                        logger.debug(f"Image compressed: {len(image_bytes)/1024:.1f}KB → {len(new_image_bytes)/1024:.1f}KB")
                    
                except Exception as e:
                    logger.warning(f"Failed to compress image on page {page.number}, index {img_index}: {str(e)}")
                    continue
        
        # Apply additional optimizations
        if compression_settings["cleanup"]:
            # Use built-in optimization options
            pass
        
        # Save with maximum compression
        doc.save(
            output_path,
            garbage=4,  # Maximum garbage collection
            deflate=True,  # Use deflate compression
            clean=True,  # Clean redundant elements
            pretty=False,  # Don't pretty print (saves space)
            linear=True  # Optimize for web viewing
        )
        
        # Close the document
        doc.close()
        
        # Get output file size
        original_size = os.path.getsize(file_path)
        output_size = os.path.getsize(output_path)
        compression_ratio = original_size / output_size if output_size > 0 else 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"PDF compression completed in {elapsed_time:.2f}s, original: {original_size / (1024 * 1024):.2f}MB, "
                    f"output: {output_size / (1024 * 1024):.2f}MB, ratio: {compression_ratio:.2f}x")
        
        # Return response with more detailed information
        response = {
            "message": "PDF compressed successfully",
            "file_path": output_filename,
            "session_id": session_id,
            "original_size": original_size,
            "output_size": output_size,
            "compression_ratio": compression_ratio,
            "processing_time": elapsed_time
        }
        logger.info(f"Compression response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error compressing PDF: {str(e)}", exc_info=True)
        
        # Clean up
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error compressing PDF: {str(e)}")

@app.post("/api/unlock-pdf")
async def unlock_pdf(file: UploadFile = File(...)):
    start_time = time.time()
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    session_id = str(uuid4())
    upload_dir = UPLOAD_DIR / session_id
    result_dir = RESULT_DIR / session_id
    upload_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    try:
        await validate_pdf_file(file)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Try to unlock with common passwords and empty password
        common_passwords = ["", "1234", "12345", "password", "owner", "user", "secret", "admin", "userpass", "ownerpass"]
        unlocked = False
        output_path = None

        # First try with PyMuPDF as it's more powerful
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            
            if doc.is_encrypted:
                # Try to authenticate with empty password first
                if doc.authenticate(""):
                    # Create a new document without encryption
                    new_doc = fitz.open()
                    new_doc.insert_pdf(doc)
                    
                    output_filename = f"unlocked_{session_id}.pdf"
                    output_path = result_dir / output_filename
                    
                    # Save without encryption
                    new_doc.save(
                        output_path,
                        garbage=4,
                        deflate=True,
                        clean=True,
                        pretty=False,
                        linear=True
                    )
                    
                    # Close documents
                    new_doc.close()
                    doc.close()
                    
                    # Verify the new PDF is readable
                    test_doc = fitz.open(output_path)
                    if not test_doc.is_encrypted:
                        unlocked = True
                    test_doc.close()
                else:
                    # Try common passwords
                    for pwd in common_passwords:
                        if doc.authenticate(pwd):
                            # Create a new document without encryption
                            new_doc = fitz.open()
                            new_doc.insert_pdf(doc)
                            
                            output_filename = f"unlocked_{session_id}.pdf"
                            output_path = result_dir / output_filename
                            
                            # Save without encryption
                            new_doc.save(
                                output_path,
                                garbage=4,
                                deflate=True,
                                clean=True,
                                pretty=False,
                                linear=True
                            )
                            
                            # Close documents
                            new_doc.close()
                            doc.close()
                            
                            # Verify the new PDF is readable
                            test_doc = fitz.open(output_path)
                            if not test_doc.is_encrypted:
                                unlocked = True
                            test_doc.close()
                            break
            else:
                # PDF is not encrypted, just copy it
                output_filename = f"unlocked_{session_id}.pdf"
                output_path = result_dir / output_filename
                doc.save(output_path)
                doc.close()
                unlocked = True

        except Exception as e:
            logger.warning(f"Failed to unlock with PyMuPDF: {str(e)}")
            # If PyMuPDF fails, try PyPDF2 as fallback
            try:
                with open(file_path, "rb") as f:
                    reader = PdfReader(f)
                    if reader.is_encrypted:
                        # Try common passwords
                        for pwd in common_passwords:
                            if reader.decrypt(pwd) == 1:
                                pdf_writer = PdfWriter()
                                for page in reader.pages:
                                    pdf_writer.add_page(page)
                                
                                output_filename = f"unlocked_{session_id}.pdf"
                                output_path = result_dir / output_filename
                                with open(output_path, "wb") as output_file:
                                    pdf_writer.write(output_file)
                                unlocked = True
                                break
                    else:
                        # PDF is not encrypted, just copy it
                        output_filename = f"unlocked_{session_id}.pdf"
                        output_path = result_dir / output_filename
                        with open(output_path, "wb") as output_file:
                            reader.write(output_file)
                        unlocked = True
            except Exception as e:
                logger.warning(f"Failed to unlock with PyPDF2: {str(e)}")

        if not unlocked:
            # Clean up before raising the exception
            try:
                if upload_dir.exists():
                    for file in upload_dir.glob("*"):
                        try:
                            file.unlink()
                        except:
                            pass
                    upload_dir.rmdir()
                if result_dir.exists():
                    for file in result_dir.glob("*"):
                        try:
                            file.unlink()
                        except:
                            pass
                    result_dir.rmdir()
            except:
                pass
            raise HTTPException(
                status_code=400,
                detail="Unable to remove password protection from this PDF. The file may be using advanced encryption."
            )

        original_size = os.path.getsize(file_path)
        output_size = os.path.getsize(output_path)
        elapsed_time = time.time() - start_time
        return {
            "message": "PDF password removed successfully",
            "file_path": output_path.name,
            "session_id": session_id,
            "original_size": original_size,
            "output_size": output_size,
            "processing_time": elapsed_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing PDF password: {str(e)}", exc_info=True)
        try:
            if upload_dir.exists():
                for file in upload_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                upload_dir.rmdir()
            if result_dir.exists():
                for file in result_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                result_dir.rmdir()
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error removing PDF password: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 