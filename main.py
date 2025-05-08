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
        
        # Convert Word to PDF using python-docx2pdf
        output_path = result_dir / "converted.pdf"
        
        try:
            from docx2pdf import convert
            convert(str(file_path), str(output_path))
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Word to PDF conversion is not available. Please ensure docx2pdf is installed."
            )
        
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
            import subprocess
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 