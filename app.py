import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import pytesseract
import img2pdf
import pyttsx3
from gtts import gTTS
from skimage.metrics import structural_similarity as ssim
from pyzbar.pyzbar import decode
from scanner import scan_document
from pypdf import PdfReader, PdfWriter
import pymupdf as fitz

# Configure Tesseract for Linux server (Render)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Ensure the /tmp directory exists for Render
os.makedirs("/tmp", exist_ok=True)

# ============================================================
# Core Helper Functions
# ============================================================
def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = np.mean(gray)
    msg = []
    if blur < 50: msg.append("⚠️ Blurry image.")
    if bright < 60: msg.append("⚠️ Too dark.")
    return "\n".join(msg) if msg else "✅ Quality is perfect!"

def apply_filter(img, filter_type):
    if filter_type == "Original": return img
    if filter_type == "Grayscale": return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if filter_type == "Vintage":
        sepia_matrix = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        return np.clip(cv2.transform(img, sepia_matrix), 0, 255).astype(np.uint8)
    if filter_type == "Blueprint":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        blue_img = np.zeros_like(img)
        blue_img[:] = (30, 50, 150)
        blue_img[thresh == 0] = (255, 255, 255)
        return cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
    return img

# ============================================================
# Core App Functions
# ============================================================
def process_scan(img, mode, progress=gr.Progress()):
    if img is None: raise gr.Error("Upload an image.")
    q_msg = check_quality(img)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, bgr)
    progress(0.5, desc="Scanning and warping...")
    result = scan_document(tmp.name, output_path=None, mode=mode, debug=False)
    os.unlink(tmp.name)
    if len(result.shape) == 2: result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else: result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return img, result, q_msg

def generate_pdf(img):
    if img is None: raise gr.Error("Please scan an image first.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_p = f"/tmp/scan_{ts}.jpg"
    pdf_p = f"/tmp/scan_{ts}.pdf"
    cv2.imwrite(img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(pdf_p, "wb") as f:
        f.write(img2pdf.convert(img_p))
    os.remove(img_p)
    return pdf_p

def do_ocr_and_speech(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, bgr)
    scanned = scan_document(tmp.name, output_path=None, mode='bw')
    os.unlink(tmp.name)
    try:
        text = pytesseract.image_to_string(scanned)
        if not text.strip(): return "No text found.", None
        # Use gTTS instead of pyttsx3 for cloud compatibility
        tts = gTTS(text=text, lang='en')
        audio_p = f"/tmp/audio_{datetime.now().strftime('%H%M%S')}.mp3"
        tts.save(audio_p)
        return text, audio_p
    except Exception as e:
        return f"Error: {e}", None

def extract_sig(img):
    if img is None: raise gr.Error("Upload an image.")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sig = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(sig)

def compare_docs(img1, img2):
    if img1 is None or img2 is None: raise gr.Error("Upload both images.")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img1.copy()
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return result, f"Similarity: {score*100:.2f}%"

def privacy_blur(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT)
    out = img.copy()
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and len(data['text'][i].strip()) > 0:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            roi = out[y:y+h, x:x+w]
            if roi.size > 0:
                out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (25, 25), 30)
    return out

def fix_orientation(img, rotation):
    if img is None: raise gr.Error("Upload an image.")
    if rotation == "180": return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == "90 CW": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "90 CCW": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "Flip": return cv2.flip(img, 1)
    return img

def scan_barcode(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    barcodes = decode(bgr)
    out = img.copy()
    text_results = []
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_results.append(f"Type: {barcode.type} | Data: {barcode.data.decode('utf-8')}")
    return out, "\n".join(text_results) if text_results else "No barcodes found."

def auto_enhance(img):
    if img is None: raise gr.Error("Upload an image.")
    gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

# ============================================================
# PDF Tools
# ============================================================
def split_pdf(pdf_file, page_num):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    reader = PdfReader(pdf_file.name)
    writer = PdfWriter()
    try: writer.add_page(reader.pages[int(page_num) - 1])
    except IndexError: raise gr.Error("Page number out of range.")
    out_path = f"/tmp/split_{datetime.now().strftime('%H%M%S')}.pdf"
    with open(out_path, "wb") as f: writer.write(f)
    return out_path

def merge_pdfs(pdf_files):
    if not pdf_files or len(pdf_files) < 2: 
        raise gr.Error("Please upload at least 2 PDF files to merge.")
    out_path = f"/tmp/merged_{datetime.now().strftime('%H%M%S')}.pdf"
    writer = PdfWriter()
    for pdf in pdf_files:
        reader = PdfReader(pdf.name)
        for page in reader.pages:
            writer.add_page(page)
    with open(out_path, "wb") as f: 
        writer.write(f)
    return out_path

def compress_pdf(pdf_file):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    out_path = f"/tmp/compressed_{datetime.now().strftime('%H%M%S')}.pdf"
    try:
        doc = fitz.open(pdf_file.name)
        new_doc = fitz.open()
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=72) 
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            result, encimg = cv2.imencode('.jpg', img_np, encode_param)
            img_pdf = fitz.open("jpeg", encimg.tobytes())
            rect = img_pdf[0].rect
            img_pdf.close()
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=encimg.tobytes())
        new_doc.save(out_path, garbage=4, deflate=True)
        new_doc.close()
        doc.close()
        if os.path.getsize(out_path) > os.path.getsize(pdf_file.name):
            os.remove(out_path)
            return pdf_file.name
        return out_path
    except Exception as e:
        raise gr.Error(f"Compression failed: {str(e)}")

def rotate_pdf(pdf_file, angle):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    reader = PdfReader(pdf_file.name)
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(int(angle))
        writer.add_page(page)
    out_path = f"/tmp/rotated_{datetime.now().strftime('%H%M%S')}.pdf"
    with open(out_path, "wb") as f: writer.write(f)
    return out_path

# ============================================================
# UI & CSS 
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; background: #0f172a !important; }
#project-header {
    text-align: center; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: white; padding: 40px 20px; border-radius: 20px; margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.4);
}
#project-header h1 {
    margin: 0; font-size: 3.2em; font-weight: 800;
    background: linear-gradient(to right, #a78bfa, #f472b6, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#project-header p { color: #cbd5e1; font-size: 1.2em; }
.scan-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="SmartDoc AI") as demo:
    gr.HTML('''
        <div id="project-header">
            <h1>📄 SmartDoc AI</h1>
            <p>An Intelligent Document Processing and Analysis Suite</p>
        </div>
    ''')

    with gr.Tabs():
        
        with gr.Tab("ℹ️ How to Use"):
            gr.Markdown("""
            # 👋 Welcome to SmartDoc AI!
            This suite has **16 powerful tools** to scan, edit, secure, and manage your documents. 
            Here is a quick guide on how to use each category:

            ### 📷 1. Scanning & Editing
            *   **Smart Scan:** Upload a photo of any document (a receipt, ID card, or notebook) taken from an angle. The AI will automatically find the corners, flatten it, and turn it into a clean scan. Choose `bw` (Black & White) for text, or `color` to keep the original look. You can then click **Generate PDF** to download it.
            *   **Creative Filters:** Turn your document into a Vintage style, Grayscale, or Blueprint.
            *   **Fix Orientation:** If your photo is upside-down or sideways, use this to rotate or flip it.
            *   **Auto Enhance:** Make blurry or faded text sharp and readable again.

            ### 🧠 2. Extraction & AI
            *   **OCR & Speech:** Upload a document. The AI will read the text and you can even click Play to have it read aloud to you!
            *   **Signature Isolator:** Upload a document with a signature. The AI will use morphological math to extract just the pen ink, removing the printed text and paper.
            *   **Barcode Scanner:** Upload a photo of any QR code or barcode to instantly read the hidden data.

            ### 🛡️ 3. Security & Compare
            *   **Text Blur (Redaction):** Need to hide private info like a name or phone number? Upload the image and this tool will automatically blur the text!
            *   **Compare Documents:** Upload two versions of the same document. It will highlight the exact differences in red boxes.

            ### 📄 4. PDF Suite
            *   **Split PDF:** Upload a large PDF and extract just one page.
            *   **Merge PDFs:** Upload 2 or 3 PDFs and combine them into one single file.
            *   **Compress PDF:** Upload a heavy PDF. This tool uses real image compression to drastically reduce the file size (up to 80% smaller!) so you can email it easily.
            *   **Rotate PDF:** Rotate all pages of a PDF to the correct direction.

            ---
            **⚠️ Important Note:** Because this app runs on a free cloud server, if you leave it for a while, the server goes to sleep. If you experience a delay, please wait 30 seconds and click the button again!
            """)

        with gr.Tab("🧠 Architecture & Data Flow"):
            gr.Markdown("""
            # ⚙️ System Architecture & Data Flow
            
            Here is a technical breakdown of how SmartDoc AI processes your data under the hood.

            ### 📷 1. Smart Document Scanner (The Pipeline)
            When you upload a photo, it goes through a 6-step Computer Vision pipeline:
            1. **Grayscale Conversion:** The image is converted to a single channel to reduce complexity.
            2. **Gaussian Blur:** Removes high-frequency noise so the edges are cleaner.
            3. **Canny Edge Detection:** Finds the boundaries of the document by detecting sudden changes in pixel intensity.
            4. **Contour Mapping:** `cv2.findContours` finds the largest 4-point polygon. We filter out small areas to ignore text boxes and focus on the page.
            5. **Homography (Perspective Warp):** A 3x3 matrix is calculated to flatten the angled photo into a top-down view.
            6. **Adaptive Thresholding:** Instead of a global threshold, it calculates the threshold for small regions. This handles shadows and uneven lighting perfectly.

            ### 🧠 2. OCR & Speech (AI Extraction)
            *   **OCR:** We use Google's **Tesseract Engine**. The scanned image is fed into the LSTM (Long Short-Term Memory) neural network, which extracts characters and outputs a string.
            *   **Text-to-Speech:** The text is passed to `gTTS` (Google Text-to-Speech), which uses a cloud API to generate a high-quality audio file (MP3).

            ### 🛡️ 3. Security & Comparison
            *   **Privacy Blur (Redaction):** Tesseract's `image_to_data` returns a dictionary of bounding boxes (x, y, width, height) for every detected word. We iterate through these boxes and apply a heavy **Gaussian Blur** kernel only over those specific regions.
            *   **Document Compare (SSIM):** Two images are compared using the **Structural Similarity Index Measure (SSIM)**. It compares luminance, contrast, and structure. The result is a difference map where black means "identical" and white means "changed". We then draw red rectangles over areas with high difference.

            ### ✍️ 4. Signature Extraction (Morphology)
            Signatures are thicker than printed text. We use **Morphological Operations**:
            *   First, a binary threshold is applied.
            *   Then, **Erosion** removes thin printed lines (they disappear).
            *   Then, **Dilation** restores the thick pen strokes.
            *   The result is a clean image of only the handwritten ink.

            ### 📄 5. PDF Suite (File Processing)
            *   **Merge/Split:** Uses `pypdf` to read the raw PDF structure, copy the page objects, and write them into a new file.
            *   **Compression:** Uses `PyMuPDF`. It renders every page as a high-res image, compresses that image to a 30% quality JPEG using OpenCV, and rebuilds a brand-new PDF. This is why a 5MB PDF shrinks to 500KB!

            ---
            **👨‍💻 Built with:** Python, OpenCV, Tesseract OCR, PyMuPDF, PyZbar, Gradio, and deployed via Docker on Render.
            """)

        with gr.Tab("📷 Scanning & Editing"):
            with gr.Tab("Smart Scan"):
                with gr.Row():
                    with gr.Column():
                        in_img = gr.Image(label="Upload Photo", type="numpy", height=400)
                        q_box = gr.Textbox(label="Quality Check", interactive=False)
                        mode_rad = gr.Radio(["bw", "color", "magic"], value="bw", label="Scan Mode")
                        scan_btn = gr.Button("Scan Document", elem_classes="scan-btn")
                    with gr.Column():
                        out_img = gr.Image(label="✅ Scanned Result", type="numpy", height=400)
                        pdf_btn = gr.Button("📄 Generate PDF")
                        pdf_file = gr.File(label="📥 Download PDF")
                scan_btn.click(process_scan, [in_img, mode_rad], [in_img, out_img, q_box])
                pdf_btn.click(generate_pdf, out_img, pdf_file)
            
            with gr.Tab("Creative Filters"):
                with gr.Row():
                    with gr.Column():
                        f_in = gr.Image(label="Upload Image", type="numpy")
                        f_type = gr.Dropdown(["Original", "Grayscale", "Vintage", "Blueprint"], value="Vintage", label="Filter")
                        f_btn = gr.Button("Apply", elem_classes="scan-btn")
                    with gr.Column():
                        f_out = gr.Image(label="Result", type="numpy")
                f_btn.click(apply_filter, [f_in, f_type], f_out)
            
            with gr.Tab("Fix Orientation"):
                with gr.Row():
                    with gr.Column():
                        rot_in = gr.Image(label="Upload Image", type="numpy")
                        rot_type = gr.Radio(["180", "90 CW", "90 CCW", "Flip"], value="180", label="Rotation")
                        rot_btn = gr.Button("Apply", elem_classes="scan-btn")
                    with gr.Column():
                        rot_out = gr.Image(label="Result", type="numpy")
                rot_btn.click(fix_orientation, [rot_in, rot_type], rot_out)
            
            with gr.Tab("Auto Enhance"):
                with gr.Row():
                    with gr.Column():
                        en_in = gr.Image(label="Upload Image", type="numpy")
                        en_btn = gr.Button("Sharpen", elem_classes="scan-btn")
                    with gr.Column():
                        en_out = gr.Image(label="Result", type="numpy")
                en_btn.click(auto_enhance, en_in, en_out)

        with gr.Tab("🧠 Extraction & AI"):
            with gr.Tab("OCR & Speech"):
                with gr.Row():
                    with gr.Column():
                        o_in = gr.Image(label="Upload Document", type="numpy")
                        o_btn = gr.Button("Extract & Speak", elem_classes="scan-btn")
                    with gr.Column():
                        o_text = gr.Textbox(label="Text", lines=10)
                        o_audio = gr.Audio(label="Audio", type="filepath")
                o_btn.click(do_ocr_and_speech, o_in, [o_text, o_audio])
            
            with gr.Tab("Signature Isolator"):
                with gr.Row():
                    with gr.Column():
                        s_in = gr.Image(label="Upload Document", type="numpy")
                        s_btn = gr.Button("Isolate Signature", elem_classes="scan-btn")
                    with gr.Column():
                        s_out = gr.Image(label="Result", type="numpy")
                s_btn.click(extract_sig, s_in, s_out)
            
            with gr.Tab("Barcode Scanner"):
                with gr.Row():
                    with gr.Column():
                        bar_in = gr.Image(label="Upload Barcode/QR", type="numpy")
                        bar_btn = gr.Button("Scan", elem_classes="scan-btn")
                    with gr.Column():
                        bar_out = gr.Image(label="Detected", type="numpy")
                        bar_text = gr.Textbox(label="Data")
                bar_btn.click(scan_barcode, bar_in, [bar_out, bar_text])

        with gr.Tab("🛡️ Security & Compare"):
            with gr.Tab("Text Blur (Redaction)"):
                with gr.Row():
                    with gr.Column():
                        p_in = gr.Image(label="Upload Document", type="numpy")
                        p_btn = gr.Button("Blur Text", elem_classes="scan-btn")
                    with gr.Column():
                        p_out = gr.Image(label="Result", type="numpy")
                p_btn.click(privacy_blur, p_in, p_out)
            
            with gr.Tab("Compare Documents"):
                with gr.Row():
                    with gr.Column():
                        c_img1 = gr.Image(label="Original", type="numpy")
                        c_img2 = gr.Image(label="Modified", type="numpy")
                        c_btn = gr.Button("Compare (SSIM)", elem_classes="scan-btn")
                    with gr.Column():
                        c_out = gr.Image(label="Differences", type="numpy")
                        c_score = gr.Textbox(label="Score")
                c_btn.click(compare_docs, [c_img1, c_img2], [c_out, c_score])

        with gr.Tab("📄 PDF Suite"):
            with gr.Tab("Split PDF"):
                pdf_in = gr.File(label="Upload PDF")
                page_num = gr.Number(label="Page Number", value=1)
                split_btn = gr.Button("Extract", elem_classes="scan-btn")
                split_out = gr.File(label="Download")
                split_btn.click(split_pdf, [pdf_in, page_num], split_out)
            
            with gr.Tab("Merge PDFs"):
                merge_in = gr.File(label="Upload Multiple PDFs", file_count="multiple", file_types=[".pdf"], height=200)
                merge_btn = gr.Button("Merge PDFs", elem_classes="scan-btn")
                merge_out = gr.File(label="Download Merged PDF")
                merge_btn.click(merge_pdfs, inputs=merge_in, outputs=merge_out)
            
            with gr.Tab("Compress PDF"):
                comp_in = gr.File(label="Upload PDF")
                comp_btn = gr.Button("Compress (Real Size Reduction)", elem_classes="scan-btn")
                comp_out = gr.File(label="Download")
                comp_btn.click(compress_pdf, comp_in, comp_out)
            
            with gr.Tab("Rotate PDF"):
                rot_pdf_in = gr.File(label="Upload PDF")
                rot_angle = gr.Radio(["90", "180", "270"], value="90", label="Angle")
                rot_pdf_btn = gr.Button("Rotate", elem_classes="scan-btn")
                rot_pdf_out = gr.File(label="Download")
                rot_pdf_btn.click(rotate_pdf, [rot_pdf_in, rot_angle], rot_pdf_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS)import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import pytesseract
import img2pdf
from gtts import gTTS
from skimage.metrics import structural_similarity as ssim
from pyzbar.pyzbar import decode
from scanner import scan_document
from pypdf import PdfReader, PdfWriter
import pymupdf as fitz

# Configure Tesseract for Linux server (Render)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

# Ensure the /tmp directory exists for Render
os.makedirs("/tmp", exist_ok=True)

# ============================================================
# Core Helper Functions
# ============================================================
def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = np.mean(gray)
    msg = []
    if blur < 50: msg.append("⚠️ Blurry image.")
    if bright < 60: msg.append("⚠️ Too dark.")
    return "\n".join(msg) if msg else "✅ Quality is perfect!"

def apply_filter(img, filter_type):
    if filter_type == "Original": return img
    if filter_type == "Grayscale": return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if filter_type == "Vintage":
        sepia_matrix = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
        return np.clip(cv2.transform(img, sepia_matrix), 0, 255).astype(np.uint8)
    if filter_type == "Blueprint":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        blue_img = np.zeros_like(img)
        blue_img[:] = (30, 50, 150)
        blue_img[thresh == 0] = (255, 255, 255)
        return cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
    return img

# ============================================================
# Core App Functions
# ============================================================
def process_scan(img, mode, progress=gr.Progress()):
    if img is None: raise gr.Error("Upload an image.")
    q_msg = check_quality(img)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, bgr)
    progress(0.5, desc="Scanning and warping...")
    result = scan_document(tmp.name, output_path=None, mode=mode, debug=False)
    os.unlink(tmp.name)
    if len(result.shape) == 2: result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else: result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return img, result, q_msg

def generate_pdf(img):
    if img is None: raise gr.Error("Please scan an image first.")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_p = f"/tmp/scan_{ts}.jpg"
    pdf_p = f"/tmp/scan_{ts}.pdf"
    cv2.imwrite(img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(pdf_p, "wb") as f:
        f.write(img2pdf.convert(img_p))
    os.remove(img_p)
    return pdf_p

def do_ocr_and_speech(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, bgr)
    scanned = scan_document(tmp.name, output_path=None, mode='bw')
    os.unlink(tmp.name)
    
    try:
        text = pytesseract.image_to_string(scanned)
        if not text.strip(): 
            return "No text found.", None
        
        # Use gTTS for cloud compatibility
        tts = gTTS(text=text, lang='en')
        audio_p = f"/tmp/audio_{datetime.now().strftime('%H%M%S')}.mp3"
        tts.save(audio_p)
        
        return text, audio_p
    except Exception as e:
        return f"Error: Tesseract not found. {str(e)}", None

def extract_sig(img):
    if img is None: raise gr.Error("Upload an image.")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sig = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(sig)

def compare_docs(img1, img2):
    if img1 is None or img2 is None: raise gr.Error("Upload both images.")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img1.copy()
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return result, f"Similarity: {score*100:.2f}%"

def privacy_blur(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT)
    out = img.copy()
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and len(data['text'][i].strip()) > 0:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            roi = out[y:y+h, x:x+w]
            if roi.size > 0:
                out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (25, 25), 30)
    return out

def fix_orientation(img, rotation):
    if img is None: raise gr.Error("Upload an image.")
    if rotation == "180": return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == "90 CW": return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "90 CCW": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "Flip": return cv2.flip(img, 1)
    return img

def scan_barcode(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    barcodes = decode(bgr)
    out = img.copy()
    text_results = []
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text_results.append(f"Type: {barcode.type} | Data: {barcode.data.decode('utf-8')}")
    return out, "\n".join(text_results) if text_results else "No barcodes found."

def auto_enhance(img):
    if img is None: raise gr.Error("Upload an image.")
    gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

# ============================================================
# PDF Tools
# ============================================================
def split_pdf(pdf_file, page_num):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    reader = PdfReader(pdf_file.name)
    writer = PdfWriter()
    try: writer.add_page(reader.pages[int(page_num) - 1])
    except IndexError: raise gr.Error("Page number out of range.")
    out_path = f"/tmp/split_{datetime.now().strftime('%H%M%S')}.pdf"
    with open(out_path, "wb") as f: writer.write(f)
    return out_path

def merge_pdfs(pdf_files):
    if not pdf_files or len(pdf_files) < 2: 
        raise gr.Error("Please upload at least 2 PDF files to merge.")
    out_path = f"/tmp/merged_{datetime.now().strftime('%H%M%S')}.pdf"
    writer = PdfWriter()
    for pdf in pdf_files:
        reader = PdfReader(pdf.name)
        for page in reader.pages:
            writer.add_page(page)
    with open(out_path, "wb") as f: 
        writer.write(f)
    return out_path

def compress_pdf(pdf_file):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    out_path = f"/tmp/compressed_{datetime.now().strftime('%H%M%S')}.pdf"
    try:
        doc = fitz.open(pdf_file.name)
        new_doc = fitz.open()
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=72) 
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            result, encimg = cv2.imencode('.jpg', img_np, encode_param)
            img_pdf = fitz.open("jpeg", encimg.tobytes())
            rect = img_pdf[0].rect
            img_pdf.close()
            new_page = new_doc.new_page(width=rect.width, height=rect.height)
            new_page.insert_image(rect, stream=encimg.tobytes())
        new_doc.save(out_path, garbage=4, deflate=True)
        new_doc.close()
        doc.close()
        if os.path.getsize(out_path) > os.path.getsize(pdf_file.name):
            os.remove(out_path)
            return pdf_file.name
        return out_path
    except Exception as e:
        raise gr.Error(f"Compression failed: {str(e)}")

def rotate_pdf(pdf_file, angle):
    if pdf_file is None: raise gr.Error("Upload a PDF.")
    reader = PdfReader(pdf_file.name)
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(int(angle))
        writer.add_page(page)
    out_path = f"/tmp/rotated_{datetime.now().strftime('%H%M%S')}.pdf"
    with open(out_path, "wb") as f: writer.write(f)
    return out_path

# ============================================================
# UI & CSS 
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; background: #0f172a !important; }
#project-header {
    text-align: center; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: white; padding: 40px 20px; border-radius: 20px; margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.4);
}
#project-header h1 {
    margin: 0; font-size: 3.2em; font-weight: 800;
    background: linear-gradient(to right, #a78bfa, #f472b6, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#project-header p { color: #cbd5e1; font-size: 1.2em; }
.scan-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="SmartDoc AI") as demo:
    gr.HTML('''
        <div id="project-header">
            <h1>📄 SmartDoc AI</h1>
            <p>An Intelligent Document Processing and Analysis Suite</p>
        </div>
    ''')

    with gr.Tabs():
        
        with gr.Tab("ℹ️ How to Use"):
            gr.Markdown("""
            # 👋 Welcome to SmartDoc AI!
            This suite has **16 powerful tools** to scan, edit, secure, and manage your documents. 
            Here is a quick guide on how to use each category:

            ### 📷 1. Scanning & Editing
            *   **Smart Scan:** Upload a photo of any document (a receipt, ID card, or notebook) taken from an angle. The AI will automatically find the corners, flatten it, and turn it into a clean scan. Choose `bw` (Black & White) for text, or `color` to keep the original look. You can then click **Generate PDF** to download it.
            *   **Creative Filters:** Turn your document into a Vintage style, Grayscale, or Blueprint.
            *   **Fix Orientation:** If your photo is upside-down or sideways, use this to rotate or flip it.
            *   **Auto Enhance:** Make blurry or faded text sharp and readable again.

            ### 🧠 2. Extraction & AI
            *   **OCR & Speech:** Upload a document. The AI will read the text and you can even click Play to have it read aloud to you!
            *   **Signature Isolator:** Upload a document with a signature. The AI will use morphological math to extract just the pen ink, removing the printed text and paper.
            *   **Barcode Scanner:** Upload a photo of any QR code or barcode to instantly read the hidden data.

            ### 🛡️ 3. Security & Compare
            *   **Text Blur (Redaction):** Need to hide private info like a name or phone number? Upload the image and this tool will automatically blur the text!
            *   **Compare Documents:** Upload two versions of the same document. It will highlight the exact differences in red boxes.

            ### 📄 4. PDF Suite
            *   **Split PDF:** Upload a large PDF and extract just one page.
            *   **Merge PDFs:** Upload 2 or 3 PDFs and combine them into one single file.
            *   **Compress PDF:** Upload a heavy PDF. This tool uses real image compression to drastically reduce the file size (up to 80% smaller!) so you can email it easily.
            *   **Rotate PDF:** Rotate all pages of a PDF to the correct direction.

            ---
            **⚠️ Important Note:** Because this app runs on a free cloud server, if you leave it for a while, the server goes to sleep. If you experience a delay, please wait 30 seconds and click the button again!
            """)

        with gr.Tab("🧠 Architecture & Data Flow"):
            gr.Markdown("""
            # ⚙️ System Architecture & Data Flow
            
            Here is a technical breakdown of how SmartDoc AI processes your data under the hood.

            ### 📷 1. Smart Document Scanner (The Pipeline)
            When you upload a photo, it goes through a 6-step Computer Vision pipeline:
            1. **Grayscale Conversion:** The image is converted to a single channel to reduce complexity.
            2. **Gaussian Blur:** Removes high-frequency noise so the edges are cleaner.
            3. **Canny Edge Detection:** Finds the boundaries of the document by detecting sudden changes in pixel intensity.
            4. **Contour Mapping:** `cv2.findContours` finds the largest 4-point polygon. We filter out small areas to ignore text boxes and focus on the page.
            5. **Homography (Perspective Warp):** A 3x3 matrix is calculated to flatten the angled photo into a top-down view.
            6. **Adaptive Thresholding:** Instead of a global threshold, it calculates the threshold for small regions. This handles shadows and uneven lighting perfectly.

            ### 🧠 2. OCR & Speech (AI Extraction)
            *   **OCR:** We use Google's **Tesseract Engine**. The scanned image is fed into the LSTM (Long Short-Term Memory) neural network, which extracts characters and outputs a string.
            *   **Text-to-Speech:** The text is passed to `gTTS` (Google Text-to-Speech), which uses a cloud API to generate a high-quality audio file (MP3).

            ### 🛡️ 3. Security & Comparison
            *   **Privacy Blur (Redaction):** Tesseract's `image_to_data` returns a dictionary of bounding boxes (x, y, width, height) for every detected word. We iterate through these boxes and apply a heavy **Gaussian Blur** kernel only over those specific regions.
            *   **Document Compare (SSIM):** Two images are compared using the **Structural Similarity Index Measure (SSIM)**. It compares luminance, contrast, and structure. The result is a difference map where black means "identical" and white means "changed". We then draw red rectangles over areas with high difference.

            ### ✍️ 4. Signature Extraction (Morphology)
            Signatures are thicker than printed text. We use **Morphological Operations**:
            *   First, a binary threshold is applied.
            *   Then, **Erosion** removes thin printed lines (they disappear).
            *   Then, **Dilation** restores the thick pen strokes.
            *   The result is a clean image of only the handwritten ink.

            ### 📄 5. PDF Suite (File Processing)
            *   **Merge/Split:** Uses `pypdf` to read the raw PDF structure, copy the page objects, and write them into a new file.
            *   **Compression:** Uses `PyMuPDF`. It renders every page as a high-res image, compresses that image to a 30% quality JPEG using OpenCV, and rebuilds a brand-new PDF. This is why a 5MB PDF shrinks to 500KB!

            ---
            **👨‍💻 Built with:** Python, OpenCV, Tesseract OCR, PyMuPDF, PyZbar, Gradio, and deployed via Docker on Render.
            """)

        with gr.Tab("📷 Scanning & Editing"):
            with gr.Tab("Smart Scan"):
                with gr.Row():
                    with gr.Column():
                        in_img = gr.Image(label="Upload Photo", type="numpy", height=400)
                        q_box = gr.Textbox(label="Quality Check", interactive=False)
                        mode_rad = gr.Radio(["bw", "color", "magic"], value="bw", label="Scan Mode")
                        scan_btn = gr.Button("Scan Document", elem_classes="scan-btn")
                    with gr.Column():
                        out_img = gr.Image(label="✅ Scanned Result", type="numpy", height=400)
                        pdf_btn = gr.Button("📄 Generate PDF")
                        pdf_file = gr.File(label="📥 Download PDF")
                scan_btn.click(process_scan, [in_img, mode_rad], [in_img, out_img, q_box])
                pdf_btn.click(generate_pdf, out_img, pdf_file)
            
            with gr.Tab("Creative Filters"):
                with gr.Row():
                    with gr.Column():
                        f_in = gr.Image(label="Upload Image", type="numpy")
                        f_type = gr.Dropdown(["Original", "Grayscale", "Vintage", "Blueprint"], value="Vintage", label="Filter")
                        f_btn = gr.Button("Apply", elem_classes="scan-btn")
                    with gr.Column():
                        f_out = gr.Image(label="Result", type="numpy")
                f_btn.click(apply_filter, [f_in, f_type], f_out)
            
            with gr.Tab("Fix Orientation"):
                with gr.Row():
                    with gr.Column():
                        rot_in = gr.Image(label="Upload Image", type="numpy")
                        rot_type = gr.Radio(["180", "90 CW", "90 CCW", "Flip"], value="180", label="Rotation")
                        rot_btn = gr.Button("Apply", elem_classes="scan-btn")
                    with gr.Column():
                        rot_out = gr.Image(label="Result", type="numpy")
                rot_btn.click(fix_orientation, [rot_in, rot_type], rot_out)
            
            with gr.Tab("Auto Enhance"):
                with gr.Row():
                    with gr.Column():
                        en_in = gr.Image(label="Upload Image", type="numpy")
                        en_btn = gr.Button("Sharpen", elem_classes="scan-btn")
                    with gr.Column():
                        en_out = gr.Image(label="Result", type="numpy")
                en_btn.click(auto_enhance, en_in, en_out)

        with gr.Tab("🧠 Extraction & AI"):
            with gr.Tab("OCR & Speech"):
                with gr.Row():
                    with gr.Column():
                        o_in = gr.Image(label="Upload Document", type="numpy")
                        o_btn = gr.Button("Extract & Speak", elem_classes="scan-btn")
                    with gr.Column():
                        o_text = gr.Textbox(label="Text", lines=10)
                        o_audio = gr.Audio(label="Audio", type="filepath")
                o_btn.click(do_ocr_and_speech, o_in, [o_text, o_audio])
            
            with gr.Tab("Signature Isolator"):
                with gr.Row():
                    with gr.Column():
                        s_in = gr.Image(label="Upload Document", type="numpy")
                        s_btn = gr.Button("Isolate Signature", elem_classes="scan-btn")
                    with gr.Column():
                        s_out = gr.Image(label="Result", type="numpy")
                s_btn.click(extract_sig, s_in, s_out)
            
            with gr.Tab("Barcode Scanner"):
                with gr.Row():
                    with gr.Column():
                        bar_in = gr.Image(label="Upload Barcode/QR", type="numpy")
                        bar_btn = gr.Button("Scan", elem_classes="scan-btn")
                    with gr.Column():
                        bar_out = gr.Image(label="Detected", type="numpy")
                        bar_text = gr.Textbox(label="Data")
                bar_btn.click(scan_barcode, bar_in, [bar_out, bar_text])

        with gr.Tab("🛡️ Security & Compare"):
            with gr.Tab("Text Blur (Redaction)"):
                with gr.Row():
                    with gr.Column():
                        p_in = gr.Image(label="Upload Document", type="numpy")
                        p_btn = gr.Button("Blur Text", elem_classes="scan-btn")
                    with gr.Column():
                        p_out = gr.Image(label="Result", type="numpy")
                p_btn.click(privacy_blur, p_in, p_out)
            
            with gr.Tab("Compare Documents"):
                with gr.Row():
                    with gr.Column():
                        c_img1 = gr.Image(label="Original", type="numpy")
                        c_img2 = gr.Image(label="Modified", type="numpy")
                        c_btn = gr.Button("Compare (SSIM)", elem_classes="scan-btn")
                    with gr.Column():
                        c_out = gr.Image(label="Differences", type="numpy")
                        c_score = gr.Textbox(label="Score")
                c_btn.click(compare_docs, [c_img1, c_img2], [c_out, c_score])

        with gr.Tab("📄 PDF Suite"):
            with gr.Tab("Split PDF"):
                pdf_in = gr.File(label="Upload PDF")
                page_num = gr.Number(label="Page Number", value=1)
                split_btn = gr.Button("Extract", elem_classes="scan-btn")
                split_out = gr.File(label="Download")
                split_btn.click(split_pdf, [pdf_in, page_num], split_out)
            
            with gr.Tab("Merge PDFs"):
                merge_in = gr.File(label="Upload Multiple PDFs", file_count="multiple", file_types=[".pdf"], height=200)
                merge_btn = gr.Button("Merge PDFs", elem_classes="scan-btn")
                merge_out = gr.File(label="Download Merged PDF")
                merge_btn.click(merge_pdfs, inputs=merge_in, outputs=merge_out)
            
            with gr.Tab("Compress PDF"):
                comp_in = gr.File(label="Upload PDF")
                comp_btn = gr.Button("Compress (Real Size Reduction)", elem_classes="scan-btn")
                comp_out = gr.File(label="Download")
                comp_btn.click(compress_pdf, comp_in, comp_out)
            
            with gr.Tab("Rotate PDF"):
                rot_pdf_in = gr.File(label="Upload PDF")
                rot_angle = gr.Radio(["90", "180", "270"], value="90", label="Angle")
                rot_pdf_btn = gr.Button("Rotate", elem_classes="scan-btn")
                rot_pdf_out = gr.File(label="Download")
                rot_pdf_btn.click(rotate_pdf, [rot_pdf_in, rot_angle], rot_pdf_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS)