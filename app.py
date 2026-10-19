import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import pytesseract
import img2pdf
import pyttsx3
from skimage.metrics import structural_similarity as ssim
from pyzbar.pyzbar import decode
from scanner import scan_document

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
# Tab 1: Smart Scan
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
    
    progress(1.0, desc="Done!")
    return img, result, q_msg

def generate_pdf(img):
    if img is None: raise gr.Error("Please scan an image first.")
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_p = f"output/scan_{ts}.jpg"
    pdf_p = f"output/scan_{ts}.pdf"
    cv2.imwrite(img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(pdf_p, "wb") as f:
        f.write(img2pdf.convert(img_p))
    os.remove(img_p)
    return pdf_p

# ============================================================
# Tab 3: OCR & Speech
# ============================================================
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
        engine = pyttsx3.init()
        audio_p = f"output/audio_{datetime.now().strftime('%H%M%S')}.mp3"
        engine.save_to_file(text, audio_p)
        engine.runAndWait()
        return text, audio_p
    except Exception as e:
        return f"Error: {e}", None

# ============================================================
# Tab 4: Signature
# ============================================================
def extract_sig(img):
    if img is None: raise gr.Error("Upload an image.")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    sig = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(sig)

# ============================================================
# Tab 5: Comparator (SSIM)
# ============================================================
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

# ============================================================
# Tab 6: Privacy Blur (Text)
# ============================================================
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

# ============================================================
# Tab 7: Fix Orientation
# ============================================================
def fix_orientation(img, rotation):
    if img is None: raise gr.Error("Upload an image.")
    if rotation == "180 (Upside Down)":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif rotation == "90 Clockwise":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "90 Counter-Clockwise":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == "Flip Horizontal":
        return cv2.flip(img, 1)
    return img

# ============================================================
# Tab 8: Barcode / QR Scanner
# ============================================================
def scan_barcode(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    barcodes = decode(bgr)
    out = img.copy()
    text_results = []
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        text_results.append(f"Type: {barcode_type} | Data: {barcode_data}")
    if not text_results:
        return out, "No barcodes or QR codes found."
    return out, "\n".join(text_results)

# ============================================================
# Tab 9: AI Auto-Enhance
# ============================================================
def auto_enhance(img):
    if img is None: raise gr.Error("Upload an image.")
    gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return sharpened

# ============================================================
# UI & CSS 
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; background: #0f172a !important; }
#project-header {
    text-align: center; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    color: white; padding: 40px 20px; border-radius: 20px; margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(67, 56, 202, 0.4); border: 1px solid rgba(255, 255, 255, 0.1);
}
#project-header h1 {
    margin: 0; font-size: 3.2em; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(to right, #a78bfa, #f472b6, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#project-header p { margin: 15px 0 0 0; font-size: 1.2em; color: #cbd5e1; font-weight: 400; }
.scan-btn { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
    color: white !important; font-weight: bold !important; border: none !important;
}
footer { display: none !important; }
@media (max-width: 768px) {
    #project-header { padding: 25px 15px; border-radius: 15px; }
    #project-header h1 { font-size: 2em; }
    #project-header p { font-size: 0.95em; }
    .gradio-container { padding: 5px !important; }
}
"""

with gr.Blocks(title="SmartDoc AI") as demo:
    gr.HTML('''
        <div id="project-header">
            <h1>📄 SmartDoc AI</h1>
            <p>An Intelligent Document Processing and Analysis Suite</p>
        </div>
    ''')

    with gr.Tabs():
        # --- TAB 1: SMART SCAN ---
        with gr.Tab("📷 Smart Scan"):
            with gr.Row():
                with gr.Column():
                    in_img = gr.Image(label="Upload Photo", type="numpy", height=400)
                    q_box = gr.Textbox(label="Quality Check", interactive=False)
                    mode_rad = gr.Radio(["bw", "color", "magic"], value="bw", label="Scan Mode")
                    scan_btn = gr.Button("Scan Document", variant="primary", elem_classes="scan-btn")
                with gr.Column():
                    out_img = gr.Image(label="✅ Scanned Result", type="numpy", height=400)
                    pdf_btn = gr.Button("📄 Generate PDF", variant="secondary")
                    pdf_file = gr.File(label="📥 Click to download PDF")
            
            scan_btn.click(process_scan, [in_img, mode_rad], [in_img, out_img, q_box])
            pdf_btn.click(generate_pdf, inputs=out_img, outputs=pdf_file)

        # --- TAB 2: CREATIVE FILTERS ---
        with gr.Tab("🎨 Creative Filters"):
            with gr.Row():
                with gr.Column():
                    f_in = gr.Image(label="Upload Document", type="numpy")
                    f_type = gr.Dropdown(["Original", "Grayscale", "Vintage", "Blueprint"], value="Vintage", label="Choose Filter")
                    f_btn = gr.Button("Apply Filter", elem_classes="scan-btn")
                with gr.Column():
                    f_out = gr.Image(label="Filtered Result", type="numpy")
            f_btn.click(apply_filter, [f_in, f_type], f_out)

        # --- TAB 3: OCR & SPEECH ---
        with gr.Tab("🧠 OCR & Speech"):
            with gr.Row():
                with gr.Column():
                    o_in = gr.Image(label="Upload Document", type="numpy")
                    o_btn = gr.Button("Extract & Speak", elem_classes="scan-btn")
                with gr.Column():
                    o_text = gr.Textbox(label="Text", lines=10)
                    o_audio = gr.Audio(label="Audio", type="filepath")
            o_btn.click(do_ocr_and_speech, o_in, [o_text, o_audio])

        # --- TAB 4: SIGNATURE ---
        with gr.Tab("✍️ Signature"):
            with gr.Row():
                with gr.Column():
                    s_in = gr.Image(label="Upload Document", type="numpy")
                    s_btn = gr.Button("Isolate Signature", elem_classes="scan-btn")
                with gr.Column():
                    s_out = gr.Image(label="Extracted", type="numpy")
            s_btn.click(extract_sig, s_in, s_out)

        # --- TAB 5: COMPARE ---
        with gr.Tab("🔍 Compare Docs"):
            with gr.Row():
                with gr.Column():
                    c_img1 = gr.Image(label="Original Document", type="numpy")
                    c_img2 = gr.Image(label="Modified Document", type="numpy")
                    c_btn = gr.Button("Find Differences", elem_classes="scan-btn")
                with gr.Column():
                    c_out = gr.Image(label="Differences Highlighted", type="numpy")
                    c_score = gr.Textbox(label="Similarity Score")
            c_btn.click(compare_docs, [c_img1, c_img2], [c_out, c_score])

        # --- TAB 6: PRIVACY BLUR ---
        with gr.Tab("🛡️ Text Blur"):
            with gr.Row():
                with gr.Column():
                    p_in = gr.Image(label="Upload Document", type="numpy")
                    p_btn = gr.Button("Blur Text", elem_classes="scan-btn")
                with gr.Column():
                    p_out = gr.Image(label="Redacted Result", type="numpy")
            p_btn.click(privacy_blur, p_in, p_out)

        # --- TAB 7: FIX ORIENTATION ---
        with gr.Tab("🔄 Fix Orientation"):
            gr.Markdown("Rotate or flip your document to fix upside-down or sideways scans.")
            with gr.Row():
                with gr.Column():
                    rot_in = gr.Image(label="Upload Document", type="numpy")
                    rot_type = gr.Radio(["180 (Upside Down)", "90 Clockwise", "90 Counter-Clockwise", "Flip Horizontal"], value="180 (Upside Down)", label="Choose Adjustment")
                    rot_btn = gr.Button("Apply Fix", elem_classes="scan-btn")
                with gr.Column():
                    rot_out = gr.Image(label="Fixed Result", type="numpy")
            rot_btn.click(fix_orientation, [rot_in, rot_type], rot_out)

        # --- TAB 8: BARCODE SCANNER ---
        with gr.Tab("📊 Barcode Scanner"):
            with gr.Row():
                with gr.Column():
                    bar_in = gr.Image(label="Upload Barcode/QR", type="numpy")
                    bar_btn = gr.Button("Scan Code", elem_classes="scan-btn")
                with gr.Column():
                    bar_out = gr.Image(label="Detected", type="numpy")
                    bar_text = gr.Textbox(label="Decoded Data", lines=5)
            bar_btn.click(scan_barcode, bar_in, [bar_out, bar_text])

        # --- TAB 9: AUTO ENHANCE ---
        with gr.Tab("✨ Auto Enhance"):
            with gr.Row():
                with gr.Column():
                    en_in = gr.Image(label="Upload Blurry Document", type="numpy")
                    en_btn = gr.Button("Sharpen Text", elem_classes="scan-btn")
                with gr.Column():
                    en_out = gr.Image(label="Crisp Result", type="numpy")
            en_btn.click(auto_enhance, en_in, en_out)

        # --- TAB 10: INFO ---
        with gr.Tab("ℹ️ Info"):
            gr.Markdown("### Welcome to the SmartDoc AI Suite!\n\nThis is a 10-tool AI suite built entirely with OpenCV, Tesseract OCR, and Gradio.")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS)