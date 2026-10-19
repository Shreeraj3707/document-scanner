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
from scanner import scan_document

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ============================================================
# Core Image Processing Functions
# ============================================================

def check_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = np.mean(gray)
    msg = []
    if blur < 50: msg.append("⚠️ Blurry image. Hold steady!")
    if bright < 60: msg.append("⚠️ Too dark. Use flash.")
    return "\n".join(msg) if msg else "✅ Quality is perfect!"

def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0: return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    if abs(angle) > 0.5:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

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
    progress(0.3, desc="Scanning and warping...")
    result = scan_document(tmp.name, output_path=None, mode=mode, debug=False)
    os.unlink(tmp.name)
    if len(result.shape) == 2: result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else: result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    progress(0.7, desc="Deskewing...")
    result = deskew(result)
    return result, q_msg

def save_pdf(img):
    if img is None: raise gr.Error("Nothing to save.")
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    img_p, pdf_p = f"output/scan_{ts}.png", f"output/scan_{ts}.pdf"
    cv2.imwrite(img_p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    with open(pdf_p, "wb") as f: f.write(img2pdf.convert(img_p))
    os.remove(img_p)
    return pdf_p

# ============================================================
# Tab 2: Watermarker
# ============================================================
def add_watermark(img, text):
    if img is None or not text: return img
    out = img.copy()
    h, w = out.shape[:2]
    # Create a blank overlay
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    # Put the text in the center
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1, w / 500)
    thickness = max(2, int(w / 300))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Rotate overlay
    M = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
    overlay = cv2.warpAffine(overlay, M, (w, h))
    # Blend
    out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    return out

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
# Tab 5: Comparator
# ============================================================
def compare_docs(img1, img2):
    if img1 is None or img2 is None: raise gr.Error("Upload both images.")
    # Resize img2 to match img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    diff = (diff * 255).astype("uint8")
    # Threshold the difference
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Find contours of changes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img1.copy()
    for c in contours:
        if cv2.contourArea(c) > 50: # ignore tiny noise
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return result, f"Similarity: {score*100:.2f}%"

# ============================================================
# Tab 6: Privacy Blur (Redaction)
# ============================================================
def privacy_blur(img):
    if img is None: raise gr.Error("Upload an image.")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Detect text using Tesseract bounding boxes
    data = pytesseract.image_to_data(bgr, output_type=pytesseract.Output.DICT)
    out = img.copy()
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60 and len(data['text'][i].strip()) > 0:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            # Blur the region
            roi = out[y:y+h, x:x+w]
            if roi.size > 0:
                out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (25, 25), 30)
    return out

# ============================================================
# UI & CSS
# ============================================================
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
#title-header { text-align: center; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white; padding: 25px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
.scan-btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="AI Document Suite") as demo:
    gr.HTML('<div id="title-header"><h1>🚀 Ultimate AI Document Suite</h1><p>Scan • Watermark • OCR • Redact • Compare</p></div>')

    with gr.Tabs():
        # --- TAB 1: SMART SCAN ---
        with gr.Tab("📷 Smart Scan"):
            with gr.Row():
                with gr.Column():
                    in_img = gr.Image(label="Upload Photo", type="numpy", height=400)
                    q_box = gr.Textbox(label="Quality Check", interactive=False)
                    mode_rad = gr.Radio(["bw", "color", "magic"], value="bw", label="Scan Mode")
                    scan_btn = gr.Button("Scan & Auto-Deskew", variant="primary", elem_classes="scan-btn")
                with gr.Column():
                    out_img = gr.Image(label="Scanned Result", type="numpy", height=400)
                    pdf_btn = gr.Button("📄 Download PDF", variant="primary")
                    pdf_out = gr.File(label="PDF", visible=False)
            scan_btn.click(process_scan, [in_img, mode_rad], [out_img, q_box])
            pdf_btn.click(save_pdf, out_img, pdf_out).then(lambda: gr.update(visible=True), outputs=pdf_out)

        # --- TAB 2: WATERMARKER ---
        with gr.Tab("🎨 Watermarker"):
            with gr.Row():
                with gr.Column():
                    w_in = gr.Image(label="Upload Document", type="numpy")
                    w_text = gr.Textbox(label="Watermark Text", value="CONFIDENTIAL")
                    w_btn = gr.Button("Apply Watermark", elem_classes="scan-btn")
                with gr.Column():
                    w_out = gr.Image(label="Watermarked Result", type="numpy")
            w_btn.click(add_watermark, [w_in, w_text], w_out)

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

        # --- TAB 4: PRIVACY BLUR ---
        with gr.Tab("🛡️ Privacy Blur"):
            gr.Markdown("Automatically detects and blurs text to hide sensitive information.")
            with gr.Row():
                with gr.Column():
                    p_in = gr.Image(label="Upload Document", type="numpy")
                    p_btn = gr.Button("Blur Text (Redact)", elem_classes="scan-btn")
                with gr.Column():
                    p_out = gr.Image(label="Redacted Result", type="numpy")
            p_btn.click(privacy_blur, p_in, p_out)

        # --- TAB 5: COMPARATOR ---
        with gr.Tab("🔍 Compare Docs"):
            gr.Markdown("Upload two versions of a document to see what changed.")
            with gr.Row():
                with gr.Column():
                    c_img1 = gr.Image(label="Original Document", type="numpy")
                    c_img2 = gr.Image(label="Modified Document", type="numpy")
                    c_btn = gr.Button("Find Differences", elem_classes="scan-btn")
                with gr.Column():
                    c_out = gr.Image(label="Differences Highlighted", type="numpy")
                    c_score = gr.Textbox(label="Similarity Score")
            c_btn.click(compare_docs, [c_img1, c_img2], [c_out, c_score])

        # --- TAB 6: SIGNATURES ---
        with gr.Tab("✍️ Signature Extractor"):
            with gr.Row():
                with gr.Column():
                    s_in = gr.Image(label="Upload Document", type="numpy")
                    s_btn = gr.Button("Isolate Signature", elem_classes="scan-btn")
                with gr.Column():
                    s_out = gr.Image(label="Extracted", type="numpy")
            s_btn.click(extract_sig, s_in, s_out)

        # --- TAB 7: ABOUT ---
        with gr.Tab("ℹ️ Info"):
            gr.Markdown("""
            ### The Ultimate Document Suite
            - **Smart Scan:** Perspective warp + Quality check + Deskewing.
            - **Watermark:** Add custom text overlays with transparency.
            - **OCR & Speech:** Tesseract text extraction + pyttsx3 AI voice.
            - **Privacy Blur:** Uses Tesseract bounding boxes to blur text.
            - **Comparator:** Structural Similarity (SSIM) to find document changes.
            - **Signature:** Morphological isolation of ink.
            """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS)