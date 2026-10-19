# 🚀 Ultimate AI Document Suite

An advanced Computer Vision application built with **OpenCV**, **Tesseract OCR**, and **Gradio**. It goes far beyond a simple scanner by integrating 8 different document intelligence features into a single web interface.

## 🌟 Features

1. **📷 Smart Scan:** Auto-detects document edges, applies perspective transform (warping), and auto-deskews (straightens) the text.
2. **🔍 Quality Assessment:** Warns the user if the input photo is blurry or too dark before scanning.
3. **🧠 OCR & Text-to-Speech:** Extracts printed text using Tesseract and reads it aloud using pyttsx3.
4. **🛡️ Privacy Blur (Redaction):** Automatically detects text bounding boxes and applies Gaussian blur to hide sensitive information.
5. **🎨 Watermarker:** Overlays custom diagonal text (e.g., "CONFIDENTIAL") with transparency.
6. **🔍 Document Comparator:** Uses Structural Similarity (SSIM) to compare two documents and highlights the exact differences.
7. **✍️ Signature Extractor:** Uses Morphological Operations to isolate thick pen ink from printed text.
8. **📄 PDF Export:** Instantly converts the scanned image into a high-quality PDF document.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Shreeraj3707/document-scanner.git
cd document-scanner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py