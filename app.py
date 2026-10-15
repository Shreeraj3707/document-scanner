import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime

from scanner import scan_document


# ============================================================
# Backend functions
# ============================================================

def process_image(input_image, mode, progress=gr.Progress()):
    if input_image is None:
        raise gr.Error("Please upload an image first.")

    progress(0.1, desc="Loading image...")
    bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    tmp_in = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_in.close()
    cv2.imwrite(tmp_in.name, bgr)

    progress(0.3, desc="Detecting document edges...")

    try:
        result = scan_document(tmp_in.name, output_path=None, mode=mode, debug=False)
    except Exception as e:
        os.unlink(tmp_in.name)
        raise gr.Error(f"Scanning failed: {str(e)}")

    progress(0.7, desc="Enhancing output...")
    os.unlink(tmp_in.name)

    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    progress(1.0, desc="Done!")
    return result_rgb


def save_result(result_image):
    if result_image is None:
        raise gr.Error("Nothing to save. Run a scan first.")

    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/scan_{timestamp}.png"

    bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr)
    return filename


def process_batch(files, mode, progress=gr.Progress()):
    if not files:
        raise gr.Error("Please upload at least one image.")

    os.makedirs("output/batch", exist_ok=True)
    results = []

    for i, file_obj in enumerate(files):
        progress((i + 1) / len(files), desc=f"Processing {i+1}/{len(files)}...")

        img = cv2.imread(file_obj.name)
        if img is None:
            continue

        result = scan_document(file_obj.name, output_path=None, mode=mode, debug=False)

        if len(result.shape) == 2:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        else:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        out_path = f"output/batch/scan_{i+1}.png"
        cv2.imwrite(out_path, cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))
        results.append(out_path)

    return results


# ============================================================
# Custom CSS
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
#title-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
#title-header h1 {
    margin: 0;
    font-size: 2.2em;
}
#title-header p {
    margin: 8px 0 0 0;
    opacity: 0.9;
}
.scan-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: bold !important;
    font-size: 1.1em !important;
    border: none !important;
}
footer { display: none !important; }
"""


# ============================================================
# UI (Gradio 6 compatible)
# ============================================================

with gr.Blocks(title="Document Scanner") as demo:

    # ---------- Header ----------
    gr.HTML("""
        <div id="title-header">
            <h1>Document Scanner</h1>
            <p>Upload a photo of any document — get a clean, flat scan</p>
        </div>
    """)

    with gr.Tabs():

        # ==================== TAB 1: SINGLE SCAN ====================
        with gr.Tab("Single Scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(
                        label="Upload Document Photo",
                        type="numpy",
                        height=400
                    )

                    mode_selector = gr.Radio(
                        choices=["bw", "color"],
                        value="bw",
                        label="Enhancement Mode",
                        info="bw = clean black & white | color = shadow removal"
                    )

                    with gr.Row():
                        scan_btn = gr.Button(
                            "Scan Document",
                            variant="primary",
                            size="lg",
                            elem_classes="scan-btn"
                        )
                        clear_btn = gr.Button("Clear", size="lg")

                with gr.Column(scale=1):
                    output_img = gr.Image(
                        label="Scanned Result",
                        type="numpy",
                        height=400
                    )

                    with gr.Row():
                        save_btn = gr.Button("Save to Output Folder", size="lg")
                        saved_path = gr.Textbox(
                            label="Saved to",
                            interactive=False
                        )

            with gr.Accordion("How it works", open=False):
                gr.Markdown("""
                ### The Pipeline
                1. **Grayscale conversion** — simplify processing
                2. **Gaussian blur** — reduce noise before edge detection
                3. **Canny edge detection** — auto-tunes thresholds to find document borders
                4. **Contour detection** — finds the largest 4-sided shape
                5. **Perspective transform** — warps the document to a flat rectangle
                6. **Adaptive threshold** — produces a clean, scanner-like output

                ### Tips for best results
                - Place document on a **contrasting background**
                - Make sure all **4 corners** are visible
                - Photograph at a **slight angle**
                - Use **even lighting**
                """)

            scan_btn.click(
                fn=process_image,
                inputs=[input_img, mode_selector],
                outputs=output_img
            )

            save_btn.click(
                fn=save_result,
                inputs=output_img,
                outputs=saved_path
            )

            clear_btn.click(
                fn=lambda: (None, None, ""),
                inputs=None,
                outputs=[input_img, output_img, saved_path]
            )

        # ==================== TAB 2: BATCH ====================
        with gr.Tab("Batch Processing"):
            gr.Markdown("### Process multiple documents at once")
            gr.Markdown("*Upload 2-10 images. All results save to `output/batch/`.*")

            batch_files = gr.File(
                label="Upload Multiple Images",
                file_count="multiple",
                file_types=["image"],
                height=200
            )

            batch_mode = gr.Radio(
                choices=["bw", "color"],
                value="bw",
                label="Enhancement Mode"
            )

            batch_btn = gr.Button(
                "Process All",
                variant="primary",
                size="lg",
                elem_classes="scan-btn"
            )

            batch_output = gr.File(
                label="Download Results",
                file_count="multiple"
            )

            batch_btn.click(
                fn=process_batch,
                inputs=[batch_files, batch_mode],
                outputs=batch_output
            )

        # ==================== TAB 3: ABOUT ====================
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Project

            **Document Scanner** is a classical computer vision project that
            automatically detects, flattens, and enhances documents from photos.

            ### Technologies Used
            - **OpenCV** — image processing primitives
            - **NumPy** — numerical operations
            - **Gradio** — interactive web interface
            - **Python 3.10**

            ### Key Techniques
            | Technique | Purpose |
            |---|---|
            | Canny Edge Detection | Find document boundaries |
            | Douglas-Peucker Approximation | Simplify contours to 4 corners |
            | Perspective Transform (Homography) | Warp to flat rectangle |
            | Adaptive Thresholding | Handle uneven lighting |
            | Shadow Removal (division) | Preserve color while removing shadows |

            ### References
            - Canny, J. (1986). A Computational Approach to Edge Detection. IEEE TPAMI.
            - OpenCV Documentation: https://docs.opencv.org/
            """)

    gr.Markdown(
        "<center><small>Built with OpenCV + Gradio - Image Processing Course Project</small></center>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
        css=CUSTOM_CSS
    )