import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from utils import resize_image, four_point_transform
from scanner import find_document_contour, enhance_scan


def visualize_pipeline(image_path):
    os.makedirs("output", exist_ok=True)

    original = cv2.imread(image_path)
    if original is None:
        print(f"Could not load: {image_path}")
        return

    resized = resize_image(original, max_dim=800)

    # Step 1: Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 2: Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Edges (try a mid-range threshold for visualization)
    edged = cv2.Canny(blurred, 50, 150)

    # Step 4: Contour
    contour = find_document_contour(resized)
    contour_vis = resized.copy()
    if contour is not None:
        cv2.drawContours(contour_vis, [contour], -1, (0, 255, 0), 3)

    # Step 5: Warp + enhance
    if contour is not None:
        warped = four_point_transform(resized, contour.reshape(4, 2))
        scanned = enhance_scan(warped)
    else:
        warped = resized
        scanned = gray

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Input Photo")

    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title("2. Grayscale")

    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title("3. Gaussian Blur")

    axes[1, 0].imshow(edged, cmap='gray')
    axes[1, 0].set_title("4. Canny Edges")

    axes[1, 1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Detected Contour")

    axes[1, 2].imshow(scanned, cmap='gray')
    axes[1, 2].set_title("6. Final Scan")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("output/pipeline_diagram.png", dpi=120, bbox_inches='tight')
    print("[+] Saved output/pipeline_diagram.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pipeline.py input/test.jpg")
        sys.exit(1)
    visualize_pipeline(sys.argv[1])