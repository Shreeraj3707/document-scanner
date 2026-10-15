import cv2
import numpy as np
import argparse
import os
from utils import four_point_transform, resize_image


def find_document_contour(image):
    """Find largest 4-sided contour (the document)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def enhance_scan(warped):
    """Clean B&W output via adaptive threshold."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    return scanned


def remove_shadows(image):
    """Remove shadows while keeping color."""
    rgb_planes = cv2.split(image)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(norm)
    return cv2.merge(result_planes)


def scan_document(image_path, output_path=None, mode='bw', debug=False):
    """Full pipeline."""
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")

    resized = resize_image(original, max_dim=1000)
    ratio = original.shape[0] / resized.shape[0]

    contour = find_document_contour(resized)

    if contour is None:
        print("[!] No document detected. Returning original.")
        return original

    if debug:
        os.makedirs("output", exist_ok=True)
        debug_img = resized.copy()
        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
        cv2.imwrite("output/debug_contour.jpg", debug_img)
        print("[+] Saved output/debug_contour.jpg")

    contour = contour.reshape(4, 2) * ratio
    warped = four_point_transform(original, contour)

    if mode == 'bw':
        scanned = enhance_scan(warped)
    else:
        scanned = remove_shadows(warped)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, scanned)
        print(f"[+] Saved scan to: {output_path}")

    return scanned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Scanner")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("--output", default="output/scan.jpg")
    parser.add_argument("--mode", choices=['bw', 'color'], default='bw')
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    scan_document(
        image_path=args.input,
        output_path=args.output,
        mode=args.mode,
        debug=args.debug
    )