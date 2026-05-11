# ChromaFaceDetect

> **Color-Based Facial Detection System Using MATLAB and YCbCr Color Space Analysis**

![MATLAB](https://img.shields.io/badge/MATLAB-R2020a%2B-orange?style=flat-square&logo=mathworks)
![License](https://img.shields.io/github/license/n00rtahsin/ChromaFaceDetect?style=flat-square)
![Language](https://img.shields.io/badge/language-MATLAB-blue?style=flat-square)

---

## Overview

**ChromaFaceDetect** is a MATLAB-based face detection system that leverages the **YCbCr color space** to identify human faces in still images through skin color segmentation and morphological image processing — no deep learning or pre-trained models required.

Instead of relying on machine learning classifiers, this system uses the chrominance channels (Cb and Cr) of the YCbCr color space to isolate skin-toned regions, filters them by shape and size, and verifies facial features through edge analysis and vertical projection peaks.

---

## How It Works

The detection pipeline runs in four stages:

```
Input Image (RGB)
      │
      ▼
 1. Color Space Conversion
      RGB → YCbCr
      │
      ▼
 2. Skin Segmentation
      Threshold Cb & Cr channels → Binary skin mask
      │
      ▼
 3. Morphological Filtering
      Open → Close → Remove small regions → Label blobs
      │
      ▼
 4. Feature Verification
      Aspect ratio + Eccentricity + Edge peak detection
      │
      ▼
 Detected Faces (Bounding Boxes)
```

### Stage Breakdown

| Stage | Technique | Details |
|---|---|---|
| Color Conversion | `rgb2ycbcr()` | Separates luminance from chrominance for lighting-robust detection |
| Skin Mask | Threshold on Cb & Cr | `Cb ∈ [0.4, 0.6]`, `Cr ∈ [0.54, 0.68]` |
| Noise Removal | Morphological open/close | Disk structuring element of radius 5; removes regions < 1000 px |
| Candidate Filtering | Aspect ratio & eccentricity | Ratio ∈ [0.5, 1.5], eccentricity < 0.85, area > 2000 px |
| Feature Verification | Canny edges + peak detection | Requires ≥ 3 vertical projection peaks (eyes, nose, mouth) |

---

## Project Structure

```
ChromaFaceDetect/
├── facedetfinl.m          # Main MATLAB script (all detection logic)
├── ChromaFaceDetect.pdf   # Project report / documentation
├── README.md
└── SECURITY.md
```

### Functions in `facedetfinl.m`

| Function | Description |
|---|---|
| `main_face_detection()` | Entry point — opens a file dialog, loads image, runs detection, displays results |
| `detect_faces(img)` | Core detection: skin segmentation, morphological filtering, and region analysis |
| `has_facial_features(region)` | Validates a candidate region using Canny edge detection and vertical projection |
| `display_results(img, bbox)` | Renders the original image with green bounding boxes around detected faces |

---

## Requirements

- **MATLAB** R2020a or later
- **Image Processing Toolbox** (required for `rgb2ycbcr`, `imopen`, `imclose`, `bwlabel`, `regionprops`, `edge`, etc.)

To verify your toolbox is installed, run in MATLAB:
```matlab
ver('images')
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/n00rtahsin/ChromaFaceDetect.git
cd ChromaFaceDetect
```

**2. Open MATLAB** and navigate to the project directory:
```matlab
cd('path/to/ChromaFaceDetect')
```

**3. Run the detector**
```matlab
main_face_detection()
```

A file dialog will open — select any `.jpg`, `.png`, or `.bmp` image. The script will process the image and display results in a figure window with detected faces marked by green bounding boxes.

---

## Example Output

After running on an input image, a figure is displayed titled **"Face Detection Results"** showing:
- The original image
- Green rectangles drawn around each detected face
- A title indicating the total number of faces found (e.g., `Detected 2 faces`)

---

## Supported Image Formats

| Format | Extension |
|---|---|
| JPEG | `.jpg` |
| PNG  | `.png` |
| Bitmap | `.bmp` |

---

## Design Decisions & Limitations

**Why YCbCr?**
The YCbCr color space separates brightness (Y) from color information (Cb, Cr), making skin detection more robust to changes in lighting compared to raw RGB thresholding.

**Known limitations:**
- Performance degrades in very dark or overexposed lighting conditions
- Skin color thresholds are calibrated for general use and may not generalize to all ethnicities equally well
- Currently operates on static images only (no video/webcam support)
- Detection accuracy depends heavily on image resolution and face size

---

## Documentation

The `ChromaFaceDetect.pdf` file included in the repository provides the full technical report, including background theory on the YCbCr color model, the methodology, experimental results, and references.

---

## Contributing

Contributions are welcome! If you'd like to improve detection accuracy, add video support, or extend the algorithm, feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## License

This project is licensed under the terms found in the repository. See the [SECURITY.md](SECURITY.md) for security policy details.

---

## Author

**Noor Tahsin** — [@n00rtahsin](https://github.com/n00rtahsin)

---

*Built with MATLAB · Powered by chrominance*
