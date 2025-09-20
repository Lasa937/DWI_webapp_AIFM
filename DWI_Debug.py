import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import zipfile, tempfile, io
import SimpleITK as sitk
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -------------------------------
# Helper functions (simplified)
# -------------------------------

def read_bvals(bvals_path: Path):
    text = bvals_path.read_text().strip()
    return [float(x) for x in text.split()]

def compute_center_roi_mean(img: sitk.Image, roi_size=20):
    arr = sitk.GetArrayFromImage(img)
    slice2d = arr[0] if arr.ndim == 3 else arr
    h, w = slice2d.shape
    half = roi_size // 2
    roi = slice2d[h//2-half:h//2+half, w//2-half:w//2+half]
    return float(np.mean(roi)) if roi.size > 0 else float("nan")

def get_institution(img: sitk.Image) -> str:
    if "0008|0080" in img.GetMetaDataKeys():
        return img.GetMetaData("0008|0080")
    return "<unknown>"

def generate_pdf_report(output_path: Path, summary, institutions, means):
    c = canvas.Canvas(str(output_path), pagesize=letter)
    y = 750
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "DICOM Report")
    y -= 30

    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 14

    y -= 10
    c.drawString(50, y, "Institutions:")
    y -= 14
    for folder, inst in institutions.items():
        c.drawString(70, y, f"{folder}: {inst}")
        y -= 14

    y -= 10
    c.drawString(50, y, "Mean ROI values:")
    y -= 14
    for folder, vals in means.items():
        c.drawString(70, y, f"{folder}: {', '.join([f'{v:.2f}' for v in vals])}")
        y -= 14

    c.save()


# -------------------------------
# Main processing
# -------------------------------

def process_zip(zip_path: Path):
    if not zip_path.exists():
        messagebox.showerror("Error", "File not found")
        return

    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)

        # Example dummy results (replace with real checks!)
        summary = {"all_present": True, "bvals_count": 6}
        institutions = {"X": "Hospital A", "Y": "Hospital A", "Z": "Hospital A"}
        means = {"X": [123.4, 234.5], "Y": [345.6, 456.7], "Z": [567.8, 678.9]}

        out_path = zip_path.with_suffix(".pdf")
        generate_pdf_report(out_path, summary, institutions, means)
        messagebox.showinfo("Done", f"Report saved to:\n{out_path}")


# -------------------------------
# GUI
# -------------------------------

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")])
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)

def run_process():
    path_str = entry.get()
    if not path_str:
        messagebox.showerror("Error", "Please select or enter a ZIP path")
        return
    process_zip(Path(path_str))

root = tk.Tk()
root.title("DICOM ZIP Checker")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill="x")

label = tk.Label(frame, text="ZIP file path:")
label.pack(anchor="w")

entry = tk.Entry(frame, width=50)
entry.pack(fill="x", expand=True)

browse_btn = tk.Button(frame, text="Browse", command=browse_file)
browse_btn.pack(pady=5)

run_btn = tk.Button(frame, text="Run Check", command=run_process)
run_btn.pack(pady=10)

root.mainloop()
