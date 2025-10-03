from __future__ import annotations
import io
import os
import uuid
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, abort
from werkzeug.utils import secure_filename

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

import streamlit as st
from io import BytesIO

###############################################
# App config
###############################################
app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / 'runs'
RUNS_DIR.mkdir(exist_ok=True)

# ZIP must contain bvals.txt and subfolders X, Y, Z with DICOM series
REQUIRED_SUBFOLDERS = ["X", "Y", "Z"]
ROI_DIAMETER_PX = 20
ROI_RADIUS_PX = ROI_DIAMETER_PX / 2.0

###############################################
# Templates
###############################################
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>DICOM ZIP → ADC Report</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
    .card { max-width: 720px; margin: 0 auto; padding: 1.5rem; border: 1px solid #e3e3e3; border-radius: 12px; box-shadow: 0 3px 12px rgba(0,0,0,0.04); }
    h1 { margin-top: 0; font-size: 1.4rem; }
    .hint { color: #555; font-size: .95rem; }
    .upload { display:flex; gap:.75rem; align-items:center; }
    input[type=file] { flex: 1; }
    button { appearance:none; border:0; background:#111; color:white; padding:.6rem 1rem; border-radius:10px; cursor:pointer; }
    .footer { color:#666; text-align:center; margin-top:1.5rem; font-size:.85rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>DICOM ZIP → ADC PDF report</h1>
    <p class="hint">Upload a ZIP structured as: <code>/bvals.txt</code>, and DICOM folders <code>/X</code>, <code>/Y</code>, <code>/Z</code>.<br>
    Processing starts immediately after upload. No ROI upload—an auto ROI (20&nbsp;px ⌀, COM of X[0]) is used.</p>
    <form class="upload" action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data">
      <input type="file" name="zipfile" accept=".zip" required>
      <button type="submit">Upload & Process</button>
    </form>
  </div>
  <p class="footer">&copy; {{ year }}</p>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Processed • {{ run_id }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .card { padding: 1rem; border: 1px solid #e3e3e3; border-radius: 12px; }
    a.button { display:inline-block; padding:.6rem 1rem; background:#111; color:#fff; text-decoration:none; border-radius:10px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; }
    th { background: #f6f6f6; text-align: left; }
    code { background:#f6f6f6; padding:2px 4px; border-radius:6px; }
  </style>
</head>
<body>
  <h2>Run <code>{{ run_id }}</code></h2>
  <div class="grid">
    <div class="card">
      <h3>Downloads</h3>
      <p>
        <a class="button" href="{{ url_for('files', run_id=run_id, filename=pdf_name) }}">Download PDF</a>
      </p>
      <ul>
        <li><a href="{{ url_for('files', run_id=run_id, filename=adc_plot_name) }}">ADC plot (PNG)</a></li>
        <li><a href="{{ url_for('files', run_id=run_id, filename=overlay_name) }}">Mask overlay (PNG)</a></li>
        <li><a href="{{ url_for('files', run_id=run_id, filename=csv_name) }}">ROI means (CSV)</a></li>
        <li><a href="{{ url_for('files', run_id=run_id, filename=npz_name) }}">Arrays+bvals (NPZ)</a></li>
      </ul>
    </div>
    <div class="card">
      <h3>Quick summary</h3>
      <table>
        <tr><th>ROI mask</th><td>{{ roi_summary }}</td></tr>
        <tr><th>b-values</th><td>{{ bcount }}</td></tr>
        <tr><th>X/Y/Z shape</th><td>{{ shape }}</td></tr>
      </table>
    </div>
  </div>
  <p style="margin-top:1rem"><a href="{{ url_for('index') }}">Process another ZIP</a></p>
</body>
</html>
"""

###############################################
# Core helpers
###############################################

def read_bvals_column(bvals_path: Path):
    if not bvals_path.exists():
        raise FileNotFoundError(f"Missing required file: {bvals_path}")
    vals = []
    for i, line in enumerate(bvals_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            raise ValueError(f"Non-numeric b-value on line {i}: {line!r}")
    if not vals:
        raise ValueError("bvals.txt is empty or contains no numeric values")
    return vals


def read_first_series_in_dir(d: Path) -> sitk.Image:
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {d}")
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(d), series_ids[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    return reader.Execute()


def _center_of_mass_2d(slice2d: np.ndarray):
    a = np.asarray(slice2d, dtype=float)
    a = np.nan_to_num(a, nan=0.0)
    total = a.sum()
    h, w = a.shape
    if not np.isfinite(total) or total <= 0:
        return (h / 2.0, w / 2.0)
    y_idx = np.arange(h)[:, None]
    x_idx = np.arange(w)[None, :]
    cy = float((y_idx * a).sum() / total)
    cx = float((x_idx * a).sum() / total)
    cy = min(max(cy, 0.0), h - 1.0)
    cx = min(max(cx, 0.0), w - 1.0)
    return (cy, cx)


def _circular_mask_2d(shape_hw, center_yx, radius_px):
    h, w = shape_hw
    cy, cx = center_yx
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    return dist2 <= (ROI_RADIUS_PX ** 2)


def make_auto_roi_mask_for_series(arr3d: np.ndarray):
    if arr3d.ndim != 3:
        raise ValueError(f'Expected 3D array (Z,H,W), got shape {arr3d.shape}')
    z, h, w = arr3d.shape
    first = arr3d[0]
    cy, cx = _center_of_mass_2d(first)
    mask2d = _circular_mask_2d((h, w), (cy, cx), ROI_RADIUS_PX)
    mask3d = np.broadcast_to(mask2d, (z, h, w))
    return mask3d.astype(bool), (cy, cx)


def means_over_roi_per_slice(arr3d: np.ndarray, mask3d: np.ndarray):
    if arr3d.shape != mask3d.shape:
        raise ValueError(f"Array shape {arr3d.shape} and mask shape {mask3d.shape} must match")
    z = arr3d.shape[0]
    means = []
    for i in range(z):
        roi_vals = arr3d[i][mask3d[i]]
        means.append(float(np.nanmean(roi_vals)) if roi_vals.size else np.nan)
    return np.asarray(means, dtype=float)


def fit_adc_nlls(bvals: np.ndarray, signals: np.ndarray):
    b = np.asarray(bvals, dtype=float)
    s = np.asarray(signals, dtype=float)
    m = np.isfinite(b) & np.isfinite(s) & (s > 0)
    b = b[m]
    s = s[m]
    if b.size < 2:
        return np.nan, np.nan, np.nan
    bmax = float(np.nanmax(b)) if b.size else 1000.0
    adc_upper = 0.01 if bmax <= 3000 else 0.005

    def best_S0_for(adc):
        w = np.exp(-b * adc)
        num = np.sum(s * w)
        den = np.sum(w * w)
        if den <= 0:
            return np.nan
        return max(num / den, 1e-12)

    def rss(adc, S0):
        pred = S0 * np.exp(-b * adc)
        r = s - pred
        return float(np.sum(r * r))

    best_adc, best_S0, best_rss = np.nan, np.nan, np.inf
    lo, hi = 0.0, adc_upper
    for _ in range(3):
        grid = np.linspace(lo, hi, 400)
        for a in grid:
            S0 = best_S0_for(a)
            if not np.isfinite(S0):
                continue
            r = rss(a, S0)
            if r < best_rss:
                best_rss, best_adc, best_S0 = r, float(a), float(S0)
        span = (hi - lo) / 20 if np.isfinite(best_adc) else (hi - lo)
        lo = max(0.0, best_adc - span)
        hi = min(adc_upper, best_adc + span)

    if np.isfinite(best_adc) and np.isfinite(best_S0):
        y = s
        yhat = best_S0 * np.exp(-b * best_adc)
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    return best_adc, best_S0, r2

###############################################
# Enhancements: CI for ADC and discrepancy plot
###############################################

def fit_adc_with_ci(bvals: np.ndarray, signals: np.ndarray, n_boot: int = 200, alpha: float = 0.05):
    """Fit ADC and return (adc, S0, r2, ci_lo, ci_hi).
    CI via bootstrap over (b, S) pairs using the existing NLLS fitter.
    """
    adc, S0, r2 = fit_adc_nlls(bvals, signals)
    b = np.asarray(bvals, dtype=float)
    s = np.asarray(signals, dtype=float)
    m = np.isfinite(b) & np.isfinite(s) & (s > 0)
    b = b[m]
    s = s[m]
    if b.size < 2 or not np.isfinite(adc):
        return adc, S0, r2, np.nan, np.nan
    n = b.size
    boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, _, _ = fit_adc_nlls(b[idx], s[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 5:
        return adc, S0, r2, np.nan, np.nan
    boot = np.sort(np.array(boot))
    lo = float(np.percentile(boot, alpha/2*100))
    hi = float(np.percentile(boot, (1-alpha/2)*100))
    return float(adc), float(S0), float(r2), lo, hi


def create_discrepancy_plot_png(out_path: Path, bvals, series_signals: dict, fits: dict, mean_adc: float):
    """Plot % discrepancy of per-b ADC (-(1/b) ln(S/S0)) vs mean ADC, per direction.
    %Δ = 100 * (ADC_b - mean_adc) / mean_adc
    """
    out_path = Path(out_path)
    plt.figure(figsize=(6.0, 4.0), dpi=150)
    for d in ['X','Y','Z']:
        b = np.asarray(bvals, dtype=float)
        s = np.asarray(series_signals[d], dtype=float)
        S0 = fits[d]['S0']
        m = np.isfinite(b) & np.isfinite(s) & (s > 0) & np.isfinite(S0) & (S0 > 0) & (b > 0)
        if not m.any():
            continue
        b_use = b[m]
        s_use = s[m]
        adc_b = -(1.0 / b_use) * np.log(s_use / S0)
        perc = 100.0 * (adc_b - mean_adc) / mean_adc
        plt.scatter(b_use, perc, s=16, label=f"{d}")
    plt.axhline(0, linewidth=1)
    plt.xlabel('b-value')
    plt.ylabel('% discrepancy vs mean ADC')
    plt.title('Per-b ADC discrepancies')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def create_adc_plot_png(out_path: Path, bvals, series_signals: dict, fits: dict):
    out_path = Path(out_path)
    plt.figure(figsize=(6.0, 4.0), dpi=150)
    for d in ['X','Y','Z']:
        b = np.asarray(bvals, dtype=float)
        s = np.asarray(series_signals[d], dtype=float)
        m = np.isfinite(b) & np.isfinite(s) & (s > 0)
        plt.scatter(b[m], s[m], label=f"{d} data", s=16)
        adc, S0 = fits[d]['ADC'], fits[d]['S0']
        if np.isfinite(adc) and np.isfinite(S0):
            bfit = np.linspace(float(np.nanmin(b[m])) if m.any() else 0.0,
                               float(np.nanmax(b[m])) if m.any() else 1.0, 200)
            sfit = S0 * np.exp(-bfit * adc)
            plt.plot(bfit, sfit, label=f"{d} fit")
    plt.xlabel('b-value')
    plt.ylabel('Mean signal in ROI')
    plt.title('DWI signal vs b with monoexponential fits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def create_mask_overlay_png(out_path: Path, first_slice: np.ndarray, mask2d: np.ndarray):
    out_path = Path(out_path)
    plt.figure(figsize=(4.0, 4.0), dpi=150)
    plt.imshow(first_slice, cmap='gray')
    try:
        from skimage import measure
        contours = measure.find_contours(mask2d.astype(float), 0.5)
        for c in contours:
            plt.plot(c[:,1], c[:,0], linewidth=1.5, color='red')
    except Exception:
        edges = mask2d.astype(float)
        plt.imshow(np.ma.masked_where(edges == 0, edges), alpha=0.3)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_path


def draw_table_with_grid(c, x, y, col_widths, rows, row_height=16, header_rows=1):
    n_rows = len(rows)
    n_cols = len(col_widths)
    total_w = sum(col_widths)
    for r in range(n_rows + 1):
        yy = y - r * row_height
        c.line(x, yy, x + total_w, yy)
    xx = x
    for cw in col_widths:
        c.line(xx, y, xx, y - n_rows * row_height)
        xx += cw
    c.line(x + total_w, y, x + total_w, y - n_rows * row_height)
    yy = y - row_height + 4
    for i, row in enumerate(rows):
        xx = x + 4
        c.setFont('Helvetica-Bold' if i < header_rows else 'Helvetica', 10)
        for j, cell in enumerate(row):
            c.drawString(xx, yy, str(cell))
            xx += col_widths[j]
        yy -= row_height
    return y - n_rows * row_height


def generate_pdf_report(
    output_path: Path,
    summary,
    bvals_list,
    adc_table: dict | None = None,
    adc_plot_png: Path | None = None,
    mask_overlay_png: Path | None = None,
    discrepancy_plot_png: Path | None = None,
):
    from reportlab.lib.utils import ImageReader

    # ---------- helpers ----------
    def draw_image_fit(c, img_path, left, bottom, right, top, pad=8):
        """Fit & center image in [left,right]x[bottom,top] with padding."""
        try:
            ir = ImageReader(str(img_path))
            iw, ih = ir.getSize()
        except Exception:
            return
        box_w = max(1, (right - left) - 2 * pad)
        box_h = max(1, (top - bottom) - 2 * pad)
        scale = min(box_w / iw, box_h / ih)
        w = iw * scale
        h = ih * scale
        x = left + (right - left - w) / 2
        y = bottom + (top - bottom - h) / 2
        c.drawImage(str(img_path), x, y, width=w, height=h, preserveAspectRatio=True, mask="auto")

    def _fmt_ci(lo, hi):
        if np.isfinite(lo) and np.isfinite(hi):
            return f"[{lo*1e6:.0f}, {hi*1e6:.0f}]"
        return "—"

    # ---------- page & margins ----------
    margin = 57  # ~2 cm
    width, height = letter
    c = rl_canvas.Canvas(str(output_path), pagesize=letter)

    inner_left   = margin
    inner_right  = width - margin
    inner_bottom = margin
    inner_top    = height - margin
    content_w    = inner_right - inner_left
    content_h    = inner_top - inner_bottom

    # ---------- title & summary ----------
    y = inner_top
    c.setFont("Helvetica-Bold", 16)
    c.drawString(inner_left, y, "DICOM ADC Report")
    y -= 18

    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(inner_left, y, f"{k}: {v}")
        y -= 12

    # b-values directly under the summary
    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inner_left, y, "b-values:")
    y -= 14
    c.setFont("Helvetica", 10)
    preview_full = ", ".join((str(int(v)) if float(v).is_integer() else f"{v:g}") for v in bvals_list)
    text_obj = c.beginText(inner_left, y)
    max_chars = int((inner_right - inner_left) / 5.2)  # crude wrap estimate
    for i in range(0, len(preview_full), max_chars):
        text_obj.textLine(preview_full[i:i + max_chars])
        y -= 12
    c.drawText(text_obj)
    y -= 8

    # ---------- reserve bottom area for two side-by-side graphs ----------
    gap = 50
    bottom_h = max(140, int(content_h * 0.38))  # ~38% of content height
    bottom_bottom = inner_bottom
    bottom_top    = inner_bottom + bottom_h

    # Boxes for the two graphs (bottom row)
    left_box  = (inner_left, bottom_bottom, inner_left + content_w / 2 - gap / 2, bottom_top)
    right_box = (inner_left + content_w / 2 + gap / 2, bottom_bottom, inner_right, bottom_top)

    # ---------- top section (above bottom graphs) ----------
    top_top    = y
    top_bottom = bottom_top + gap
    top_h      = max(1, top_top - top_bottom)

    # Top-right image: width = 1/3 of inner width, anchored to top-right
    if mask_overlay_png is not None:
        img_target_w = content_w / 3.0
        # Determine image height to not exceed available top area
        try:
            ir = ImageReader(str(mask_overlay_png))
            iw, ih = ir.getSize()
            scale = img_target_w / iw
            img_h = ih * scale
            max_h = top_h  # can't exceed
            if img_h > max_h:
                scale = max_h / ih
                img_w = iw * scale
                img_h = max_h
            else:
                img_w = img_target_w
        except Exception:
            img_w = img_target_w
            img_h = top_h

        # Anchor to top-right inside margins
        img_left   = inner_right - 2*img_w
        img_right  = inner_right - img_w
        img_top    = top_top -gap/2
        img_bottom = img_top - img_h
        draw_image_fit(c, mask_overlay_png, img_left, img_bottom, img_right, img_top, pad=0)

        # Table must go below the image (to avoid overlap)
        table_top_limit = img_bottom - gap
    else:
        # No image: use the whole top area
        table_top_limit = top_top

    # ---------- centered ADC table (no S0 column, ADC×10⁻⁶ + 95% CI) ----------
    if adc_table is not None:
        # Build rows
        rows = [["Direction", "ADC (×10⁻⁶ 1/mm²)", "95% CI", "R²"]]
        for d in ["X", "Y", "Z", "Mean"]:
            if d in adc_table:
                row = adc_table[d]
                adc = row.get("ADC", np.nan)
                r2  = row.get("R2",  np.nan)
                lo  = row.get("CI_lo", np.nan)
                hi  = row.get("CI_hi", np.nan)
                adc_u = adc * 1e6 if np.isfinite(adc) else np.nan
                rows.append([
                    d,
                    f"{adc_u:.0f}" if np.isfinite(adc_u) else "",
                    _fmt_ci(lo, hi),
                    f"{r2:.4f}" if np.isfinite(r2) else "",
                ])

        # Choose a nice table width and center it
        pad_table = 10
        table_w = min(content_w * 0.75, content_w)   # up to 75% of inner width
        table_x = inner_left + (content_w - table_w) / 2.0
        # Column widths
        col_widths = [table_w * 0.22, table_w * 0.30, table_w * 0.33, table_w * 0.15]
        # Place the table top at table_top_limit; if not enough vertical room, clamp to bottom_top
        table_top_y = max(table_top_limit, bottom_top + 60)  # ensure it stays above bottom graphs
        draw_table_with_grid(c, table_x + pad_table, table_top_y, col_widths, rows, row_height=18)

    # ---------- bottom graphs side-by-side ----------
    if adc_plot_png is not None:
        x0, y0, x1, y1 = left_box
        draw_image_fit(c, adc_plot_png, x0, y0, x1, y1, pad=8)

    if discrepancy_plot_png is not None:
        x0, y0, x1, y1 = right_box
        draw_image_fit(c, discrepancy_plot_png, x0, y0, x1, y1, pad=8)

    c.save()


###############################################
# Processing pipeline
###############################################


# To run:  streamlit run streamlit_app.py

def build_pdf_bytes(summary, bvals, adc_table, adc_plot_png_bytes, mask_overlay_png_bytes, discrepancy_png_bytes):
    # Write images to temp files to feed reportlab
    with tempfile.TemporaryDirectory() as t:
        tp = Path(t)
        plot_p = tp / 'plot.png'
        overlay_p = tp / 'overlay.png'
        discrepancy_p = tp / 'disc.png'
        with open(plot_p, 'wb') as f: f.write(adc_plot_png_bytes)
        with open(overlay_p, 'wb') as f: f.write(mask_overlay_png_bytes)
        with open(discrepancy_p, 'wb') as f: f.write(discrepancy_png_bytes)
        pdf_p = tp / 'report.pdf'
        generate_pdf_report(pdf_p, summary, bvals, adc_table=adc_table, adc_plot_png=plot_p, mask_overlay_png=overlay_p, discrepancy_plot_png=discrepancy_p)
        return pdf_p.read_bytes()

import streamlit as st
from io import BytesIO

st.set_page_config(page_title="DICOM ZIP → ADC Report", layout="centered")
st.title("DICOM ZIP → ADC Report")
st.caption("Upload a ZIP with /bvals.txt and DICOM folders /X, /Y, /Z. Auto-ROI (20 px ⌀, COM on X[0]).")

u = st.file_uploader("Upload ZIP", type=["zip"], accept_multiple_files=False)
if u is not None:
    # Save to temp and process immediately
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        up_zip = td_path / u.name
        up_zip.write_bytes(u.getbuffer())

        try:
            # Process (reusing helpers)
            with zipfile.ZipFile(up_zip, 'r') as zf:
                zf.extractall(td_path)

            bvals_path = td_path / 'bvals.txt'
            bvals = read_bvals_column(bvals_path)
            arrays = {}
            shapes = []
            for name in REQUIRED_SUBFOLDERS:
                if not (td_path / name).is_dir():
                    st.error(f"Missing required subfolder: {name}")
                    st.stop()
                img = read_first_series_in_dir(td_path / name)
                nd = sitk.GetArrayFromImage(img)
                arrays[name] = nd
                shapes.append(nd.shape)
            if not (shapes[0] == shapes[1] == shapes[2]):
                st.error(f"X/Y/Z shapes differ: {shapes}")
                st.stop()

            mask3d, (cy, cx) = make_auto_roi_mask_for_series(arrays['X'])
            
            # Per-slice means
            means_X = means_over_roi_per_slice(arrays['X'], mask3d)
            means_Y = means_over_roi_per_slice(arrays['Y'], mask3d)
            means_Z = means_over_roi_per_slice(arrays['Z'], mask3d)

	    # ADC fits per direction
            b = np.asarray(bvals, dtype=float)
            series_signals = {'X': means_X, 'Y': means_Y, 'Z': means_Z}
            fits = {}
            for d in ['X','Y','Z']:
               adc, S0, r2, lo, hi = fit_adc_with_ci(b, series_signals[d])
               fits[d] = {'ADC': adc, 'S0': S0, 'R2': r2, 'CI_lo': lo, 'CI_hi': hi}
            mean_adc = float(np.nanmean([fits['X']['ADC'], fits['Y']['ADC'], fits['Z']['ADC']]))
            adc_table = {**{k: {kk: vv for kk, vv in v.items()} for k, v in fits.items()},'Mean': {'ADC': mean_adc, 'R2': np.nan, 'CI_lo': np.nan, 'CI_hi': np.nan}}

	    # Plot and overlay (to bytes)
            buf_plot = BytesIO()
            with tempfile.TemporaryDirectory() as t2:
                p = Path(t2)/'plot.png'
                create_adc_plot_png(p, bvals, series_signals, fits)
                buf_plot.write(p.read_bytes()); buf_plot.seek(0)
            buf_overlay = BytesIO()
            with tempfile.TemporaryDirectory() as t3:
                x0 = arrays['X'][0]
                m2 = mask3d[0]
                p2 = Path(t3)/'overlay.png'
                create_mask_overlay_png(p2, x0, m2)
                buf_overlay.write(p2.read_bytes()); buf_overlay.seek(0)
            buf_disc = BytesIO()
            with tempfile.TemporaryDirectory() as t4:
                p3 = Path(t4)/'disc.png'
                create_discrepancy_plot_png(p3, bvals, series_signals, fits, mean_adc)
                buf_disc.write(p3.read_bytes()); buf_disc.seek(0)
                
        except Exception as e:
            st.error(f"Processing failed: {e}")
    
    # Show results
    st.subheader("Results")
    st.write({k: v for k,v in {
        'ROI mask': f'Auto (20 px ⌀), COM on X[0] at (y={cy:.1f}, x={cx:.1f})',
        'b-values count': len(bvals),
        'X/Y/Z shape': arrays['X'].shape,
    }.items()})

    import pandas as pd
    def _fmt_ci(lo, hi):
        if np.isfinite(lo) and np.isfinite(hi):
            return f"[{lo*1e6:.0f}, {hi*1e6:.0f}]"
        return '—'
    df = pd.DataFrame([
        {'Direction':'X', 'ADC (×10⁻⁶ 1/mm²)':round(fits['X']['ADC']*1e6), '95% CI':_fmt_ci(fits['X']['CI_lo'], fits['X']['CI_hi']), 'R²':round(fits['X']['R2'],4)},
        {'Direction':'Y', 'ADC (×10⁻⁶ 1/mm²)':round(fits['Y']['ADC']*1e6), '95% CI':_fmt_ci(fits['Y']['CI_lo'], fits['Y']['CI_hi']), 'R²':round(fits['Y']['R2'],4)},
        {'Direction':'Z', 'ADC (×10⁻⁶ 1/mm²)':round(fits['Z']['ADC']*1e6), '95% CI':_fmt_ci(fits['Z']['CI_lo'], fits['Z']['CI_hi']), 'R²':round(fits['Z']['R2'],4)},
        {'Direction':'Mean', 'ADC (×10⁻⁶ 1/mm²)':round(mean_adc*1e6), '95% CI':'—', 'R²':'—'},
    ])
    st.dataframe(df, use_container_width=True)

    st.image(buf_plot, caption='DWI data and fits')
    st.image(buf_overlay, caption='X: first slice with ROI contour (red)')
    st.image(buf_disc, caption='Per-b ADC % discrepancies vs mean ADC')

    # Build downloadable artifacts
    # CSV
    csv_io = io.StringIO()
    import csv as _csv
    w = _csv.writer(csv_io)
    w.writerow(['bval', 'mean_X', 'mean_Y', 'mean_Z'])
    n = max(len(bvals), len(means_X), len(means_Y), len(means_Z))
    for i in range(n):
        bv = bvals[i] if i < len(bvals) else np.nan
        xx = means_X[i] if i < len(means_X) else np.nan
        yy = means_Y[i] if i < len(means_Y) else np.nan
        zz = means_Z[i] if i < len(means_Z) else np.nan
        w.writerow([bv, xx, yy, zz])
    st.download_button('Download ROI means (CSV)', csv_io.getvalue(), file_name='roi_means.csv', mime='text/csv')

    # NPZ
    npz_buf = BytesIO()
    np.savez_compressed(npz_buf, X=arrays['X'], Y=arrays['Y'], Z=arrays['Z'], bvals=np.array(bvals, dtype=float))
    st.download_button('Download arrays+bvals (NPZ)', npz_buf.getvalue(), file_name='arrays_bvals.npz')

    # PDF
    pdf_bytes = build_pdf_bytes(summary={'ROI mask': f'Auto (20 px ⌀), COM on X[0] at (y={cy:.1f}, x={cx:.1f})','b-values count': len(bvals), 'X/Y/Z shape': arrays['X'].shape},
    bvals=bvals,
    adc_table=adc_table,
    adc_plot_png_bytes=buf_plot.getvalue(),
    mask_overlay_png_bytes=buf_overlay.getvalue(),
    discrepancy_png_bytes=buf_disc.getvalue())
    st.download_button('Download PDF report', pdf_bytes, file_name='report.pdf', mime='application/pdf')



    

