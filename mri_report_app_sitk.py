import streamlit as st
import tempfile
import os
import SimpleITK as sitk
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def load_dicom_series(uploaded_files, temp_dir):
    """
    Save uploaded DICOM files to a persistent temp folder and read them as a series with SimpleITK.
    """
    for uploaded_file in uploaded_files:
        tmp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

    # Get list of files that belong to the same series
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(temp_dir)

    if not series_IDs:
        st.error("No DICOM series found in uploaded files.")
        return None, None

    # Assume first series
    series_file_names = reader.GetGDCMSeriesFileNames(temp_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)

    image = reader.Execute()
    return image, series_file_names


def extract_metadata(dicom_file_path):
    """
    Extract metadata from one representative DICOM file using SimpleITK.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(dicom_file_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    tags = ["0010|0010",  # Patient Name
            "0010|0020",  # Patient ID
            "0010|0040",  # Patient Sex
            "0010|1010",  # Patient Age
            "0008|0020",  # Study Date
            "0008|0060"]  # Modality

    metadata = {}
    for tag in tags:
        value = reader.GetMetaData(tag) if reader.HasMetaDataKey(tag) else "N/A"
        metadata[tag] = value

    return metadata


def create_pdf_report(data_dict):
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp_pdf.name)
    styles = getSampleStyleSheet()

    story = [Paragraph("MRI Structured Report", styles['Title'])]
    for key, val in data_dict.items():
        story.append(Paragraph(f"<b>{key}:</b> {val}", styles['Normal']))

    doc.build(story)
    return tmp_pdf.name


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="MRI DICOM Report Tool (SimpleITK)", layout="centered")
st.title("🧠 MRI DICOM Structured Report Generator (SimpleITK)")

uploaded_files = st.file_uploader(
    "Upload all DICOM files from one MRI acquisition",
    type=["dcm"],
    accept_multiple_files=True
)

if uploaded_files:
    # Create a temp folder that will persist until the script ends
    temp_dir = tempfile.mkdtemp()

    st.info(f"{len(uploaded_files)} DICOM files uploaded. Processing...")

    image, file_names = load_dicom_series(uploaded_files, temp_dir)

    if image and file_names:
        # Extract metadata from first file
        meta = extract_metadata(file_names[0])

        # Example findings placeholder
        findings = "MRI volume loaded successfully. No abnormalities detected (placeholder)."

        # Map DICOM tags to readable names
        report_data = {
            "Patient Name": meta.get("0010|0010", "N/A"),
            "Patient ID": meta.get("0010|0020", "N/A"),
            "Sex": meta.get("0010|0040", "N/A"),
            "Age": meta.get("0010|1010", "N/A"),
            "Study Date": meta.get("0008|0020", "N/A"),
            "Modality": meta.get("0008|0060", "N/A"),
            "Findings": findings
        }

        pdf_path = create_pdf_report(report_data)

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Structured Report (PDF)",
                data=pdf_file,
                file_name="mri_report.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Could not load DICOM series.")

