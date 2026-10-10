"""
PDF Generation Service for OncoVision Clinical Research Platform.

Generates a professional, printable PDF report for an individual saved ultrasound scan.
Includes patient metadata, scan diagnostics, class probabilities, model version,
original ultrasound image, Grad-CAM explanation heatmap, and mandatory clinical disclaimer.
"""

import io
from datetime import datetime
from typing import Optional, Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage


def _prepare_image_flowable(image_bytes: Optional[bytes], max_width: float = 230, max_height: float = 230) -> Paragraph | Image:
    """
    Converts raw image bytes into a ReportLab Image Flowable with aspect ratio constraint.
    Returns a fallback Paragraph if image_bytes is None or corrupted.
    """
    if not image_bytes:
        return Paragraph(
            "<font color='#64748b'><b>[ Image Not Available ]</b><br/><font size='8'>Stored image file could not be retrieved from storage bucket.</font></font>",
            getSampleStyleSheet()['Normal']
        )

    try:
        pil_img = PILImage.open(io.BytesIO(image_bytes))
        width, height = pil_img.size

        # Calculate aspect ratio scaling
        scale = min(max_width / width, max_height / height)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))

        # Re-encode PIL image into PNG bytes for ReportLab
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        return Image(img_buffer, width=new_w, height=new_h)
    except Exception as ex:
        print(f"[WARNING] PDF image processing failed: {str(ex)}")
        return Paragraph(
            "<font color='#ef4444'><b>[ Image Processing Error ]</b><br/><font size='8'>Failed to decode image data.</font></font>",
            getSampleStyleSheet()['Normal']
        )


def generate_scan_report_pdf(
    patient: Dict[str, Any],
    scan: Dict[str, Any],
    original_image_bytes: Optional[bytes] = None,
    gradcam_image_bytes: Optional[bytes] = None
) -> bytes:
    """
    Generates a complete server-side PDF report buffer for a saved scan record.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0284c7')
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6
    )

    cell_bold_style = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    cell_value_style = ParagraphStyle(
        'CellValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    disclaimer_title_style = ParagraphStyle(
        'DisclaimerTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#991b1b')
    )

    disclaimer_text_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#7f1d1d')
    )

    story = []

    # 1. Header Banner Table
    gen_time_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    header_data = [
        [
            Paragraph("<b>ONCOVISION</b><br/><font size='9' color='#0284c7'>Breast Ultrasound AI Analysis System</font>", title_style),
            Paragraph(f"<b>Report Generated:</b><br/>{gen_time_str}<br/><font color='#64748b'>Confidential Diagnostic Record</font>", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[320, 220])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=14))

    # 2. Patient & Scan Overview (2 Columns Layout Table)
    patient_code = patient.get("patient_code", "N/A")
    full_name = patient.get("full_name", "N/A")
    age = patient.get("age", "N/A")
    gender = patient.get("gender", "N/A")
    patient_registered_at = patient.get("created_at", "")
    if patient_registered_at:
        try:
            reg_date_obj = datetime.fromisoformat(patient_registered_at.replace("Z", "+00:00"))
            patient_registered_at = reg_date_obj.strftime("%b %d, %Y")
        except Exception:
            pass

    scan_id = scan.get("id", "N/A")
    scan_created_at = scan.get("created_at", "")
    scan_date_str = scan_created_at
    if scan_created_at:
        try:
            scan_date_obj = datetime.fromisoformat(scan_created_at.replace("Z", "+00:00"))
            scan_date_str = scan_date_obj.strftime("%b %d, %Y at %I:%M %p UTC")
        except Exception:
            pass

    pred_class = scan.get("predicted_class", "N/A").upper()
    model_ver = scan.get("model_version", "EfficientNet-B0 (256x256 + TTA)")

    # Color badge for class
    badge_bg = "#ecfdf5" if pred_class == "NORMAL" else ("#e0f2fe" if pred_class == "BENIGN" else "#fef2f2")
    badge_fg = "#047857" if pred_class == "NORMAL" else ("#0369a1" if pred_class == "BENIGN" else "#b91c1c")

    patient_info_table_data = [
        [
            Paragraph("Patient ID Code:", cell_bold_style),
            Paragraph(f"<b>{patient_code}</b>", cell_value_style),
            Paragraph("Scan Record ID:", cell_bold_style),
            Paragraph(f"<code>{scan_id[:13]}...</code>", cell_value_style)
        ],
        [
            Paragraph("Full Name:", cell_bold_style),
            Paragraph(full_name, cell_value_style),
            Paragraph("Scan Timestamp:", cell_bold_style),
            Paragraph(scan_date_str, cell_value_style)
        ],
        [
            Paragraph("Age / Gender:", cell_bold_style),
            Paragraph(f"{age} yrs / {gender}", cell_value_style),
            Paragraph("Model Architecture:", cell_bold_style),
            Paragraph(model_ver, cell_value_style)
        ],
        [
            Paragraph("Patient Registered:", cell_bold_style),
            Paragraph(patient_registered_at, cell_value_style),
            Paragraph("AI Classification:", cell_bold_style),
            Paragraph(f"<font color='{badge_fg}'><b>{pred_class}</b></font>", cell_value_style)
        ]
    ]

    info_table = Table(patient_info_table_data, colWidths=[110, 160, 110, 160])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(Paragraph("Patient & Scan Profile", section_heading_style))
    story.append(info_table)
    story.append(Spacer(1, 14))

    # 3. Class Probabilities Table
    norm_p = float(scan.get("normal_probability", 0.0))
    benign_p = float(scan.get("benign_probability", 0.0))
    malig_p = float(scan.get("malignant_probability", 0.0))

    prob_table_data = [
        [
            Paragraph("<b>Diagnostic Class</b>", cell_bold_style),
            Paragraph("<b>Predicted Probability (%)</b>", cell_bold_style),
            Paragraph("<b>Confidence Visual</b>", cell_bold_style)
        ],
        [
            Paragraph("Normal", cell_value_style),
            Paragraph(f"<b>{norm_p * 100:.2f}%</b>", cell_value_style),
            Paragraph(f"<font color='#0284c7'>{'█' * int(norm_p * 20)}</font>", cell_value_style)
        ],
        [
            Paragraph("Benign", cell_value_style),
            Paragraph(f"<b>{benign_p * 100:.2f}%</b>", cell_value_style),
            Paragraph(f"<font color='#0284c7'>{'█' * int(benign_p * 20)}</font>", cell_value_style)
        ],
        [
            Paragraph("Malignant", cell_value_style),
            Paragraph(f"<b>{malig_p * 100:.2f}%</b>", cell_value_style),
            Paragraph(f"<font color='#ef4444'>{'█' * int(malig_p * 20)}</font>", cell_value_style)
        ]
    ]

    prob_table = Table(prob_table_data, colWidths=[140, 160, 240])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(Paragraph("AI Class Probabilities Breakdown", section_heading_style))
    story.append(prob_table)
    story.append(Spacer(1, 14))

    # 4. Images Section (Side-by-Side Dual Display)
    orig_flowable = _prepare_image_flowable(original_image_bytes)
    grad_flowable = _prepare_image_flowable(gradcam_image_bytes)

    img_table_data = [
        [
            Paragraph("<b>Original Ultrasound Scan</b>", ParagraphStyle('ImgHeader1', parent=cell_bold_style, textColor=colors.HexColor('#0284c7'))),
            Paragraph("<b>Grad-CAM Heatmap Explanation</b>", ParagraphStyle('ImgHeader2', parent=cell_bold_style, textColor=colors.HexColor('#0369a1')))
        ],
        [
            orig_flowable,
            grad_flowable
        ]
    ]

    img_table = Table(img_table_data, colWidths=[265, 265])
    img_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#334155')),
        ('INNERGRID', (0, 0), (-1, -1), 1, colors.HexColor('#1e293b')),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(Paragraph("Ultrasound & AI Explainability Heatmap", section_heading_style))
    story.append(img_table)
    story.append(Spacer(1, 14))

    # 5. Mandatory Medical Disclaimer Box
    disclaimer_text = (
        "<b>CLINICAL RESEARCH DISCLAIMER:</b> For research and educational use only. "
        "This AI output is not a medical diagnosis and must not replace evaluation by a qualified healthcare professional."
    )
    disclaimer_table_data = [[
        Paragraph(disclaimer_text, disclaimer_text_style)
    ]]

    disclaimer_table = Table(disclaimer_table_data, colWidths=[540])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#f87171')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))

    story.append(disclaimer_table)

    # Build PDF into memory buffer
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
