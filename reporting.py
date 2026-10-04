from __future__ import annotations

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as ReportImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf(row, assessment: dict, links: list[dict], analyst: str, decision: str, notes: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#17365D")))
    story = [
        Paragraph("AI-Assisted Listing Assessment Report", styles["ReportTitle"]),
        Paragraph("Decision-support output — not a finding of criminality or product illegitimacy", styles["Italic"]),
        Spacer(1, 8*mm),
    ]
    summary = [
        ["Case reference", f"DEMO-{row['listing_id']}"], ["Listing", row["title"]],
        ["Brand / category", f"{row['brand']} / {row['category']}"],
        ["Seller", f"{row['seller_name']} ({row['seller_id']})"],
        ["AI risk", f"{assessment['overall_risk']}/100 — {assessment['risk_level']}"],
        ["Human decision", decision], ["Analyst", analyst or "Not recorded"],
    ]
    table = Table(summary, colWidths=[42*mm, 120*mm])
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(0,-1),colors.HexColor("#EAF0F7")),("GRID",(0,0),(-1,-1),0.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("PADDING",(0,0),(-1,-1),6)]))
    story += [table, Spacer(1, 6*mm), Paragraph("Automated indicators", styles["Heading2"])]
    for reason in assessment["reasons"]:
        story.append(Paragraph(f"• {reason}", styles["BodyText"]))
    story += [Spacer(1, 4*mm), Paragraph("Model component scores", styles["Heading2"])]
    scores = [["Component", "Score"]] + [[k.replace("_risk", "").title(), f"{assessment[k]*100:.1f}%"] for k in ["visual_risk","text_risk","price_risk","behaviour_risk","network_risk","history_risk"]]
    score_table = Table(scores, colWidths=[80*mm, 45*mm])
    score_table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#17365D")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.5,colors.grey),("PADDING",(0,0),(-1,-1),5)]))
    story += [score_table, Spacer(1, 4*mm), Paragraph("Visual explanation", styles["Heading2"])]
    vision = assessment.get("vision_details", {})
    story.append(Paragraph(f"Backend: {vision.get('backend', 'Not recorded')}. Embedding similarity: {vision.get('embedding_similarity', 0):.1%}; structural similarity: {vision.get('structural_similarity', 0):.1%}; OCR similarity: {vision.get('ocr_similarity', 0):.1%}.", styles["BodyText"]))
    heatmap = vision.get("heatmap_path")
    if heatmap:
        story += [Spacer(1, 2*mm), ReportImage(heatmap, width=75*mm, height=75*mm)]
    story += [Spacer(1, 4*mm), Paragraph("Related entities", styles["Heading2"])]
    if links:
        for link in links:
            story.append(Paragraph(f"• Seller {link['seller_id']} via {link['shared_entity']}", styles["BodyText"]))
    else:
        story.append(Paragraph("No direct shared-identifier link detected in the available data.", styles["BodyText"]))
    story += [Spacer(1, 4*mm), Paragraph("Analyst notes", styles["Heading2"]), Paragraph(notes or "No notes recorded.", styles["BodyText"]), Spacer(1, 5*mm)]
    story.append(Paragraph("Required next step: A qualified human must verify the product and evidence before any enforcement or adverse action. Original evidence, source metadata, and chain-of-custody records must be retained separately.", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
