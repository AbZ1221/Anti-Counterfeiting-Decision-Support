import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import assess_all, get_reviews, linked_sellers, save_review
from reporting import build_pdf

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Anti-Counterfeiting Decision Support", page_icon="🔎", layout="wide")


@st.cache_resource
def load_system():
    from core import load_data
    return assess_all(load_data())


df, graph, _ = load_system()
st.title("AI Anti-Counterfeiting Investigation Support")
st.warning("This prototype prioritises suspicious listings. It does not confirm counterfeiting or criminal conduct. Human verification is mandatory.")
queue, investigation, governance = st.tabs(["Alert queue", "Investigation workspace", "Governance & limitations"])

with queue:
    c1, c2 = st.columns(2)
    selected_levels = c1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    search = c2.text_input("Search listing, seller, or brand")
    filtered = df[df["risk_level"].isin(selected_levels)].copy()
    if search:
        mask = filtered[["listing_id", "title", "seller_name", "brand"]].astype(str).apply(
            lambda c: c.str.contains(search, case=False, regex=False)
        ).any(axis=1)
        filtered = filtered[mask]
    filtered = filtered.sort_values("overall_risk", ascending=False)
    st.dataframe(filtered[["listing_id", "title", "seller_name", "brand", "price", "market_price", "overall_risk", "risk_level"]], use_container_width=True, hide_index=True)
    st.caption(f"{len(filtered)} listing(s) shown. Thresholds are demonstration values and must be validated before operational use.")

with investigation:
    listing_id = st.selectbox("Select a listing", df.sort_values("overall_risk", ascending=False)["listing_id"].tolist())
    row = df[df["listing_id"] == listing_id].iloc[0]
    assessment = {k: row[k] for k in ["listing_id", "visual_risk", "vision_details", "text_risk", "price_risk", "behaviour_risk", "network_risk", "history_risk", "overall_risk", "risk_level", "reasons"]}
    links = linked_sellers(graph, row["seller_id"])
    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"{row['title']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("AI risk", f"{row['overall_risk']}/100")
        m2.metric("Level", row["risk_level"])
        m3.metric("Price gap", f"{max(0, 1-row['price']/row['market_price']):.0%}")
        st.write(f"**Seller:** {row['seller_name']} ({row['seller_id']})")
        st.write(f"**Description:** {row['description']}")
        st.subheader("Why it was prioritised")
        for reason in row["reasons"]:
            st.write(f"- {reason}")
        score_df = pd.DataFrame({
            "Signal": ["Vision", "Text", "Price", "Behaviour", "Network", "History"],
            "Score": [row.visual_risk, row.text_risk, row.price_risk, row.behaviour_risk, row.network_risk, row.history_risk],
        })
        st.bar_chart(score_df.set_index("Signal"))
    with right:
        st.subheader("Advanced visual examination")
        image_col, ref_col = st.columns(2)
        # use_column_width supports the Streamlit version on your Mac.
        image_col.image(str(ROOT / row["image_path"]), caption="Listing image", use_column_width=True)
        ref_col.image(str(ROOT / row["reference_path"]), caption="Verified reference", use_column_width=True)
        details = row["vision_details"]
        st.image(details["heatmap_path"], caption="Difference heatmap (red/yellow = stronger difference)", use_column_width=True)
        cv_metrics = pd.DataFrame({
            "CV signal": ["Embedding similarity", "Structural similarity", "OCR similarity", "Image quality"],
            "Score": [details["embedding_similarity"], details["structural_similarity"], details["ocr_similarity"], details["quality_score"]],
        })
        st.dataframe(cv_metrics, use_container_width=True, hide_index=True)
        st.caption(f"Vision backend: {details['backend']}")
        if details["detected_text"]:
            st.code(details["detected_text"], language=None)
        if details.get("object_detections"):
            st.write("Detected logos/security elements")
            st.dataframe(pd.DataFrame(details["object_detections"]), use_container_width=True, hide_index=True)
        st.subheader("Entity connections")
        if links:
            link_table = pd.DataFrame(links).rename(columns={"seller_id": "Related seller", "shared_entity": "Shared identifier"})
            st.dataframe(link_table, use_container_width=True, hide_index=True)
        else:
            st.info("No direct account connection found.")

    st.divider()
    st.subheader("Human review")
    analyst = st.text_input("Analyst name", value="Demo Analyst")
    decision = st.selectbox("Human decision", ["Unreviewed", "Needs more information", "Likely genuine", "Suspicious — continue investigation", "Refer for expert/rightsholder verification"])
    notes = st.text_area("Factual notes (avoid unsupported conclusions)")
    if st.button("Save review", type="primary"):
        save_review(listing_id, analyst, decision, notes, assessment)
        st.success("Human review and assessment snapshot saved to the audit trail.")
    reviews = get_reviews(listing_id)
    if not reviews.empty:
        st.dataframe(reviews, use_container_width=True, hide_index=True)
    pdf = build_pdf(row, assessment, links, analyst, decision, notes)
    st.download_button("Download investigator-ready PDF", pdf, file_name=f"assessment_{listing_id}.pdf", mime="application/pdf")
    with st.expander("Assessment JSON"):
        st.code(json.dumps(assessment, indent=2), language="json")

with governance:
    st.markdown("""
    ### Required controls before real deployment
    - Obtain lawful access to data and complete privacy, security, and legal assessments.
    - Preserve original evidence, timestamps, hashes, provenance, and chain of custody.
    - Train only on authoritative outcomes and test for sampling, geographic, and category bias.
    - Calibrate thresholds against investigator capacity and monitor false positives and model drift.
    - Implement authentication, role-based access, encryption, retention rules, and immutable audit logs.
    - Never allow a model or generated report to make an autonomous enforcement decision.

    ### Known prototype limitations
    The sample is synthetic and deliberately small. It uses reference-based vision with optional CLIP and OCR. The included images are demonstrations, not a validated counterfeit dataset. A custom logo/security-mark detector requires authorised images and human annotations. There are no live marketplace or customs integrations.
    """)
