# Advanced Vision AI Anti-Counterfeiting Investigation Prototype

This is an explainable decision-support prototype. It prioritises suspicious e-commerce listings, links related entities, and creates investigator-ready PDF reports. It **does not determine guilt or prove that an item is counterfeit**. All high-risk results require human verification.

## Included capabilities

- Advanced reference-based computer vision with CLIP support
- Multi-scale packaging/layout comparison and anomaly heatmaps
- OCR consistency checking for brands, labels, batch codes, and packaging text
- Image-quality assessment to flag evidence that is too weak for confident analysis
- Custom YOLO training scaffold for logos, security marks, barcodes, and batch codes
- TF-IDF + Logistic Regression text-risk model
- Transparent price, seller-behaviour, history, and network indicators
- Entity linking using seller, phone, email, address, image hash, and payment reference
- Network graph and connected-account analysis
- Explainable 0–100 combined risk score
- Human review, notes, case status, and audit trail in SQLite
- Evidence-linked PDF report
- Synthetic sample dataset and automated tests

## Installation (Windows or macOS/Linux)

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate an environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. For the lightweight offline demonstration, install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For the full computer-vision stack, use:

```bash
pip install -r requirements-vision.txt
streamlit run app.py
```

The full stack uses CLIP image embeddings and Tesseract OCR when available. Install the Tesseract desktop engine separately and ensure its executable is on your system PATH. Model weights are downloaded by their official Python packages on first use; for restricted environments, pre-approve and store weights locally.

The browser opens at `http://localhost:8501`.

## Investigator workflow

1. Open **Alert queue** and select a listing.
2. Review the risk score and the reasons behind it.
3. Inspect linked accounts and shared identifiers.
4. Record a human decision: unreviewed, needs information, likely genuine, suspicious, or refer for expert verification.
5. Add factual notes and an analyst name.
6. Generate the PDF report. Predictions and human findings remain clearly separated.

## Computer-vision workflow

1. Put rights-holder-approved reference images in `data/reference_images/`.
2. Put captured marketplace images in `data/listing_images/`.
3. Add `image_path` and `reference_path` for every CSV record.
4. Configure the embedding model and fusion weights in `cv_config.json`.
5. Run the dashboard. The engine returns visual risk, embedding similarity, structural similarity, OCR similarity, image quality, and a heatmap.
6. Treat low-quality or missing-image results as insufficient evidence, never as proof of counterfeiting.

### Train a specialised detector

For high-quality logo/security-mark localisation, annotate authorised images in YOLO format and place them under:

```text
training/logo_dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Then review `training/logo_dataset.yaml` and run:

```bash
python tools/train_logo_detector.py
```

Do not enable the detector until it has been validated on unseen products, brands, devices, lighting conditions, packaging revisions, and geographic markets.

## Replace the sample data

Use the column structure in `data/listings.csv`. Labels are used only to train/evaluate the demo text model:

- `confirmed_genuine`
- `confirmed_counterfeit`
- `unverified`

Never label an item as confirmed counterfeit merely because it is inexpensive, reported, or removed from a platform. Use an authoritative outcome such as rights-holder verification, laboratory examination, seizure result, or legal finding.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Production limitations

This demonstration does not include secure authentication, encrypted evidence storage, production chain-of-custody controls, platform APIs, malware scanning, formal retention policies, or jurisdiction-specific privacy controls. Visual similarity is an investigative lead, not an authenticity determination. Before operational use, obtain legal approval, conduct privacy and security assessments, validate by brand and product version, calibrate thresholds, implement role-based access, and keep qualified humans responsible for all adverse decisions.
