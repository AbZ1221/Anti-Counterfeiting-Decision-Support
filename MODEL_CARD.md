# Model Card: Multimodal Anti-Counterfeiting Decision Support

## Intended use

Triage authorised e-commerce or enforcement records for review by trained analysts. The system compares a listing image with a rights-holder-approved reference, analyses listing text and contextual risk signals, links shared identifiers, and prioritises records.

## Prohibited use

- Autonomous takedowns, seizures, denial of service, accusation, or criminal conclusions
- Facial recognition or identification of people
- Training on unlawfully collected or unverified personal data
- Treating model scores as legal proof of counterfeiting

## Vision inputs and outputs

Inputs: one listing image, one verified reference image, expected brand/package text, and optional trained logo-detector weights.

Outputs: visual anomaly score, embedding similarity, structural similarity, OCR similarity, image-quality score, detected regions, difference heatmap, backend/version, and explanation reasons.

## Validation requirements

Evaluate separately by brand, product version, camera type, resolution, lighting, viewing angle, geography, language, and marketplace. Report precision, recall, F1, PR-AUC, calibration error, false-positive rate, false-negative rate, and abstention/insufficient-evidence rate. Use a time- and seller-separated test set.

## Known limitations

Packaging redesigns, legitimate parallel imports, reflections, compression, occlusion, low resolution, adversarial image edits, and inaccurate reference images can produce misleading scores. OCR may fail on stylised or multilingual text. CLIP similarity is not an authenticity test. Heatmaps show pixel differences, not causal proof.

## Human oversight

Every alert requires examination of original evidence and, where appropriate, rights-holder, laboratory, customs, or legal verification. Record the analyst decision independently from the AI assessment and preserve the model version used.

