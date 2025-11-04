# End-to-End Invoice Understanding System — Technical Report

# Summary

This document describes an end-to-end system for extracting structured information from invoices. The system combines a layout-aware transformer model with classical OCR and rule-based post-processing to provide robust extraction across a variety of invoice layouts and image qualities. The report describes the dataset, preprocessing steps, model training and evaluation, inference pipeline, deployment approach and steps to reproduce the work.

# System description

The system accepts a document file (PDF or image) and returns a JSON object containing extracted fields and a short raw-text preview. The service is exposed as a REST endpoint and is backed by a small web interface that allows users to upload a document and view the results. The core components are:

• Document ingestion and conversion: multipage PDFs are converted to images; image files are normalized for OCR.  
• OCR and text extraction: Tesseract extracts the text for fallback processing and for assisting the layout model.  
• Layout-aware model: LayoutLMv3 is fine-tuned for token-level classification on labeled invoice data.  
• Post-processing: regex-based and heuristic rules consolidate tokens into final fields (invoice number, date, company, totals).  
• API and UI: a FastAPI server serves an endpoint that runs the pipeline and returns JSON. A Streamlit front end lets users upload files and visualize results.

# Architecture (high level)

The pipeline follows a linear flow. A user uploads a document through the web UI or sends it to the REST endpoint. If the document is a PDF it is converted to image(s); the first page is used for extraction. The image is passed to the OCR engine to obtain raw text and to the processor that prepares inputs for the LayoutLMv3 model. The model predicts token-level labels, which are then post-processed by a rule layer to produce final fields. The server returns a JSON response containing the fields and a raw-text preview.

A simple textual diagram of the flow:

User upload → (PDF→image if needed) → OCR + image preprocessing → LayoutLMv3 processor → Model inference → Post-processing rules → JSON response → Frontend display
```


## Dataset

The training data used in fine-tuning consists of invoice images and their corresponding token-level annotations. Each training record contains the following items: image file name, sequence of words (tokens), bounding box coordinates aligned with tokens when available, and token-level labels indicating fields such as invoice number, invoice date, total amount and tax amount. The dataset was organized into training and validation splits with an 80/20 ratio. Data augmentations included simple synthetic variations: small rotations, brightness adjustments and JPEG compression to improve robustness to scanning quality.

Data format conventions:
- Images are stored under a common root with nested directories.  
- Labels are provided in JSON lines format where each line is a record containing file_name, words, boxes (optional), and entities or extracted_fields.  
- The annotation scheme is flat token-level labels, mapped to a small label set used during training.

---

## Preprocessing

Preprocessing is implemented in a modular script. Steps include:
- Locate image file for each JSON record using recursive lookup of the file name.
- Open the image and convert to RGB.
- Extract tokens: use tokens supplied in the JSON record where available; otherwise fall back to OCR tokens.
- Build bounding boxes or placeholders when boxes are not provided.
- Use the LayoutLMv3 processor to combine image, tokens and boxes into model-ready tensors with consistent truncation and padding to the model maximum sequence length.
- Convert tensors to serializable lists when building the dataset for the Trainer.

The preprocessing includes sanity checks:
- Skip records where the image cannot be found or tokens are missing.
- Ensure label arrays match token sequence length (pad or trim to match model input length).

---

## Model, training and configuration

The modeling choice was LayoutLMv3, initialized from `microsoft/layoutlmv3-base`. The model was adapted for token classification by setting a small label vocabulary matching the invoice fields of interest. Training specifics:

- Model: LayoutLMv3ForTokenClassification
- Processor: LayoutLMv3Processor (with OCR disabled during training if token annotations exist; enable OCR during inference as needed)
- Loss: CrossEntropyLoss (standard for token classification)
- Optimizer: AdamW
- Learning rate: 5e-5
- Batch size: 1 (per-device) due to memory constraints on the training environment
- Epochs: 3
- Checkpoints and final weights saved in `./models/layoutlmv3_finetuned/`

Training pipeline notes:
- Training was run locally using the MPS backend on macOS where available.  
- The code uses Hugging Face Trainer with `remove_unused_columns=False` to preserve required tensors.  
- Special care was taken to ensure labels were padded to model input length to prevent shape mismatches during loss computation.

---

## Evaluation

Evaluation was performed on the held-out validation split. Metrics of interest were field-level precision, recall, and F1 where a field is considered correctly extracted if the predicted token span matches the ground truth span for that field. In addition, token-level accuracy was recorded to track convergence.

Summary (example reporting):
- Training loss: 0.006 (observed at the last epoch).  
- Validation behavior was monitored using token-level and entity-level checks. (Include concrete numbers from your run if available; the training logs show per-run metrics and runtime.)

Note: For a full production deployment, a larger labeled validation set and cross-validation would be recommended.

---

## Inference and fallback behavior

Because invoices can have widely varying layout and image quality, the service includes a robust fallback path. During inference:
- The processor runs with `apply_ocr=True` to obtain tokens and boxes when OCR information is not supplied.  
- The model performs token classification and outputs label indices that are mapped to label names using model configuration.  
- A post-processing stage aggregates token-level labels into fields. If the model does not yield confident or any labels for a document, a fallback path runs classic OCR with Tesseract and a set of flexible pattern-based extractors (regular expressions and keyword heuristics) to find likely invoice fields or resume contact details.  
- The returned JSON always includes a preview of raw OCR text and the extracted entities, ensuring the API never returns an empty or uninformative response.

This design prioritizes a reliable user experience: even when the model is uncertain, the service provides a readable result a user can inspect and correct.

---

## REST API and frontend

The service exposes a single POST endpoint `/predict` that accepts a multipart form upload of a file. The endpoint reads the uploaded file, converts PDFs to images, runs the OCR and the inference pipeline, and returns a JSON object containing:
- status: success/error indicator
- file_name: the uploaded file name
- predictions: object with `raw_text` (preview) and `extracted_entities` (dictionary of field: value)

A minimal frontend implemented in Streamlit provides a drag-and-drop upload, an upload preview, a processing spinner, and a clean card view of extracted fields and the raw text preview. The frontend calls the API endpoint on the local server and renders the returned JSON in a human-readable format.

---

## Deployment approach

Local testing:
- Launch the API server with Uvicorn:
  `uvicorn src.api_app:app --reload`
- Launch the UI:
  `streamlit run ui/app_ui.py`

Production deployment considerations:
- Containerize the application using a Dockerfile that installs system dependencies (Poppler for PDF conversion, Tesseract for OCR) and Python dependencies from `requirements.txt`.
- Choose a hosting target such as Render, Railway or an EC2 instance. For Render, use the Docker deployment option and enable auto-deploys from the GitHub repository.
- Use process managers or worker-based servers such as Gunicorn with Uvicorn workers for production throughput.
- Configure CORS, HTTPS termination and a small persistent store for logs when moving to production.

---

# Reproducibility and file layout

The repository should include:
- `src/` with training scripts, preprocessing scripts, inference utilities and the FastAPI app.  
- `ui/` containing the Streamlit app and static assets.  
- `models/layoutlmv3_finetuned/` with model weights and processor/tokenizer files.  
- `requirements.txt` listing Python dependencies and a short README with exact run commands.  
- `report.md` (this file) and an optional `report.pdf` export for submission.

Commands to reproduce main steps:
- Install dependencies:
  `pip install -r requirements.txt`
- Train:
  `python src/stage6_layoutlmv3_training.py`
- Run API:
  `uvicorn src.api_app:app --reload`
- Run UI:
  `streamlit run ui/app_ui.py`

# Results and observations

The fine-tuned model showed good convergence on the labeled invoice data. The combined design with OCR fallback resulted in a practical system that prioritizes useful output for users. The fallback ensures that even documents that differ from training layouts yield usable structured information. During integration, common issues were dependency mismatches and token/label length alignment; these were addressed by careful tensor handling and dataset sanity checks.

## Limitations

There are practical limitations to note:
- The model was fine-tuned for invoices only; it is not expected to generalize to entirely different document types without additional labeled data.  
- OCR quality depends on input image resolution and contrast. Low-quality scans will degrade extraction quality.  
- The current architecture handles the first page of multi-page documents only. Extending to multi-page aggregation is straightforward but requires additional engineering.  
- Fine-grained field validation (for example cross-checking totals or tax calculations) is not yet implemented.

# Recommendations and future work

To move the system toward production-readiness:
- Expand training data to include a diversity of vendors and formats.  
- Label additional fields for broader coverage (for example vendor address, line items).  
- Add a confidence score computation and thresholding so downstream systems can route low-confidence results for manual review.  
- Introduce a lightweight persistence layer to save processed documents, outputs and corrections for continuous model improvement.  
- Automate multi-page aggregation and batch processing for bulk uploads.

# How to use and deliverables

Deliverables to include:
- The complete codebase with modular scripts for preprocessing, training, inference and deployment.  
- Trained model files and the final checkpoint stored in `models/layoutlmv3_finetuned/`.  
- A working REST API and a basic frontend for demo.  
- A README containing run instructions and a short reproduction guide.  
- The report file (this document) and a PDF export for submission.

## Contact
For questions about the repository or to request clarifications, contact:

Malhar Dave  
Flikt Technology Web Solutions
malhardavepc@gmail.com
