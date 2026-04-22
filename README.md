# 🧠 Smart Vision — AI-Powered Video Analysis with LLaMA 3.2 Vision on Vertex AI

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-red?logo=streamlit)](https://streamlit.io)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vertex%20AI-4285F4?logo=googlecloud)](https://cloud.google.com/vertex-ai)
[![Model](https://img.shields.io/badge/Model-LLaMA%203.2%2090B%20Vision-purple)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Smart Vision** is a Streamlit-based web application that leverages **Meta's LLaMA 3.2 90B Vision** model (via Google Cloud Vertex AI MaaS) to intelligently analyze video streams and images in real time. Upload a video or use your webcam — the app extracts frames, sends them to the vision model, and generates structured, scenario-specific results with a final AI-generated summary.

---

## ✨ Features

- 🎥 **Video Upload & Live Camera Support** — Process pre-recorded videos or live webcam feeds
- 🖼️ **Automated Frame Extraction** — Extracts frames at configurable intervals using OpenCV
- ☁️ **Google Cloud Storage Integration** — Uploads frames to GCS for secure Vision API access
- 🤖 **LLaMA 3.2 90B Vision (Vertex AI MaaS)** — State-of-the-art multimodal AI for frame understanding
- 📋 **4 Specialized Analysis Scenarios** — Tailored prompts for different real-world use cases
- 📝 **AI End-to-End Summary** — Aggregates all frame results into a comprehensive video summary

---

## 🎯 Use Case Scenarios

| Scenario | Description |
|---|---|
| 📦 **OCR — Product Details** | Extracts brand name, pack size, and product type from labels |
| 📅 **Expiry Date Extraction** | Reads expiry dates and MRP from packaging |
| 🔍 **Product Counting** | Identifies brands and counts items by brand using image recognition |
| 🥦 **Freshness Detection** | Predicts shelf life and freshness of fresh produce |

---

## 🏗️ Architecture

```
Video / Camera Input
        │
        ▼
 OpenCV Frame Extractor
  (every N seconds)
        │
        ▼
 Google Cloud Storage
   (frame upload)
        │
        ▼
 Vertex AI MaaS Endpoint
   LLaMA 3.2 90B Vision
        │
        ▼
 Per-Frame Description
        │
        ▼
 LLaMA Final Summarizer
        │
        ▼
 Streamlit UI Output
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ayush1233/smart-vision.git
cd smart-vision
```

### 2. Install Dependencies

```bash
pip install streamlit google-cloud-storage google-cloud-aiplatform google-auth opencv-python requests
```

Or install everything from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Set Up Google Cloud Credentials

This project uses a **Google Cloud Service Account** with access to:
- **Vertex AI API** (for LLaMA 3.2 Vision via MaaS)
- **Cloud Storage API** (for frame uploads)

#### Steps:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable **Vertex AI API** and **Cloud Storage API**
4. Go to **IAM & Admin → Service Accounts**
5. Create a service account with roles:
   - `Vertex AI User`
   - `Storage Object Admin`
6. Download the JSON key file
7. Set the environment variable:

```bash
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\your-service-account.json"

# Linux / macOS
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-service-account.json"
```

Or place the key file path directly in `GoogleCloudTry.py`:

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"path/to/your-service-account.json"
```

### 4. Configure the App

Open `GoogleCloudTry.py` and update these constants:

```python
ENDPOINT    = "us-central1-aiplatform.googleapis.com"   # Your Vertex AI endpoint region
REGION      = "us-central1"                              # Your GCP region
PROJECT_ID  = "your-gcp-project-id"                     # Your Google Cloud project ID
MODEL_NAME  = "meta/llama-3.2-90b-vision-instruct-maas" # LLaMA model on Vertex AI
BUCKET_NAME = "your-gcs-bucket-name"                    # Your GCS bucket for frame storage
```

### 5. Create a GCS Bucket

```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

### 6. Run the App

```bash
streamlit run GoogleCloudTry.py
```

Visit `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
smart-vision/
├── GoogleCloudTry.py     # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignored files (credentials, cache, etc.)
└── README.md             # Project documentation
```

---

## 🔧 Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `ENDPOINT` | Vertex AI regional endpoint | `us-central1-aiplatform.googleapis.com` |
| `REGION` | Google Cloud region | `us-central1` |
| `PROJECT_ID` | GCP project ID | *(your project)* |
| `MODEL_NAME` | LLaMA model identifier on Vertex AI | `meta/llama-3.2-90b-vision-instruct-maas` |
| `BUCKET_NAME` | GCS bucket for frame uploads | *(your bucket)* |
| `interval` | Frame extraction interval in seconds | `2` |
| `max_tokens` | Max tokens per LLaMA response | `100` |

---

## 🛡️ Security Notes

- **Never commit your service account JSON key** to version control
- The `.gitignore` file is configured to exclude all `*.json` credential files
- Use environment variables or secret managers (e.g. GCP Secret Manager) in production
- Consider rotating your service account keys regularly

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `opencv-python` | Video/camera frame extraction |
| `google-cloud-storage` | GCS frame upload |
| `google-cloud-aiplatform` | Vertex AI SDK |
| `google-auth` | Service account authentication |
| `requests` | REST API calls to Vertex AI |

---

## 🌐 Google Cloud Setup Reference

- [Enable Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)
- [LLaMA on Vertex AI MaaS Docs](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama)
- [Create a GCS Bucket](https://console.cloud.google.com/storage)
- [Service Account Setup](https://console.cloud.google.com/iam-admin/serviceaccounts)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Meta AI — LLaMA 3.2 Vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai)
- [Streamlit](https://streamlit.io)
