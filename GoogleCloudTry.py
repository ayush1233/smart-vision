import os
import base64
import tempfile
import requests
import cv2
import streamlit as st

# ──────────────────────────────────────────────────────────
# VERTEX AI CONFIGURATION  (only used if backend = Vertex AI)
# ──────────────────────────────────────────────────────────
VERTEX_CREDENTIALS = r"path/to/your-service-account.json"   # ← update
VERTEX_ENDPOINT    = "us-central1-aiplatform.googleapis.com"
VERTEX_REGION      = "us-central1"
VERTEX_PROJECT     = "your-gcp-project-id"                  # ← update
VERTEX_MODEL       = "meta/llama-3.2-90b-vision-instruct-maas"
GCS_BUCKET         = "your-gcs-bucket-name"                 # ← update

# ──────────────────────────────────────────────────────────
# OLLAMA CONFIGURATION  (only used if backend = Ollama)
# ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2-vision"   # alternatives: "llava", "llava:13b", "moondream"

# ──────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────
frame_contents = []


# ══════════════════════════════════════════════════════════
#  OLLAMA BACKEND
# ══════════════════════════════════════════════════════════

def ollama_call(image_path: str, prompt: str) -> str | None:
    """Send a frame to the local Ollama vision model and return the text response."""
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            st.error(f"Ollama error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`"
        )
        return None
    except Exception as e:
        st.error(f"Ollama error: {e}")
        return None


def ollama_summarize(all_frame_text: str) -> str | None:
    """Ask Ollama to summarise the collected frame descriptions."""
    prompt = (
        "Based on the following frame-by-frame descriptions of a video, "
        "write a concise, coherent summary:\n\n"
        + all_frame_text
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            st.error(f"Ollama summarise error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Ollama summarise error: {e}")
        return None


# ══════════════════════════════════════════════════════════
#  VERTEX AI BACKEND
# ══════════════════════════════════════════════════════════

def _vertex_credentials():
    """Return a refreshed Vertex AI token."""
    from google.auth.transport.requests import Request
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(
        VERTEX_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    return creds


def _upload_to_gcs(local_path: str) -> str | None:
    """Upload a file to GCS and return the gs:// URI."""
    try:
        from google.cloud import storage
        from google.oauth2.service_account import Credentials

        creds = Credentials.from_service_account_file(
            VERTEX_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = storage.Client(credentials=creds)
        bucket = client.bucket(GCS_BUCKET)
        blob   = bucket.blob(f"frames/{os.path.basename(local_path)}")
        blob.upload_from_filename(local_path)
        uri = f"gs://{bucket.name}/{blob.name}"
        return uri
    except Exception as e:
        st.error(f"GCS upload failed: {e}")
        return None


def vertex_call(image_path: str, prompt: str) -> str | None:
    """Upload frame to GCS then call LLaMA on Vertex AI."""
    gcs_uri = _upload_to_gcs(image_path)
    if not gcs_uri:
        return None

    creds = _vertex_credentials()
    url = (
        f"https://{VERTEX_ENDPOINT}/v1beta1/projects/{VERTEX_PROJECT}"
        f"/locations/{VERTEX_REGION}/endpoints/openapi/chat/completions"
    )
    payload = {
        "model": VERTEX_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": gcs_uri}},
                    {"type": "text",      "text": prompt},
                ],
            }
        ],
        "max_tokens": 150,
    }
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type":  "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False, timeout=120)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.error(f"Vertex AI error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Vertex AI error: {e}")
        return None


def vertex_summarize(all_frame_text: str) -> str | None:
    """Summarise collected frame descriptions via Vertex AI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
    summary_prompt = (
        "Based on the following extracted information from individual frames, "
        "provide a concise summary of the video:\n\n" + all_frame_text
    )
    tmp.write(summary_prompt)
    tmp.close()

    result = vertex_call(tmp.name, "Summarise the video based on this document.")
    os.unlink(tmp.name)
    return result


# ══════════════════════════════════════════════════════════
#  UNIFIED DISPATCH
# ══════════════════════════════════════════════════════════

def analyze_frame(image_path: str, prompt: str, backend: str) -> str | None:
    if backend == "🦙 Ollama (Local)":
        return ollama_call(image_path, prompt)
    else:
        return vertex_call(image_path, prompt)


def summarize_all(backend: str) -> str | None:
    combined = "\n".join(frame_contents)
    if backend == "🦙 Ollama (Local)":
        return ollama_summarize(combined)
    else:
        return vertex_summarize(combined)


# ══════════════════════════════════════════════════════════
#  VIDEO PROCESSING
# ══════════════════════════════════════════════════════════

def process_video(video_source, output_folder: str, interval: int, scenario_func, backend: str):
    """Extract frames and analyse each with the selected backend."""
    os.makedirs(output_folder, exist_ok=True)

    video = (
        cv2.VideoCapture(video_source)
        if isinstance(video_source, str)
        else video_source
    )

    fps            = video.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps * interval))
    frame_count    = 0
    saved_count    = 0

    progress_bar  = st.progress(0, text="Processing frames…")
    result_area   = st.empty()

    while True:
        ok, frame = video.read()
        if not ok:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

            prompt = scenario_func()
            result_area.info(f"🔍 Analysing frame {saved_count}…")
            content = analyze_frame(frame_path, prompt, backend)

            if content:
                frame_contents.append(content)
                result_area.success(f"**Frame {saved_count}:** {content}")

            progress_bar.progress(min(saved_count / 20, 1.0))

        frame_count += 1

    video.release()
    cv2.destroyAllWindows()
    progress_bar.empty()

    # ── Final summary ──
    if frame_contents:
        st.subheader("📝 Video Summary")
        with st.spinner("Generating summary…"):
            summary = summarize_all(backend)
        if summary:
            st.success(summary)


# ══════════════════════════════════════════════════════════
#  SCENARIO PROMPTS
# ══════════════════════════════════════════════════════════

SCENARIOS = {
    "📦 OCR — Extract product details":      "Extract brand name, pack size, and product type from the image.",
    "📅 Expiry date & MRP detection":         "Extract the expiry date and MRP visible in the image.",
    "🔍 Product counting by brand":           "Identify each brand visible and count how many items per brand.",
    "🥦 Freshness detection of produce":      "Assess the freshness of any fruits or vegetables. Estimate shelf life.",
}


# ══════════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════════

st.set_page_config(page_title="Smart Vision", page_icon="🧠", layout="wide")
st.title("🧠 Smart Vision")
st.caption("AI-powered video analysis — choose your backend below")

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    backend = st.radio(
        "**AI Backend**",
        ["🦙 Ollama (Local)", "☁️ Vertex AI (Cloud)"],
        help=(
            "Ollama: runs locally, no credentials needed. "
            "Vertex AI: uses Google Cloud + LLaMA 3.2 90B."
        ),
    )

    if backend == "🦙 Ollama (Local)":
        ollama_model_choice = st.selectbox(
            "Ollama model",
            ["llama3.2-vision", "llava", "llava:13b", "moondream"],
        )
        OLLAMA_MODEL = ollama_model_choice

        if st.button("🔌 Check Ollama connection"):
            try:
                r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
                if r.status_code == 200:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if ollama_model_choice in models:
                        st.success(f"✅ Connected — `{ollama_model_choice}` is ready")
                    else:
                        st.warning(
                            f"⚠️ Connected but `{ollama_model_choice}` not found.\n\n"
                            f"Run: `ollama pull {ollama_model_choice}`\n\n"
                            f"Available: {', '.join(models) or 'none'}"
                        )
                else:
                    st.error("Ollama returned an unexpected response.")
            except Exception:
                st.error("❌ Ollama not reachable. Start it with: `ollama serve`")
    else:
        st.info(
            "**Vertex AI** requires:\n"
            "- GCP service account JSON\n"
            "- GCS bucket\n"
            "- Update constants at top of `GoogleCloudTry.py`"
        )

    st.divider()
    interval = st.slider("Frame interval (seconds)", 1, 10, 2)

# ── Main area ────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    use_case = st.selectbox("**Analysis scenario**", list(SCENARIOS.keys()))

with col2:
    input_source = st.radio("**Input source**", ["Upload a video", "Use camera"])

scenario_prompt_fn = lambda: SCENARIOS[use_case]
output_folder = tempfile.mkdtemp()

st.divider()

if input_source == "Upload a video":
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if video_file and st.button("▶️ Start Analysis"):
        frame_contents.clear()
        tmp_video = os.path.join(output_folder, "input.mp4")
        with open(tmp_video, "wb") as f:
            f.write(video_file.read())
        st.video(tmp_video)
        process_video(tmp_video, output_folder, interval, scenario_prompt_fn, backend)

elif input_source == "Use camera":
    if st.button("▶️ Start Camera Analysis"):
        frame_contents.clear()
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            process_video(cam, output_folder, interval, scenario_prompt_fn, backend)
        else:
            st.error("❌ Could not open camera.")
