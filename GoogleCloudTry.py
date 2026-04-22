import os
import streamlit as st
import cv2
import tempfile
from google.auth.transport.requests import Request
from google.cloud import storage
from google.oauth2.service_account import Credentials
import requests
from datetime import timedelta

# Google Cloud Configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"path/to/your-service-account.json"  # Update this path
ENDPOINT = "us-central1-aiplatform.googleapis.com"
REGION = "us-central1"
PROJECT_ID = "your-gcp-project-id"  # Update with your project ID
MODEL_NAME = "meta/llama-3.2-90b-vision-instruct-maas"
BUCKET_NAME = "your-gcs-bucket-name"  # Update with your GCS bucket name
frame_contents = []

def upload_image_to_gcs(local_path):
    """Uploads an image to Google Cloud Storage and returns its GCS URI."""
    try:
        credentials = Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"images/{os.path.basename(local_path)}")
        blob.upload_from_filename(local_path)
        gcs_uri = f"gs://{bucket.name}/{blob.name}"
        print(f"Image uploaded to GCS with URI: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        st.error(f"Failed to upload image to GCS: {str(e)}")
        return None

def call_llama3_vision_api(gcs_uri, prompt):
    """Calls the LLaMA 3.2 Vision API with a GCS URI and prompt."""
    try:
        credentials = Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(Request())
        url = f"https://{ENDPOINT}/v1beta1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
        payload = {
            "model": MODEL_NAME,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image_url": {"url": gcs_uri}, "type": "image_url"},
                        {"text": prompt, "type": "text"}
                    ]
                }
            ],
            "max_tokens": 100
        }
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, json=payload, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error calling LLaMA API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error during API call: {str(e)}")
        return None

def process_frame(gcs_uri, prompt):
    stii = st.empty()
    """Processes a single frame by calling the LLaMA API with a GCS URI and prompt."""
    response = call_llama3_vision_api(gcs_uri, prompt)
    if response:
        content = response.get("choices", [])[0].get("message", {}).get("content", "")
        stii.write(content)
        frame_contents.append(content)

def summarize_video_content():
    """Sends a combined prompt to the model to summarize all collected frame contents."""
    print('Summarization starts...')
    combined_content = "\n".join(frame_contents)
    summary_prompt = (
        "Based on the following extracted information from individual frames, provide a useful summary as if describing the contents of a video:\n\n"
        f"{combined_content}\n\n"
        "Summarize this information and provide only the most relevant details for the purpose of the selected scenario."
    )
    temp_filename = "summary_prompt.txt"
    with open(temp_filename, "w") as f:
        f.write(summary_prompt)
    gcs_uri = upload_image_to_gcs(temp_filename)
    if gcs_uri:
        try:
            summary_response = call_llama3_vision_api(gcs_uri, "Summarize the video contents based on this document.")
            if summary_response and "choices" in summary_response:
                summary_content = summary_response["choices"][0].get("message", {}).get("content", "")
                st.write("Video Summary:")
                st.write(summary_content)
            else:
                st.error("Failed to retrieve summary content from the LLaMA API response.")
                print("LLaMA API Response:", summary_response)
        except Exception as e:
            st.error(f"An error occurred while calling the LLaMA API: {str(e)}")
            print(f"LLaMA API call error: {str(e)}")
    else:
        st.error("Failed to upload the summary prompt to GCS for summarization.")
    os.remove(temp_filename)

def process_frames_from_video(video_source, output_folder, interval, scenario_func):
    """Processes frames from a video, uploads each to GCS, and processes them with the LLaMA API."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = cv2.VideoCapture(video_source) if isinstance(video_source, str) else video_source
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = 0
    saved_frame_count = 0
    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            gcs_uri = upload_image_to_gcs(frame_filename)
            if gcs_uri:
                prompt = scenario_func()
                process_frame(gcs_uri, prompt)
        frame_count += 1
    video.release()
    cv2.destroyAllWindows()
    summarize_video_content()

# Scenario Functions
def prompt_ocr_details():
    return "Extract brand name, pack size, and product type."

def prompt_expiry_date():
    return "Extract expiry date and MRP if available."

def prompt_image_recognition():
    return "Identify brand of each product and count for each brand."

def prompt_freshness_detection():
    return "Predict shelf life and detect freshness of produce."

# Streamlit Interface
st.title("Smart Vision")

output_folder = tempfile.mkdtemp()
input_option = st.radio("Choose input source", ("Upload a video", "Use Camera"))
use_case = st.selectbox(
    "Choose the use case scenario",
    [
        "OCR to extract details from image/label",
        "Using OCR to get expiry date details",
        "Image recognition and IR-based counting",
        "Detecting freshness of fresh produce"
    ]
)

scenario_func_map = {
    "OCR to extract details from image/label": prompt_ocr_details,
    "Using OCR to get expiry date details": prompt_expiry_date,
    "Image recognition and IR-based counting": prompt_image_recognition,
    "Detecting freshness of fresh produce": prompt_freshness_detection
}
scenario_func = scenario_func_map[use_case]

if input_option == "Upload a video":
    video_file = st.file_uploader("Upload your video file", type=["mp4", "mov", "avi"])
    if video_file:
        temp_video_path = os.path.join(output_folder, "temp_video.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        st.video(temp_video_path)
        st.text("Processing video...")
        process_frames_from_video(temp_video_path, output_folder, interval=2, scenario_func=scenario_func)
elif input_option == "Use Camera":
    st.text("Starting camera...")
    camera = cv2.VideoCapture(0)
    if camera.isOpened():
        st.text("Processing camera stream...")
        process_frames_from_video(camera, output_folder, interval=2, scenario_func=scenario_func)
    else:
        st.error("Could not open camera.")
