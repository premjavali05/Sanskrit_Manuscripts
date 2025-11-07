# app.py
import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import pillow_avif  # enables AVIF support for Pillow
import requests
import os
import logging
import re
import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -------------------- ENV + LOGGING --------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
MISTRAL_AGENT_ID = st.secrets.get("MISTRAL_AGENT_ID")

if not MISTRAL_API_KEY or not MISTRAL_AGENT_ID:
    st.error("❌ Missing MISTRAL_API_KEY or MISTRAL_AGENT_ID in Streamlit secrets.")
    st.stop()

MISTRAL_URL = "https://api.mistral.ai/v1/agents/completions"
os.environ["NO_PROXY"] = "api.mistral.ai"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="OCR + Sanskrit Cleaner & Translator AI", layout="wide")
st.title("📖 OCR for Devanagari - Sanskrit Manuscripts + AI Cleaner + Multi-Language Translation")
st.write(
    "Upload a Sanskrit manuscript → OCR → Custom Mistral AI Agent cleans it → "
    "Translates into Indic languages + English (via AI4Bharat IndicTrans2)."
)

use_light_models = st.sidebar.checkbox("Use Lighter Models (Recommended for Free Tier - Faster, Less RAM)", value=True)
st.sidebar.info("Full models need more memory. Lighter mode uses ~200MB vs ~4GB.")

# -------------------- TRANSLATION MODELS --------------------
@st.cache_resource
def load_translation_models():
    """Load AI4Bharat IndicTrans2 + English translation models with HF token support."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    hf_token = st.secrets.get("HF_TOKEN", None)
    if not hf_token:
        st.error("❌ Missing Hugging Face token (HF_TOKEN) in Streamlit secrets.")
        st.stop()

    try:
        if use_light_models:
            st.info("🟡 Using lighter configuration (CPU-friendly).")

            # IndicTrans2 Indic→Indic (1B model, float32 for lower memory)
            model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"
            tokenizer_indic = AutoTokenizer.from_pretrained(
                model_name_indic, token=hf_token, trust_remote_code=True
            )
            model_indic = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_indic,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)

            # English (Opus-MT - lightweight)
            model_name_en = "Helsinki-NLP/opus-mt-san-en"
            tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
            model_en = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_en, low_cpu_mem_usage=True
            ).to(DEVICE)

        else:
            st.info("💪 Using full IndicTrans2 models (needs higher RAM).")

            # Indic→Indic
            model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"
            tokenizer_indic = AutoTokenizer.from_pretrained(
                model_name_indic, token=hf_token, trust_remote_code=True
            )
            model_indic = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_indic,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)

            # Indic→English
            model_name_en = "ai4bharat/indictrans2-indic-en-1B"
            tokenizer_en = AutoTokenizer.from_pretrained(
                model_name_en, token=hf_token, trust_remote_code=True
            )
            model_en = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_en,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            ).to(DEVICE)

        st.success(f"✅ Models loaded successfully on {DEVICE.upper()} "
                   f"(Light mode: {'Yes' if use_light_models else 'No'})")

        return tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE

    except torch.cuda.OutOfMemoryError:
        st.error("❌ Out of GPU memory. Use CPU or switch to light mode.")
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("❌ Out of RAM. Use light mode or upgrade Streamlit tier.")
        else:
            st.error(f"❌ Runtime error: {e}")
        raise
    except Exception as e:
        st.error(f"❌ Model load failed: {e}\nCheck your HF token or network access.")
        raise


# -------------------- HELPER FUNCTIONS --------------------
def manual_preprocess_batch(input_sentences, src_lang, tgt_lang):
    return [f"<{src_lang}> {sent.strip()} </s> <{tgt_lang}>" for sent in input_sentences]


def manual_postprocess_batch(generated_tokens):
    translations = []
    for tokens in generated_tokens:
        cleaned = re.sub(r'^<[^>]+>', '', tokens).strip()
        cleaned = re.sub(r'</s>', '', cleaned).strip()
        translations.append(cleaned)
    return translations


def preprocess_ocr_text(text: str) -> str:
    return re.sub(r"[^\u0900-\u097F\s।॥]", "", text)


def call_mistral_cleaner(noisy_text: str, max_retries=3) -> str:
    """Send OCR text to Mistral AI Agent for Sanskrit cleaning."""
    for attempt in range(max_retries):
        try:
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "agent_id": MISTRAL_AGENT_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Clean this noisy OCR Sanskrit text: {noisy_text}\n\nOutput only the cleaned Devanagari text."
                    }
                ]
            }
            response = requests.post(MISTRAL_URL, headers=headers, json=payload, proxies={"http": "", "https": ""})
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                st.warning(f"⏳ Rate limit hit. Retrying in {retry_after}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            result = response.json()
            cleaned_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return cleaned_text.strip() if cleaned_text else "Error: No output from Agent."
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                continue
            raise
        except Exception as e:
            logger.error("Error calling Mistral Agent: %s", e)
            return f"Error: {str(e)}"
    return "Error: Max retries exceeded. Try again later."


def translate_sanskrit(cleaned_sanskrit, tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE):
    """Translate Sanskrit into Indic and English using AI4Bharat + OpusMT."""
    try:
        src_lang = "san_Deva"
        target_langs = ["hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu"]
        lang_names = {
            "hin_Deva": "Hindi",
            "kan_Knda": "Kannada",
            "tam_Taml": "Tamil",
            "tel_Telu": "Telugu"
        }
        input_sentences = [cleaned_sanskrit]
        translations_dict = {}

        # English translation
        if use_light_models:
            inputs_en = tokenizer_en(input_sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                generated_en = model_en.generate(**inputs_en, max_length=512, num_beams=5, early_stopping=True)
            english_trans = tokenizer_en.decode(generated_en[0], skip_special_tokens=True).strip()
        else:
            tgt_lang_en = "eng_Latn"
            batch_en = manual_preprocess_batch(input_sentences, src_lang, tgt_lang_en)
            inputs_en = tokenizer_en(batch_en, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_en = model_en.generate(
                    **inputs_en, max_length=512, num_beams=5, num_return_sequences=1, use_cache=False
                )
            generated_en_decoded = tokenizer_en.batch_decode(generated_en, skip_special_tokens=True)
            english_trans = manual_postprocess_batch(generated_en_decoded)[0]

        # Indic translations
        for tgt_lang in target_langs:
            batch = manual_preprocess_batch(input_sentences, src_lang, tgt_lang)
            inputs = tokenizer_indic(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_tokens = model_indic.generate(
                    **inputs,
                    max_length=2048 if not use_light_models else 1024,
                    num_beams=5,
                    num_return_sequences=1,
                    use_cache=False
                )
            generated_decoded = tokenizer_indic.batch_decode(generated_tokens, skip_special_tokens=True)
            translations_indic = manual_postprocess_batch(generated_decoded)
            translations_dict[tgt_lang] = {
                "indic": translations_indic[0],
                "english": english_trans,
                "lang_name": lang_names[tgt_lang]
            }
        return translations_dict
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.error("❌ Translation OOM. Use light models or CPU mode.")
        else:
            st.error(f"❌ Translation error: {e}")
        raise


# -------------------- MAIN APP --------------------
uploaded_file = st.file_uploader("Upload a Sanskrit manuscript image", type=["png", "jpg", "jpeg", "avif"])

if "cleaned_sanskrit" not in st.session_state:
    st.session_state.cleaned_sanskrit = ""
if "translations" not in st.session_state:
    st.session_state.translations = None

if uploaded_file:
    pil_img = Image.open(uploaded_file)
    image = np.array(pil_img.convert("L"))
    inverted = cv2.bitwise_not(image)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_bg = np.ones_like(image) * 255
    final_text_only = cv2.bitwise_and(white_bg, white_bg, mask=mask)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="📷 Original Image", use_container_width=True)
    with col2:
        st.image(Image.fromarray(final_text_only), caption="🧾 Processed Text-Only Image", use_container_width=True)

    st.subheader("🔍 Extracted OCR Text")
    with st.spinner("Initializing EasyOCR..."):
        try:
            reader = easyocr.Reader(['hi', 'mr', 'ne'], gpu=False)
        except Exception as e:
            st.error(f"❌ EasyOCR initialization failed: {e}")
            st.stop()

    results = reader.readtext(image, detail=1, paragraph=True)
    extracted_text = " ".join([res[1] for res in results])

    if extracted_text.strip():
        st.success("✅ OCR Extraction Successful!")
        st.text_area("Extracted Text", extracted_text, height=200)
        noisy_text = preprocess_ocr_text(extracted_text)

        if st.button("✨ Clean OCR Text with Mistral AI Agent"):
            with st.spinner("Cleaning Sanskrit text..."):
                cleaned_sanskrit = call_mistral_cleaner(noisy_text)
                if cleaned_sanskrit.startswith("Error"):
                    st.error(cleaned_sanskrit)
                else:
                    st.session_state.cleaned_sanskrit = cleaned_sanskrit
                    st.session_state.translations = None

        if st.session_state.cleaned_sanskrit:
            st.subheader("📜 Cleaned Sanskrit Text")
            st.text_area("Cleaned Text", st.session_state.cleaned_sanskrit, height=200)

            if st.button("🌐 Translate to Indic Languages + English"):
                if use_light_models:
                    st.info("🟡 Using lighter models—faster but slightly less accurate.")
                st.warning("⏳ Translation on CPU: 30s–2 min. Models load once per session.")
                with st.spinner("Loading translation models and generating translations..."):
                    try:
                        tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE = load_translation_models()
                        translations = translate_sanskrit(
                            st.session_state.cleaned_sanskrit,
                            tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE
                        )
                        st.session_state.translations = translations
                    except Exception as e:
                        st.exception(e)

        if st.session_state.translations:
            st.subheader("🌍 Translations")
            for tgt_lang, trans_dict in st.session_state.translations.items():
                st.write(f"--- **{trans_dict['lang_name']}** ---")
                st.write(f"**Sanskrit:** {st.session_state.cleaned_sanskrit}")
                st.write(f"**{trans_dict['lang_name']}:** {trans_dict['indic']}")
                st.write(f"**English:** {trans_dict['english']}")
                st.write("---")
    else:
        st.warning("⚠️ No text detected. Try uploading a clearer image.")
else:
    st.info("👆 Upload an image to start!")
