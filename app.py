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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -------------------- ENV + LOGGING --------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")  # Use Streamlit secrets for cloud
if not MISTRAL_API_KEY:
    st.error("❌ MISTRAL_API_KEY not found in Streamlit secrets. Set it in app settings.")
    st.stop()

MISTRAL_AGENT_ID = st.secrets.get("MISTRAL_AGENT_ID")  # Your custom agent's ID
if not MISTRAL_AGENT_ID:
    st.error("❌ MISTRAL_AGENT_ID not found in Streamlit secrets. Create an agent in Mistral Console and add its ID.")
    st.stop()

MISTRAL_URL = "https://api.mistral.ai/v1/agents/completions"  # Endpoint for agents
os.environ["NO_PROXY"] = "api.mistral.ai"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="OCR + Sanskrit Cleaner & Translator AI", layout="wide")
st.title("📖 OCR for Devanagari - Sanskrit Manuscripts + AI Cleaner + Multi-Language Translation")
st.write("Upload a Sanskrit manuscript → OCR → Custom Mistral AI Agent cleans it → Translates into Indic languages + English (all via AI4Bharat IndicTrans2).")

# -------------------- TRANSLATION MODELS --------------------
@st.cache_resource
def load_translation_models():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Indic-Indic model (Sanskrit -> other Indic languages)
        model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"
        tokenizer_indic = AutoTokenizer.from_pretrained(model_name_indic, trust_remote_code=True)
        model_indic = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_indic,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        
        # Indic-En model (Sanskrit -> English; AI4Bharat only)
        model_name_en = "ai4bharat/indictrans2-indic-en-1B"
        tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, trust_remote_code=True)
        model_en = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_en,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        
    except Exception as e:
        st.error(f"❌ Model load failed: {e}. Check HF access or RAM.")
        raise
    return tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE

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
    """Keep only Devanagari letters, spaces, and Sanskrit punctuation."""
    return re.sub(r"[^\u0900-\u097F\s।॥]", "", text)

def call_mistral_cleaner(noisy_text: str) -> str:
    """Send OCR text to your custom Mistral AI Agent for Sanskrit cleaning."""
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "agent_id": MISTRAL_AGENT_ID,  # Use your agent's ID here
            "messages": [
                {
                    "role": "user",
                    "content": f"Clean this noisy OCR Sanskrit text: {noisy_text}\n\nOutput only the cleaned Devanagari text."  # Simplified—agent has full instructions baked in
                }
            ]
        }
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, proxies={"http": "", "https": ""})
        response.raise_for_status()
        result = response.json()
        cleaned_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return cleaned_text.strip() if cleaned_text else "Error: No output from Agent."
    except Exception as e:
        logger.error("Error calling Mistral Agent: %s", e)
        return f"Error: {str(e)}"

def translate_sanskrit(cleaned_sanskrit, tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE):
    """Translate Sanskrit into Indic languages and English using AI4Bharat IndicTrans2 only."""
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
    
    # English translation (using IndicTrans2 Indic-En model)
    tgt_lang_en = "eng_Latn"
    batch_en = manual_preprocess_batch(input_sentences, src_lang, tgt_lang_en)
    inputs_en = tokenizer_en(batch_en, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_en = model_en.generate(
            **inputs_en,
            max_length=512,  # Shorter for English
            num_beams=5,
            num_return_sequences=1,
            use_cache=False
        )
    generated_en_decoded = tokenizer_en.batch_decode(generated_en, skip_special_tokens=True)
    english_trans = manual_postprocess_batch(generated_en_decoded)[0]
    
    for tgt_lang in target_langs:
        batch = manual_preprocess_batch(input_sentences, src_lang, tgt_lang)
        inputs = tokenizer_indic(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_tokens = model_indic.generate(
                **inputs,
                max_length=2048,
                num_beams=5,
                num_return_sequences=1,
                use_cache=False
            )
        generated_decoded = tokenizer_indic.batch_decode(generated_tokens, skip_special_tokens=True)
        translations_indic = manual_postprocess_batch(generated_decoded)
        trans_indic = translations_indic[0]
        translations_dict[tgt_lang] = {
            "indic": trans_indic,
            "english": english_trans,  # Reuse across langs
            "lang_name": lang_names[tgt_lang]
        }
    return translations_dict

# -------------------- STREAMLIT APP --------------------
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
    reader = easyocr.Reader(['hi', 'mr', 'ne'], gpu=False)
    results = reader.readtext(image, detail=1, paragraph=True)
    extracted_text = " ".join([res[1] for res in results])

    if extracted_text.strip():
        st.success("✅ OCR Extraction Successful!")
        st.text_area("Extracted Text", extracted_text, height=200)
        noisy_text = preprocess_ocr_text(extracted_text)

        if st.button("✨ Clean OCR Text with Custom Mistral AI Agent"):
            with st.spinner("Cleaning Sanskrit text using your Mistral Agent..."):
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
                st.warning("⏳ Translation on CPU may take 1-2 min. Models load once (AI4Bharat IndicTrans2).")
                with st.spinner("Loading translation models and generating translations..."):
                    tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE = load_translation_models()
                    translations = translate_sanskrit(
                        st.session_state.cleaned_sanskrit,
                        tokenizer_indic, model_indic, tokenizer_en, model_en, DEVICE
                    )
                    st.session_state.translations = translations

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
