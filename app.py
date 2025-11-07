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
from googletrans import Translator   # pip install googletrans==4.0.0-rc1

# Mistral API key from Streamlit secrets
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    st.error("❌ MISTRAL_API_KEY not found in Streamlit secrets. Set it in the app settings.")
    st.stop()

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Proxy bypass for Mistral API
os.environ["NO_PROXY"] = "api.mistral.ai"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(page_title="OCR + Sanskrit Cleaner & Translator AI", layout="wide")
st.title("📖 OCR for Devanagari - Sanskrit Manuscripts + AI Cleaner + Multi-Language Translation")
st.write("Upload a Sanskrit manuscript image → OCR → Mistral AI cleans the text → Translate to Indic Languages + English.")

# Cache model loading
@st.cache_resource
def load_translation_models():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"🔄 Using device: {DEVICE} (Model loading may take 1-2 minutes on first run.)")
    
    # --------- Sanskrit -> Indian Languages Model ---------
    model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"  # Load directly from Hugging Face
    
    tokenizer_indic = AutoTokenizer.from_pretrained(model_name_indic, trust_remote_code=True)
    model_indic = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_indic,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    
    translator = Translator()
    return tokenizer_indic, model_indic, translator, DEVICE

def manual_preprocess_batch(input_sentences, src_lang, tgt_lang):
    """Manual preprocessing: Add language tags for IndicTrans2 models."""
    batch = []
    for sent in input_sentences:
        formatted = f"<{src_lang}> {sent} </s> <{tgt_lang}>"
        batch.append(formatted)
    return batch

def manual_postprocess_batch(generated_tokens, tgt_lang):
    """Manual postprocessing: Strip tags from output."""
    translations = []
    for tokens in generated_tokens:
        # Remove leading language tag <xxx> and any trailing </s>
        cleaned = re.sub(r'^<[^>]+>', '', tokens).strip()
        cleaned = re.sub(r'</s>', '', cleaned).strip()
        translations.append(cleaned)
    return translations

def preprocess_ocr_text(text: str) -> str:
    """Keep only Devanagari letters, spaces, and common Sanskrit punctuation."""
    return re.sub(r"[^\u0900-\u097F\s।॥]", "", text)

def call_mistral_cleaner(noisy_text: str) -> str:
    """Send OCR text to Mistral AI for Sanskrit cleaning."""
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral-medium",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
You are a Sanskrit language AI specialized in cleaning OCR text.

Rules:
1. Correct OCR errors (misrecognized characters, misplaced diacritics, missing or extra letters).
2. Remove Latin letters, numbers, and special symbols.
3. Correct spacing, punctuation, and grammar.
4. Preserve original Sanskrit meaning; do NOT add new words or translate.
5. Keep only Devanagari script output with correct Sanskrit grammar.
6. Preserve punctuation symbols like | and ॥.

Sample Input:
र ।श्रीगणेगा यनमध। ।भथरात्रिसृत्तं पीवष्ियाधित पर्प्री व्याक्षनिनविखव

Sample Output:
। श्रीगणेशाय नमः । भद्ररात्रिस्मृतं । पीयूषाधित प्रप्री व्याख्यानविख्यातम् ।

Input OCR Text:
{noisy_text}

Output only the cleaned Sanskrit text in Devanagari. Do not add explanations or translations.
"""
                        }
                    ]
                }
            ]
        }
        response = requests.post(MISTRAL_URL, headers=headers, json=payload, proxies={"http": "", "https": ""})
        response.raise_for_status()
        result = response.json()
        cleaned_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return cleaned_text.strip() if cleaned_text else "Error: No output from Mistral."
    except Exception as e:
        logger.error("Error calling Mistral AI: %s", e)
        return f"Error: {str(e)}"

def translate_sanskrit(cleaned_sanskrit: str, tokenizer_indic, model_indic, translator, DEVICE):
    """Translate cleaned Sanskrit to target languages and English using manual formatting."""
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

    for tgt_lang in target_langs:
        batch = manual_preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer_indic(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            generated_tokens = model_indic.generate(
                **inputs,
                max_length=2048,
                num_beams=5,
                num_return_sequences=1,
                use_cache=False
            )

        generated_tokens_decoded = tokenizer_indic.batch_decode(generated_tokens, skip_special_tokens=True)
        translations_indic = manual_postprocess_batch(generated_tokens_decoded, tgt_lang)
        trans_indic = translations_indic[0]

        english_trans = translator.translate(cleaned_sanskrit, src='sa', dest='en').text  # Translate from Sanskrit directly for better accuracy

        translations_dict[tgt_lang] = {
            'indic': trans_indic,
            'english': english_trans,
            'lang_name': lang_names[tgt_lang]
        }
    
    return translations_dict


# Streamlit App Flow
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "avif"])

if "cleaned_sanskrit" not in st.session_state:
    st.session_state.cleaned_sanskrit = ""
if "translations" not in st.session_state:
    st.session_state.translations = None

if uploaded_file is not None:
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
        st.image(Image.fromarray(final_text_only), caption="🧾 Cleaned Text Only", use_container_width=True)

    st.subheader("🔍 Extracted Text")
    reader = easyocr.Reader(['hi', 'mr', 'ne'], gpu=False)
    results = reader.readtext(image, detail=1, paragraph=True)

    extracted_text = " ".join([res[1] for res in results])

    if extracted_text.strip():
        st.success("✅ OCR Extraction Done")
        st.text_area("Extracted Text", extracted_text, height=200)

        noisy_text = preprocess_ocr_text(extracted_text)

        if st.button("Clean OCR Text with AI"):
            with st.spinner("Sending text to Mistral AI for cleaning..."):
                cleaned_sanskrit = call_mistral_cleaner(noisy_text)
                if cleaned_sanskrit.startswith("Error"):
                    st.error(cleaned_sanskrit)
                else:
                    st.session_state.cleaned_sanskrit = cleaned_sanskrit
                    st.session_state.translations = None

        if st.session_state.cleaned_sanskrit:
            st.subheader("✨ Cleaned Sanskrit Text")
            st.text_area("Cleaned Text", st.session_state.cleaned_sanskrit, height=200)

            if st.button("Translate to Indic Languages + English"):
                with st.spinner("Loading translation models and generating translations... (This may take 1-2 minutes on CPU)"):
                    tokenizer_indic, model_indic, translator, DEVICE = load_translation_models()
                    translations = translate_sanskrit(st.session_state.cleaned_sanskrit, tokenizer_indic, model_indic, translator, DEVICE)
                    st.session_state.translations = translations

        if st.session_state.translations:
            st.subheader("🌐 Translations")
            for tgt_lang, trans_dict in st.session_state.translations.items():
                lang_name = trans_dict['lang_name']
                st.write(f"--- {lang_name} ---")
                st.write(f"**Sanskrit:** {st.session_state.cleaned_sanskrit}")
                st.write(f"**{lang_name}:** {trans_dict['indic']}")
                st.write(f"**English:** {trans_dict['english']}")
                st.write("---")

    else:
        st.warning("⚠️ No text detected. Try uploading a clearer image.")
