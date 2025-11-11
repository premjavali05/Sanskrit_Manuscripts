# app.py
import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import pillow_avif # enables AVIF support for Pillow
import requests
import os
import logging
import re
import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator # Replaces googletrans
# -------------------- ENV + LOGGING --------------------
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_AGENT_ID = os.environ.get("MISTRAL_AGENT_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not MISTRAL_API_KEY or not MISTRAL_AGENT_ID or not HF_TOKEN:
    st.error("❌ Missing required keys in environment variables. Check Render dashboard.")
    st.stop()
MISTRAL_URL = "https://api.mistral.ai/v1/agents/completions"
os.environ["NO_PROXY"] = "api.mistral.ai"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="OCR + Sanskrit Cleaner & Translator AI", layout="wide")
st.title("📖 OCR for Devanagari - Sanskrit Manuscripts + AI Cleaner + Multi-Language Translation")
st.write(
    "Upload a Sanskrit manuscript → OCR → Mistral AI cleans it → "
    "Translates into Indic languages + English using AI4Bharat IndicTrans2 models."
)
# -------------------- CONSTANTS --------------------
VALID_TAGS = [
    "asm_Beng", "ben_Beng", "guj_Gujr", "hin_Deva", "kan_Knda",
    "mal_Mlym", "mar_Deva", "nep_Deva", "ori_Orya", "pan_Guru",
    "san_Deva", "tam_Taml", "tel_Telu", "eng_Latn"
]
TARGET_LANGS = ["hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu"]
LANG_NAMES = {
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu"
}
# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_translation_models():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        st.info("💪 Loading AI4Bharat IndicTrans2 models (requires Hugging Face token)...")
        # Indic→Indic
        model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"
        tokenizer_indic = AutoTokenizer.from_pretrained(model_name_indic, token=HF_TOKEN, trust_remote_code=True)
        model_indic = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_indic,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        # Indic→English
        model_name_en = "ai4bharat/indictrans2-indic-en-1B"
        tokenizer_en = AutoTokenizer.from_pretrained(model_name_en, token=HF_TOKEN, trust_remote_code=True)
        model_en = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_en,
            token=HF_TOKEN,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE)
        st.success(f"✅ Models loaded successfully on {DEVICE.upper()}.")
        translator = GoogleTranslator(source="auto", target="en")
        return tokenizer_indic, model_indic, tokenizer_en, model_en, translator, DEVICE
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        raise
# -------------------- HELPERS --------------------
def sanitize_text_for_tags(text: str) -> str:
    """Clean Sanskrit text to remove unwanted symbols before tagging."""
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"[।॥]+$", "", text)
    text = text.strip()
    return text
def manual_preprocess_batch(input_sentences, src_lang: str, tgt_lang: str):
    """Format text for IndicTrans2 with space-separated tags (per official docs)."""
    assert src_lang in VALID_TAGS, f"Invalid source language tag: {src_lang}"
    assert tgt_lang in VALID_TAGS, f"Invalid target language tag: {tgt_lang}"
    cleaned_batch = []
    for sent in input_sentences:
        sent = sanitize_text_for_tags(sent)
        # Format: "san_Deva eng_Latn cleaned_text" (no < > or </s>)
        cleaned_batch.append(f"{src_lang} {tgt_lang} {sent.strip()}")
    return cleaned_batch
def manual_postprocess_batch(generated_tokens, tgt_lang: str = None):
    """Postprocess to remove leading tgt_lang tag from generated text."""
    translations = []
    for tokens in generated_tokens:
        cleaned = tokens.strip()
        # Remove leading tgt_lang (e.g., "eng_Latn translated_text" -> "translated_text")
        # Fallback to removing first word+space if tgt_lang unknown
        if tgt_lang:
            cleaned = re.sub(rf"^{re.escape(tgt_lang)}\s+", "", cleaned)
        else:
            cleaned = re.sub(r"^\S+\s+", "", cleaned)
        translations.append(cleaned)
    return translations
def preprocess_ocr_text(text: str) -> str:
    """Keep only Devanagari letters, spaces, and Sanskrit punctuation."""
    return re.sub(r"[^\u0900-\u097F\s।॥]", "", text)
def call_mistral_cleaner(noisy_text: str, max_retries=3) -> str:
    """Clean OCR Sanskrit text via Mistral Agent."""
    # Full instructions for the agent
    instructions = """You are an AI agent specialized in cleaning, correcting, and restoring Sanskrit text extracted from OCR (Optical Character Recognition) outputs.
Your job is to transform noisy, imperfect, or partially garbled Sanskrit text into a clean, readable, and grammatically correct Sanskrit version written only in Devanagari script.
OBJECTIVE
Correct OCR-induced spelling errors, misrecognized characters, and misplaced diacritics.
Preserve the original Sanskrit meaning and structure.
Maintain the Devanagari script output—never use transliteration or translation.
Output only the corrected Sanskrit text, with no explanations or extra commentary.
 RULES
Do not translate Sanskrit text into English or any other language.
Do not add new words, meanings, or interpretations.
Fix common OCR issues, such as:
Misplaced or missing characters (e.g., "भथरात्रिसृत्तं" → "भद्ररात्रिस्मृतं")
Incorrect visarga, anusvāra, or vowel marks
Extra or missing spaces
Garbled or duplicated words
Presence of Latin or special characters—remove them
Keep Sanskrit grammar intact while restoring readable structure.
Preserve Sanskrit punctuation symbols like "।" and "॥".
Normalize spaces and diacritics.
If uncertain about a specific fragment, reconstruct the closest grammatically valid Sanskrit phrase.
OUTPUT FORMAT
Output must contain only the cleaned Sanskrit text.
Use Devanagari script only.
Do not include any English words, explanations, or formatting.
Sample Input:
"र) ।श्रीगणेगा यनमध। ।भथरात्रिसृत्तं ) )) पीवष्ियाधित पर्प्री व्याक्षनिनविखव ) )"
Sample Output:
"। श्रीगणेशाय नमः । भद्ररात्रिस्मृतं । पीयूषाधित प्रप्री व्याख्यानविख्यातम् ।"
Your single responsibility is:
Take corrupted OCR Sanskrit text as input → Produce a clean, readable, and grammatically valid Sanskrit version in Devanagari script only."""
    for attempt in range(max_retries):
        try:
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "agent_id": MISTRAL_AGENT_ID,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Clean this noisy OCR Sanskrit text: {noisy_text}"}
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
        except Exception as e:
            logger.error("Error calling Mistral Agent: %s", e)
            return f"Error: {str(e)}"
    return "Error: Max retries exceeded."
# -------------------- TRANSLATION --------------------
def translate_sanskrit(cleaned_sanskrit, tokenizer_indic, model_indic, tokenizer_en, model_en, translator, DEVICE):
    """Translate Sanskrit → Indic + English using IndicTrans2 + fallback."""
    try:
        src_lang = "san_Deva"
        input_sentences = [sanitize_text_for_tags(cleaned_sanskrit)]
        translations_dict = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        # English translation
        status_text.text("Translating to English...")
        tgt_lang_en = "eng_Latn"
        batch_en = manual_preprocess_batch(input_sentences, src_lang, tgt_lang_en)
        inputs_en = tokenizer_en(batch_en, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_en = model_en.generate(
                **inputs_en,
                max_length=1024, # Reduced for speed
                num_beams=3, # Reduced for speed (still accurate)
                num_return_sequences=1,
                use_cache=True # Enabled for speedup
            )
        english_raw = tokenizer_en.batch_decode(generated_en, skip_special_tokens=True)[0].strip()
        english_trans = manual_postprocess_batch([english_raw], tgt_lang_en)[0] # Remove leading tag
        if not english_trans:
            try:
                english_trans = translator.translate(cleaned_sanskrit)
            except Exception:
                english_trans = ""
        progress_bar.progress(0.2)
        # Indic translations
        for i, tgt_lang in enumerate(TARGET_LANGS):
            status_text.text(f"Translating to {LANG_NAMES[tgt_lang]}...")
            batch = manual_preprocess_batch(input_sentences, src_lang, tgt_lang)
            inputs = tokenizer_indic(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated_tokens = model_indic.generate(
                    **inputs,
                    max_length=1024, # Reduced for speed
                    num_beams=3, # Reduced for speed
                    num_return_sequences=1,
                    use_cache=True # Enabled for speedup
                )
            indic_raw = tokenizer_indic.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            trans_indic = manual_postprocess_batch([indic_raw], tgt_lang)[0] # Remove leading tag
            translations_dict[tgt_lang] = {
                "indic": trans_indic,
                "english": english_trans,
                "lang_name": LANG_NAMES[tgt_lang]
            }
            progress_bar.progress(0.2 + (i+1) * 0.8 / len(TARGET_LANGS))
        status_text.text("Translation complete!")
        progress_bar.progress(1.0)
        return translations_dict
    except AssertionError as e:
        st.error(f"❌ Language tag error: {e}. Check preprocessing & tags.")
        raise
    except Exception as e:
        st.error(f"❌ Translation failed: {e}")
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
            with st.spinner("Cleaning Sanskrit text using Mistral Agent..."):
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
                st.warning("⏳ Translation now faster: ~30-90s on GPU (enable in Colab Runtime > T4 GPU).")
                with st.spinner("Loading AI4Bharat models and generating translations..."):
                    try:
                        tokenizer_indic, model_indic, tokenizer_en, model_en, translator, DEVICE = load_translation_models()
                        translations = translate_sanskrit(
                            st.session_state.cleaned_sanskrit,
                            tokenizer_indic, model_indic, tokenizer_en, model_en, translator, DEVICE
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
