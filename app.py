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
from deep_translator import GoogleTranslator
from IndicTransToolkit.processor import IndicProcessor # ← use the official pre/post processor
# -------------------- ENV + LOGGING --------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
MISTRAL_AGENT_ID = st.secrets.get("MISTRAL_AGENT_ID")
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not MISTRAL_API_KEY or not MISTRAL_AGENT_ID or not HF_TOKEN:
    st.error("❌ Missing required keys in Streamlit secrets. Please set HF_TOKEN, MISTRAL_API_KEY, and MISTRAL_AGENT_ID.")
    st.stop()
MISTRAL_URL = "https://api.mistral.ai/v1/agents/completions"
os.environ["NO_PROXY"] = "api.mistral.ai"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# (Optional) keep CPU threads modest on Spaces
try:
    torch.set_num_threads(4)
except Exception:
    pass
# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="OCR + Sanskrit Cleaner & Translator AI", layout="wide")
st.title("📖 OCR for Devanagari - Sanskrit Manuscripts + AI Cleaner + Multi-Language Translation")
st.write(
    "Upload a Sanskrit manuscript → OCR → Mistral AI cleans it → "
    "Translates into Indic languages using AI4Bharat IndicTrans2 (single model). English via fallback."
)
TARGET_LANGS = ["hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu"] # Hindi, Kannada, Tamil, Telugu
LANG_NAMES = {
    "hin_Deva": "Hindi",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
}
# -------------------- UTILITIES --------------------
def preprocess_ocr_text(text: str) -> str:
    """Keep only Devanagari letters, spaces, and Sanskrit punctuation."""
    text = re.sub(r"[^\u0900-\u097F\s।॥]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def sanitize_for_processor(text: str) -> str:
    """Remove angle brackets and trailing dandas from the *content* (toolkit handles tags)."""
    text = text.replace("<", "").replace(">", "")
    text = re.sub(r"[।॥]+\s*$", "", text).strip()
    return text
def split_sanskrit_verses(text: str) -> list:
    """Split Sanskrit text into verses/sentences using । and ॥ as delimiters.
    Preserves the punctuation for re-joining.
    """
    # Split on । or ॥, but keep the delimiter with each chunk (except possibly last)
    parts = re.split(r'([।॥])', text.strip())
    verses = []
    current_verse = ""
    for part in parts:
        if part in ['।', '॥']:
            current_verse += part + " " # Add space after punctuation for natural flow
            verses.append(current_verse.strip())
            current_verse = ""
        else:
            current_verse += part
    if current_verse.strip(): # Add any trailing text
        verses.append(current_verse.strip())
    return [v.strip() for v in verses if v.strip()]
def call_mistral_cleaner(noisy_text: str, max_retries=3) -> str:
    """Clean OCR Sanskrit text via your Mistral Agent."""
    instructions = """You are an AI agent specialized in cleaning, correcting, and restoring Sanskrit text extracted from OCR (Optical Character Recognition) outputs.
Your job is to transform noisy, imperfect, or partially garbled Sanskrit text into a clean, readable, and grammatically correct Sanskrit version written only in Devanagari script.
OBJECTIVE
Correct OCR-induced spelling errors, misrecognized characters, and misplaced diacritics.
Preserve the original Sanskrit meaning and structure.
Maintain the Devanagari script output—never use transliteration or translation.
Output only the corrected Sanskrit text, with no explanations or extra commentary.
RULES
Do not translate Sanskrit text into any other language.
Do not add new words.
Fix errors like:
- Missing or extra characters
- Wrong vowel marks
- Garbled words
- Latin characters
- Bad spacing
Keep Sanskrit grammar intact.
Preserve punctuation symbols like । and ॥.
OUTPUT FORMAT
Only output the cleaned Sanskrit text.
No explanation. No formatting. No English.
Sample Input:
"र) ।श्रीगणेगा यनमध। ।भथरात्रिसृत्तं ) )) पीवष्ियाधित पर्प्री व्याक्षनिनविखव ) )"
Sample Output:
"। श्रीगणेशाय नमः । भद्ररात्रिस्मृतं । पीयूषाधित प्रप्री व्याख्यानविख्यातम् ।" """
    for attempt in range(max_retries):
        try:
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "agent_id": MISTRAL_AGENT_ID,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Clean this noisy OCR Sanskrit text:\n{noisy_text}"}
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
# -------------------- CACHED LOADERS --------------------
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['hi', 'mr', 'ne'], gpu=False)
@st.cache_resource
def load_indic_model_and_tools():
    """
    Load ONLY the Indic→Indic model + IndicProcessor.
    This avoids the second Indic→English model and removes tag issues entirely.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    st.info("💪 Loading AI4Bharat IndicTrans2 (Indic→Indic) with IndicProcessor...")
    model_name_indic = "ai4bharat/indictrans2-indic-indic-1B"
    tokenizer_indic = AutoTokenizer.from_pretrained(
        model_name_indic, token=HF_TOKEN, trust_remote_code=True
    )
    model_indic = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_indic,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    # Official pre/post processor for tags & normalization
    ip = IndicProcessor(inference=True)
    # English fallback translator (very light)
    translator = GoogleTranslator(source="auto", target="en")
    st.success(f"✅ IndicTrans2 (Indic→Indic) loaded on {DEVICE.upper()}.")
    return tokenizer_indic, model_indic, ip, translator, DEVICE
# -------------------- TRANSLATION --------------------
def translate_sanskrit_indic_only(cleaned_sanskrit, tokenizer_indic, model_indic, ip, translator, DEVICE):
    """
    Translate Sanskrit → {Hindi, Kannada, Tamil, Telugu} using ONLY the Indic→Indic model.
    English is produced by translating the Indic output via deep-translator (lightweight).
    """
    try:
        src_lang = "san_Deva"
        input_text = sanitize_for_processor(cleaned_sanskrit)
       
        # NEW: Split into verses for better handling
        input_verses = split_sanskrit_verses(input_text)
        st.info(f"📝 Split into {len(input_verses)} verses for accurate translation.")
       
        translations_dict = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = len(TARGET_LANGS)
       
        for i, tgt_lang in enumerate(TARGET_LANGS):
            status_text.text(f"Translating Sanskrit → {LANG_NAMES[tgt_lang]}...")
            tgt_translations = [] # Collect per-verse translations
           
            for verse in input_verses:
                input_sentences = [verse] # One verse per batch
                batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
                inputs = tokenizer_indic(
                    batch, truncation=True, padding="longest", return_tensors="pt"
                ).to(DEVICE)
               
                with torch.no_grad():
                    generated = model_indic.generate(
                        **inputs,
                        max_new_tokens=2048, # NEW: Focus on output length (tune if needed)
                        num_beams=4, # NEW: Beam search for coherence (was 1)
                        early_stopping=True, # NEW: Stop when done
                        use_cache=False, # NEW: Enable for speed (was False)
                        do_sample=False, # NEW: Deterministic with beams
                        length_penalty=1.0 # NEW: Balanced length
                    )
               
                decoded = tokenizer_indic.batch_decode(generated, skip_special_tokens=True)
                trans_indic_list = ip.postprocess_batch(decoded, lang=tgt_lang)
                verse_trans = trans_indic_list[0].strip() if trans_indic_list else ""
                tgt_translations.append(verse_trans)
           
            # Join verses with newlines for readability
            full_trans_indic = "\n".join(tgt_translations)
           
            # English via lightweight translator from the full Indic output
            try:
                english_trans = translator.translate(full_trans_indic) if full_trans_indic else translator.translate(input_text)
            except Exception:
                english_trans = ""
           
            translations_dict[tgt_lang] = {
                "indic": full_trans_indic,
                "english": english_trans,
                "lang_name": LANG_NAMES[tgt_lang],
            }
            progress_bar.progress((i + 1) / total_steps)
       
        status_text.text("Translation complete!")
        return translations_dict
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
            reader = get_easyocr_reader()
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
            if st.button("🌐 Translate to Indic Languages + English (1 model)"):
                st.warning("⏳ On CPU, first run may take a few minutes while the model loads (cached after).")
                with st.spinner("Loading IndicTrans2 (Indic→Indic) and translating..."):
                    try:
                        tokenizer_indic, model_indic, ip, translator, DEVICE = load_indic_model_and_tools()
                        translations = translate_sanskrit_indic_only(
                            st.session_state.cleaned_sanskrit,
                            tokenizer_indic, model_indic, ip, translator, DEVICE
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
                st.write(f"**English (from {trans_dict['lang_name']}):** {trans_dict['english']}")
                st.write("---")
    else:
        st.warning("⚠️ No text detected. Try uploading a clearer image.")
else:
    st.info("👆 Upload an image to start!")
