# Monument Information and Narration

A Streamlit application that helps users explore historical monuments from either a monument name or an uploaded image. It uses Mistral AI's `pixtral-large-latest` model to generate monument information, can translate the result from English to Kannada, and can narrate the text as audio in both languages.

## Features

- **Search by name:** Enter a monument's name to request historical information.
- **Analyze an image:** Upload a PNG, JPG, or JPEG image and ask the vision model to identify the monument and describe its history.
- **English results:** Display the generated monument information in the app.
- **Kannada translation:** Translate the English response into Kannada using `deep-translator`.
- **Audio narration:** Generate playable English and Kannada speech using Google Text-to-Speech (`gTTS`).
- **Image optimization:** Resize images to a maximum dimension of 800 × 800 pixels and compress them before sending them to the AI API.
- **Cached operations:** Use Streamlit caching for selected API, translation, and speech operations.

## How It Works

1. Choose **Enter Monument Name** or **Upload Monument Image**.
2. Submit the monument name or upload an image.
3. The application sends the input to Mistral's chat completions API using `pixtral-large-latest`.
4. Read the generated historical information in English.
5. Optionally generate English audio, translate the response into Kannada, and generate Kannada audio.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| User interface | Streamlit | Interactive web application |
| Image understanding and text generation | Mistral AI API — `pixtral-large-latest` | Analyze monument images and generate historical information |
| Image processing | Pillow (PIL) | Resize and compress uploaded images |
| API requests | Requests | Communicate with the Mistral API |
| Configuration | `python-dotenv` | Load the API key from a local `.env` file |
| Translation | `deep-translator` | Translate English text into Kannada |
| Text-to-speech | `gTTS` | Generate English and Kannada narration |

## Requirements

- Python 3.9 or newer
- A Mistral API key with access to the configured model
- An internet connection for Mistral, translation, and text-to-speech requests

## Installation and Setup

### 1. Get the project files

Clone this repository from GitHub or open a terminal in the directory that contains the project files.

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit requests python-dotenv deep-translator gTTS Pillow
```

### 4. Configure the Mistral API key

Create a file named `.env` in the project root, alongside the Python application script:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key
```

Replace the example value with your own key. The application checks for `MISTRAL_API_KEY` at startup and stops with an error if it is missing.

**Do not commit your `.env` file or publish your API key.** Add the following entries to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
```

### 5. Run the application

If your Streamlit script is named `app.py`, start it with:

```bash
streamlit run app.py
```

If the script has a different filename, use that filename in the command. Streamlit will show a local URL in the terminal, usually `http://localhost:8501`; open it in your browser.

## Usage

### Search by monument name

1. Select **Enter Monument Name**.
2. Enter the monument's name.
3. Click **Get Monument Information**.
4. Read the English information returned by the AI model.
5. Click **Generate English Narration** to play the English audio.
6. Click **Translate to Kannada** to show the Kannada translation.
7. Click **Generate Kannada Narration** to play the Kannada audio.

### Analyze a monument image

1. Select **Upload Monument Image**.
2. Upload a `.png`, `.jpg`, or `.jpeg` image.
3. Preview the image and click **Analyze Monument**.
4. Review the generated English information and use the narration or translation options as described above.

## Image Processing Details

- Accepted filename extensions: `.png`, `.jpg`, and `.jpeg`.
- Images are resized while preserving their aspect ratio, with a maximum width or height of 800 pixels.
- JPEG images are saved with quality set to 85; PNG images are optimized.
- The application rejects a processed image larger than 5 MB.

## Configuration

The principal settings are defined in the Python script:

| Setting | Value | Description |
|---|---|---|
| Mistral model | `pixtral-large-latest` | Model used for both name-based information generation and image analysis |
| Mistral endpoint | `https://api.mistral.ai/v1/chat/completions` | Chat completions API endpoint |
| Accepted image extensions | `png`, `jpg`, `jpeg` | Image formats accepted by the uploader |
| Maximum image dimension | `800` pixels | Maximum width or height after resizing |
| Maximum processed image size | `5 MB` | Image-size limit checked before the API request |
| Translation direction | English → Kannada | `GoogleTranslator(source='en', target='kn')` |
| Speech languages | English and Kannada | `gTTS` language codes `en` and `kn` |

## Limitations and Notes

- **AI-generated content may be inaccurate.** Monument identification, dates, names, and historical descriptions are generated by an AI model and are not independently verified by this application. Check important claims against reliable sources.
- **External services are required.** The Mistral API, `deep-translator`, and `gTTS` depend on network connectivity and may fail or be subject to provider limits or changes.
- **API access may incur costs.** Check your Mistral account's current plan, quotas, and pricing before using the application extensively.
- **Uploaded images are sent to an external AI service.** Do not upload private or sensitive images.
- **Audio is played in the app.** The current implementation does not provide a dedicated audio-download button.
- **The app is not a source-verification tool.** It does not attach historical references or citations to the generated descriptions.

## Security Notes

- Store `MISTRAL_API_KEY` in `.env`; never hard-code it in the Python script.
- Keep `.env` out of version control.
- Review logging before public deployment: the current code logs complete Mistral API response objects at INFO level.
- For a production deployment, consider adding request timeouts, more user-friendly error handling, and stricter validation of uploaded image content.

## License

No license is specified by the application code. Add a `LICENSE` file to the repository if you intend to define the terms under which others may use, modify, or distribute this project.
