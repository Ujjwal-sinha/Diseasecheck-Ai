# XrayScan AI

**XrayScan AI** is an advanced AI-powered Streamlit application for automated X-ray disease detection, clinical report generation, and explainability. It leverages state-of-the-art vision-language models (BLIP), CNNs (DenseNet121), and LLMs (via LangChain) to provide comprehensive, evidence-based radiology analysis.

---

## Features

- **Automated Disease Detection:**  
  - Uses BLIP and DenseNet121 models to classify X-ray images into Pneumonia, Tuberculosis, Fracture, or Normal.
- **Vision-Language Analysis:**  
  - Generates natural language descriptions and clinical context for uploaded X-rays.
- **Explainability:**  
  - Visualizes model reasoning with Grad-CAM, SHAP, LIME, and edge detection overlays.
- **Clinical Report Generation:**  
  - Produces detailed, markdown-formatted reports with primary findings, recommendations, and patient education.
- **PDF Export:**  
  - Download comprehensive, formatted PDF reports including images and explainability visualizations.
- **User-Friendly UI:**  
  - Built with Streamlit for easy interaction and rapid feedback.

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── style.css
├── .env
├── .gitignore
└── Dataset/
    ├── Digital Knee X-ray Images.csv
    └── KneeXray/
        ├── MedicalExpert-I/
        │   └── MedicalExpert-I/
        │       ├── 0Normal/
        │       ├── 1Doubtful/
        │       ├── 2Mild/
        │       ├── 3Moderate/
        │       └── 4Severe/
        └── MedicalExpert-II/
            └── MedicalExpert-II/
                ├── 0Normal/
                ├── 1Doubtful/
                ├── 2Mild/
                ├── 3Moderate/
                └── 4Severe/
```

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/Ujjwal-sinha/MedicheckAi.git
cd MedicheckAi
```

### 2. Install Dependencies

Create a virtual environment (recommended):

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```sh
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory with the following content (already present):

```
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

Replace the values with your actual API keys.

### 4. Download Pretrained Models

The app will automatically download BLIP and DenseNet121 weights on first run. Ensure you have a stable internet connection.

### 5. Run the Application

```sh
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload an X-ray Image:**  
   - Supported formats: JPG, JPEG, PNG.
2. **(Optional) Add Clinical Context:**  
   - Enter patient info, select relevant clinical factors, or suspected disease.
3. **Analyze:**  
   - Click "Analyze with XrayScan AI" to process the image and generate a report.
4. **Review Results:**  
   - View AI predictions, clinical notes, recommendations, and feature visualizations.
5. **Download PDF:**  
   - Click "Generate Comprehensive PDF Report" to download a detailed report.

---

## Dataset

- The `Dataset/` folder contains knee X-ray images categorized by severity and expert annotation.
- Not required for running the app, but useful for training or evaluation.

---

## Customization

- **Styling:**  
  - Modify `style.css` for custom UI appearance.
- **Model Classes:**  
  - Update the `classes` list in [`app.py`](app.py) to support additional diseases.
- **Prompt Engineering:**  
  - Edit `EFFICIENT_PROMPT_TEMPLATE` in [`app.py`](app.py) for different report styles.

---

## Requirements

See [requirements.txt](requirements.txt) for all dependencies, including:

- streamlit
- torch
- torchvision
- transformers
- langchain
- langchain_groq
- fpdf
- matplotlib
- opencv-python
- pillow
- numpy
- captum
- lime
- python-dotenv

---

## Security & Privacy

- **Disclaimer:**  
  This tool is for educational purposes only. It is not intended for clinical use. Always consult a qualified radiologist for medical diagnosis.
- **Data Privacy:**  
  Uploaded images are processed in-memory and not stored permanently.

---

## Author

Built with ❤️ by [Ujjwal Sinha](https://github.com/Ujjwal-sinha)

---

## License

This project is for educational and research purposes. See [LICENSE](LICENSE) if provided.

---

## Acknowledgements

- [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Captum](https://captum.ai/)
- [LIME](https://github.com/marcotcr/lime)

---