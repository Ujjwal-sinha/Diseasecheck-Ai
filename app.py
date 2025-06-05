import streamlit as st
from PIL import Image
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import torch
import torchvision.models as models
from torchvision import transforms
from fpdf import FPDF
import tempfile
import base64
import uuid
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import re
from captum.attr import GradientShap
from lime.lime_image import LimeImageExplainer
import torch.nn.functional as F

# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="XrayScan AI", layout="centered", page_icon="🔍")

# Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing in .env file. Please configure it.")
    st.stop()

# Cache BLIP models
@st.cache_resource
def load_blip_models():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
        return processor, model
    except Exception as e:
        st.error(f"Failed to load BLIP models: {e}")
        return None, None

processor, blip_model = load_blip_models()
if not processor or not blip_model:
    st.error("Critical error: BLIP models failed to load.")
    st.stop()

# Cache CNN model (DenseNet121)
@st.cache_resource
def load_cnn_model():
    try:
        model = models.densenet121(weights="IMAGENET1K_V1")
        model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Grayscale input
        model.classifier = torch.nn.Linear(model.classifier.in_features, 4)  # 4 classes
        model = model.to(device).eval()
        return model
    except Exception as e:
        st.error(f"Failed to load CNN model: {e}")
        return None

cnn_model = load_cnn_model()
if not cnn_model:
    st.error("Critical error: CNN model failed to load.")
    st.stop()

# Image transform for CNN
cnn_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# X-ray analysis
def analyze_xray(image: Image.Image, suspected_disease: str = None) -> dict:
    try:
        # Preprocess for BLIP
        image_blip = image.convert("L").resize((224, 224), Image.Resampling.LANCZOS)
        img_np = np.array(image_blip)
        img_np = cv2.equalizeHist(img_np)
        image_blip = Image.fromarray(img_np)

        # BLIP description
        inputs = processor(images=image_blip, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
        out = blip_model.generate(**inputs, max_length=150, num_beams=5)
        description = processor.decode(out[0], skip_special_tokens=True)

        # BLIP classification
        classes = ["Pneumonia", "Tuberculosis", "Fracture", "Normal"]
        prompt = f"Classify this X-ray image as one of: {', '.join(classes)}. Description: {description}"
        if suspected_disease and suspected_disease != "None":
            prompt += f" Suspected condition: {suspected_disease}."
        
        classification_inputs = processor(text=prompt, images=image_blip, return_tensors="pt").to(device)
        classification_out = blip_model.generate(**classification_inputs, max_length=50)
        blip_predicted_class = processor.decode(classification_out[0], skip_special_tokens=True).strip()
        blip_confidence = 0.9 if blip_predicted_class in classes else 0.5

        # CNN classification
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = cnn_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            cnn_predicted_idx = torch.argmax(probabilities, dim=1).item()
            cnn_confidence = probabilities[0, cnn_predicted_idx].item()
        cnn_predicted_class = classes[cnn_predicted_idx]

        torch.mps.empty_cache()
        return {
            "description": description,
            "blip_prediction": blip_predicted_class,
            "blip_confidence": blip_confidence,
            "cnn_prediction": cnn_predicted_class,
            "cnn_confidence": cnn_confidence,
            "image_tensor": image_tensor,
            "cnn_predicted_idx": cnn_predicted_idx
        }
    except Exception as e:
        st.error(f"X-ray analysis failed: {e}")
        return {}

# Grad-CAM implementation
def apply_gradcam(image_tensor, model, target_class):
    try:
        model.eval()
        image_tensor = image_tensor.clone().detach().requires_grad_(True).to(device)
        features = model.features(image_tensor)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        output = model.classifier(pooled.view(image_tensor.size(0), -1))
        
        model.zero_grad()
        output[0, target_class].backward()
        
        gradients = image_tensor.grad.detach()  # Detach gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # Avoid division by zero
        
        cam_np = cam.squeeze().detach().cpu().numpy()  # Detach and convert to NumPy
        image_np = image_tensor.squeeze().detach().cpu().numpy()  # Detach and convert to NumPy
        plt.imshow(image_np, cmap="gray", alpha=0.5)
        plt.imshow(cam_np, cmap="jet", alpha=0.5)
        gradcam_path = f"gradcam_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(gradcam_path, bbox_inches="tight")
        plt.close()
        return gradcam_path
    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None

# SHAP implementation
def apply_shap(image_tensor, model):
    try:
        model.eval()
        gradient_shap = GradientShap(model)
        baseline = torch.zeros_like(image_tensor).to(device)
        image_tensor = image_tensor.clone().detach().requires_grad_(True).to(device)
        attributions = gradient_shap.attribute(image_tensor, baselines=baseline, target=0)
        
        attr_np = attributions.squeeze().detach().cpu().numpy()  # Detach and convert to NumPy
        image_np = image_tensor.squeeze().detach().cpu().numpy()  # Detach and convert to NumPy
        plt.imshow(np.abs(attr_np), cmap="viridis", alpha=0.5)
        plt.imshow(image_np, cmap="gray", alpha=0.5)
        shap_path = f"shap_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()
        return shap_path
    except Exception as e:
        st.warning(f"SHAP failed: {e}")
        return None

# LIME implementation
def apply_lime(image, model, classes):
    try:
        explainer = LimeImageExplainer()
        def predict_fn(images):
            images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            images = images / 255.0
            with torch.no_grad():
                outputs = model(images)
            return F.softmax(outputs, dim=1).cpu().numpy()
        
        image_np = np.array(image.convert("L").resize((224, 224)))
        explanation = explainer.explain_instance(image_np, predict_fn, top_labels=2, num_samples=500)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True)
        plt.imshow(temp, cmap="gray")
        lime_path = f"lime_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(lime_path, bbox_inches="tight")
        plt.close()
        return lime_path
    except Exception as e:
        st.warning(f"LIME failed: {e}")
        return None

# Feature visualization (Edge detection)
def visualize_xray_features(image):
    try:
        img_np = np.array(image.convert("L").resize((224, 224)))
        edges = cv2.Canny(img_np, 100, 200)
        plt.imshow(edges, cmap="gray")
        edge_path = f"edge_output_{uuid.uuid4().hex}.png"
        plt.axis("off")
        plt.savefig(edge_path, bbox_inches="tight")
        plt.close()
        return edge_path
    except Exception as e:
        st.warning(f"Feature visualization failed: {e}")
        return None

# Prompt template
EFFICIENT_PROMPT_TEMPLATE = """
As a board-certified radiologist with 20+ years of experience, analyze this X-ray image with description: {caption}
Additional patient context: {context}

Generate a comprehensive disease analysis with:

1. **Primary Disease Identification** (Most likely 1-2 diseases)
   - Disease: {prediction}
   - Confidence Level: {confidence:.2f}
   - Key Radiographic Indicators: 

2. **Detailed Analysis**
   - Pathophysiology: (How the disease manifests in X-rays)
   - Differential Diagnosis: (Other possible conditions)

3. **Evidence-Based Recommendations**
   - Diagnostic Follow-Up: (e.g., CT, lab tests)
   - Treatment Options: 
   - Urgency Level: 

4. **Clinical Considerations**
   - Expected Timeline for Follow-Up
   - When to Refer to Specialist
   - Red Flag Symptoms

5. **Patient Education**
   - Explanation of Findings
   - Next Steps
   - Common Misconceptions

Format using markdown with clear headings. Be concise but thorough.
"""

# LangChain query
def query_langchain(description: str, predicted_class: str, confidence: float, user_context: str) -> str:
    try:
        chat = ChatGroq(
            temperature=0.3,
            model_name="llama3-70b-8192",
            groq_api_key=GROQ_API_KEY,
            request_timeout=120,
            max_tokens=2000
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI radiology specialist. Provide accurate, evidence-based analysis."),
            ("user", EFFICIENT_PROMPT_TEMPLATE)
        ])
        chain = prompt_template | chat | StrOutputParser()
        result = chain.invoke({
            "caption": description,
            "prediction": predicted_class,
            "confidence": confidence,
            "context": user_context or "None provided"
        })
        torch.mps.empty_cache()
        return result
    except Exception as e:
        st.error(f"LLM analysis failed: {e}")
        return ""

# PDF generation
class MedicalPDF(FPDF):
    def __init__(self, patient_info=""):
        super().__init__()
        self.patient_info = patient_info
        self.toc = []
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font('Helvetica', '', 12)
    
    def sanitize_text(self, text):
        replacements = {'\u2265': '>=', '\u2264': '<=', '\u2019': "'", '\u2013': '-'}
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        return text.encode('latin1', 'ignore').decode('latin1')
    
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(76, 175, 80)
            self.cell(0, 10, 'XrayScan AI Disease Report', 0, 1, 'C')
            self.line(10, 20, 200, 20)
            self.ln(10)
    
    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no() - 1} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'C')
    
    def cover_page(self):
        self.add_page()
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(76, 175, 80)
        self.cell(0, 20, 'XrayScan AI', 0, 1, 'C')
        self.set_font('Helvetica', '', 16)
        self.cell(0, 10, 'X-ray Disease Analysis Report', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', '', 12)
        if self.patient_info:
            self.multi_cell(0, 8, f'Patient Information:\n{self.sanitize_text(self.patient_info)}')
        self.ln(10)
        self.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(20)
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 8, 'For educational purposes only', 0, 1, 'C')
    
    def table_of_contents(self):
        self.add_page()
        self.toc.append(("Table of Contents", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Table of Contents', 0, 1, 'C')
        self.ln(10)
        self.set_font('Helvetica', '', 12)
        for title, page in self.toc:
            self.cell(0, 8, f'{self.sanitize_text(title)} {"." * (50 - len(title))} {page}', ln=1)
        self.ln(10)
    
    def add_image(self, image_path, width=180):
        try:
            self.image(image_path, x=(self.w - width)/2, w=width)
            self.ln(10)
        except Exception as e:
            st.error(f"Failed to add image to PDF: {e}")
    
    def add_section(self, title, body):
        self.toc.append((title, self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, self.sanitize_text(title), 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        lines = body.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- '):
                self.set_left_margin(15)
                self.multi_cell(0, 8, f'• {self.sanitize_text(line[2:])}')
            else:
                self.set_left_margin(10)
                self.multi_cell(0, 8, self.sanitize_text(line))
            self.ln(2)
        self.ln(5)
    
    def add_summary(self, report):
        self.add_page()
        self.toc.append(("Executive Summary", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(33, 150, 243)
        self.cell(0, 10, 'Executive Summary', 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        summary = "This report provides an AI-generated analysis of an X-ray image for disease detection. Key findings include:\n"
        try:
            if "1. **Primary Disease Identification**" in report:
                primary = report.split("2. **Detailed Analysis**")[0].split("1. **Primary Disease Identification**")[1]
                summary += f"- {self.sanitize_text(primary.strip()[:200])}...\n"
            if "3. **Evidence-Based Recommendations**" in report:
                recommendations = report.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0]
                summary += f"- Recommendations: {self.sanitize_text(recommendations.strip()[:200])}...\n"
            summary += "Refer to the detailed sections for comprehensive insights."
        except IndexError:
            summary += "- Unable to generate summary due to report structure."
        self.multi_cell(0, 8, self.sanitize_text(summary))
        self.ln(10)
    
    def add_explainability(self, edge_path, gradcam_path, shap_path, lime_path):
        self.add_page()
        self.toc.append(("Radiographic Features", self.page_no()))
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 10, 'Radiographic Feature Analysis', 0, 1)
        self.ln(5)
        self.set_font('Helvetica', '', 11)
        if edge_path:
            self.cell(0, 8, "Edge Detection (Key Features):", 0, 1)
            self.image(edge_path, w=90)
        if gradcam_path:
            self.cell(0, 8, "Grad-CAM (Class Activation):", 0, 1)
            self.image(gradcam_path, w=90)
        if shap_path:
            self.cell(0, 8, "SHAP (Pixel Importance):", 0, 1)
            self.image(shap_path, w=90)
        if lime_path:
            self.cell(0, 8, "LIME (Superpixel Importance):", 0, 1)
            self.image(lime_path, w=90)

# Gradient text
def gradient_text(text, color1, color2):
    return f"""
    <style>
    .gradient-text {{
        background: -webkit-linear-gradient(left, {color1}, {color2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }}
    </style>
    <div class="gradient-text">{text}</div>
    """

# UI setup
st.markdown(gradient_text("XrayScan AI", "#4CAF50", "#2196F3"), unsafe_allow_html=True)
st.markdown("### AI-Powered X-ray Disease Detection")
st.markdown("**Disclaimer**: This tool is for educational purposes only. Consult a radiologist for medical diagnosis.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    **XrayScan AI** uses vision-language and CNN models to analyze X-ray images for disease detection.
    - Processes grayscale X-rays
    - Generates detailed clinical reports
    - Provides explainability with Grad-CAM, SHAP, LIME, and edge detection
    """)
    st.divider()
    st.subheader("Input Guidance")
    st.markdown("Upload clear X-ray images (e.g., chest, bone) in JPG, JPEG, or PNG format.")

# Image input
col1, col2 = st.columns([3, 2])
with col1:
    img_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"], key="image_uploader")
    image = None
    if img_file:
        try:
            image = Image.open(img_file).convert("L")
        except Exception as e:
            st.warning("Invalid image file. Please upload a valid JPG, JPEG, or PNG file.")

with col2:
    if image:
        st.image(image, caption="Preview", use_column_width=True)

# Clinical context
with st.expander("➕ Additional Clinical Context"):
    user_context = st.text_area(
        "Patient Information",
        placeholder="Age, symptoms duration, medical history, current medications...",
        height=100
    )
    user_context = re.sub(r'[^\x00-\x7F<>{}]', '', user_context)  # Sanitize input
    clinical_factors = st.multiselect(
        "Relevant Factors",
        ["Smoking History", "Recent Trauma", "Chronic Cough", "Fever", "Immunocompromised"],
        placeholder="Select applicable conditions"
    )
    suspected_disease = st.selectbox(
        "Suspected Disease (if known)",
        ["None", "Pneumonia", "Tuberculosis", "Fracture", "Normal"],
        help="Select if the X-ray is related to a specific condition."
    )

# Analysis and report generation
col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze with XrayScan AI", type="primary", use_container_width=True, key="analyze_button"):
        if not image:
            st.warning("Please upload an X-ray image.")
            st.stop()

        with st.spinner("Processing X-ray with AI..."):
            progress_bar = st.progress(0)
            progress_bar.progress(20, text="Analyzing X-ray features")
            result = analyze_xray(image, suspected_disease)
            
            if not result:
                st.error("X-ray analysis failed.")
                st.stop()

            progress_bar.progress(40, text="Generating explainability visualizations")
            edge_path = visualize_xray_features(image)
            gradcam_path = apply_gradcam(result["image_tensor"], cnn_model, result["cnn_predicted_idx"])
            shap_path = apply_shap(result["image_tensor"], cnn_model)
            lime_path = apply_lime(image, cnn_model, ["Pneumonia", "Tuberculosis", "Fracture", "Normal"])
            
            progress_bar.progress(60, text="Correlating with clinical data")
            report = query_langchain(
                result["description"],
                result["cnn_prediction"],
                result["cnn_confidence"],
                user_context
            )
            
            progress_bar.progress(90, text="Formatting report")
            with st.container():
                st.subheader("🔬 X-ray Disease Analysis")
                st.write(f"**BLIP Prediction**: {result['blip_prediction']} (Confidence: {result['blip_confidence']:.2f})")
                st.write(f"**CNN Prediction**: {result['cnn_prediction']} (Confidence: {result['cnn_confidence']:.2f})")
                tab1, tab2, tab3, tab4 = st.tabs(["Primary Findings", "Recommendations", "Clinical Notes", "Features"])
                with tab1:
                    st.markdown(report.split("2. **Detailed Analysis**")[0])
                with tab2:
                    if "3. **Evidence-Based Recommendations**" in report:
                        st.markdown(report.split("3. **Evidence-Based Recommendations**")[1].split("4. **Clinical Considerations**")[0])
                with tab3:
                    if "4. **Clinical Considerations**" in report:
                        st.markdown(report.split("4. **Clinical Considerations**")[1])
                with tab4:
                    if edge_path:
                        st.image(edge_path, caption="Edge Detection: Key Radiographic Features")
                    if gradcam_path:
                        st.image(gradcam_path, caption="Grad-CAM: Class Activation Map")
                    if shap_path:
                        st.image(shap_path, caption="SHAP: Pixel Importance")
                    if lime_path:
                        st.image(lime_path, caption="LIME: Superpixel Importance")
            
            st.session_state.report_data = {
                "image": image,
                "report": report,
                "blip_prediction": result["blip_prediction"],
                "blip_confidence": result["blip_confidence"],
                "cnn_prediction": result["cnn_prediction"],
                "cnn_confidence": result["cnn_confidence"],
                "edge_path": edge_path,
                "gradcam_path": gradcam_path,
                "shap_path": shap_path,
                "lime_path": lime_path,
                "timestamp": datetime.now()
            }
            progress_bar.progress(100, text="Analysis complete")
            st.success("✓ Report generated")

with col2:
    if st.button("Reset", use_container_width=True, key="reset_button"):
        keys_to_clear = ['report_data']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        for file in glob.glob("*.png") + glob.glob("*.jpg"):
            try:
                os.remove(file)
            except:
                pass
        st.rerun()

# PDF report
if 'report_data' in st.session_state:
    st.divider()
    st.subheader("Report Options")
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True, key="pdf_button"):
        with st.spinner("Generating report..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                patient_info = MedicalPDF().sanitize_text(user_context or "Not provided")
                pdf = MedicalPDF(patient_info=patient_info)
                pdf.cover_page()
                pdf.add_summary(st.session_state.report_data["report"])
                pdf.table_of_contents()
                
                tmp_path = os.path.join(tmp_dir, f"image_{uuid.uuid4().hex}.jpg")
                st.session_state.report_data["image"].save(tmp_path, quality=90)
                pdf.add_image(tmp_path)
                
                report = st.session_state.report_data["report"]
                sections = [
                    ("Clinical Findings", report.split("2. **Detailed Analysis**")[0]),
                    ("Detailed Analysis", report.split("2. **Detailed Analysis**")[1].split("3. **Evidence-Based Recommendations**")[0] if "2. **Detailed Analysis**" in report else ""),
                    ("Treatment Plan", report.split("3. **Evidence-Based Recommendations**")[1] if "3. **Evidence-Based Recommendations**" in report else "")
                ]
                
                for title, body in sections:
                    if body.strip():
                        pdf.add_section(title, body)
                
                pdf.add_explainability(
                    st.session_state.report_data["edge_path"],
                    st.session_state.report_data["gradcam_path"],
                    st.session_state.report_data["shap_path"],
                    st.session_state.report_data["lime_path"]
                )
                
                pdf_output = pdf.output(dest="S").encode('latin1')
                b64 = base64.b64encode(pdf_output).decode('latin1')
                href = f'<a href="data:application/pdf;base64,{b64}" download="XrayScan_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf">📥 Download Your Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("PDF report generated successfully!")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with ❤️ by <b>Ujjwal Sinha</b> • "
    "<a href='https://github.com/Ujjwal-sinha' target='_blank'>GitHub</a></p>",
    unsafe_allow_html=True
)

# Memory cleanup
torch.mps.empty_cache()