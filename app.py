import streamlit as st
import io
import time
from typing import Optional
import logging
from main import LLMHandler
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import textwrap
import re

# Configure page
st.set_page_config(
    page_title="Smart Policy ThinkTank",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Indian Government Color Theme CSS
st.markdown("""
<style>
    /* Indian government color theme */
    :root {
        --saffron: #FF9933;
        --white: #FFFFFF;
        --green: #138808;
        --navy: #000080;
        --gold: #FFD700;
    }
    
    /* Main app background */
    .main .block-container {
        background: linear-gradient(135deg, #FFF8DC 0%, #F5F5DC 50%, #F0FFF0 100%);
        min-height: 100vh;
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background: linear-gradient(180deg, #FFF8DC 20%, #F5F5DC 50%, #F0FFF0 80%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF9933, #FFD700);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #138808, #32CD32);
        transform: translateY(-2px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F5F5F5;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF9933, #FFD700);
        color: white;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FFFFFF, #F5F5F5);
        border: 2px solid #FF9933;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #FF9933;
        border-radius: 10px;
        background-color: #FFFFFF;
    }
    
    /* Text areas */
    .stTextArea > div > div > textarea {
        border: 2px solid #FF9933;
        border-radius: 8px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #FF9933, #138808);
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #138808;
        color: white;
        border-radius: 8px;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #000080;
        color: white;
        border-radius: 8px;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #FFD700;
        color: #333333;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyAnalyzerApp:
    """Streamlit app for Government policy analysis"""
    
    def __init__(self):
        """Initialize the app"""
        self.initialize_session_state()
        self.initialize_llm()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_model' not in st.session_state:
            st.session_state.current_model = "qwen"
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    
    def initialize_llm(self):
        """Initialize LLM handler with error handling"""
        try:
            if 'llm_handler' not in st.session_state:
                st.session_state.llm_handler = LLMHandler(st.session_state.current_model)
        except Exception as e:
            st.error(f"❌ Failed to initialize LLM: {str(e)}")
            st.info("Please check your GROQ_API_KEY in the .env file")
            st.stop()
    
    def clean_text_for_pdf(self, text: str) -> str:
        """Clean text by removing markdown formatting and extra spaces"""
        
        # Remove markdown bold/italic formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'__([^_]+)__', r'\1', text)      # Remove __bold__
        text = re.sub(r'_([^_]+)_', r'\1', text)        # Remove _italic_
        
        # Remove markdown headers but keep the text
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Remove extra spaces and normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)             # Multiple spaces to single space
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Remove bullet points markdown
        text = re.sub(r'^[\s]*[-*+]\s*', '', text, flags=re.MULTILINE)
        
        # Remove numbered list markdown
        text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        return text.strip()

    def parse_policy_sections(self, text: str) -> dict:
        """Parse text into main policy sections"""
        sections = {}
        
        # Common policy section keywords
        section_keywords = [
            'objective', 'objectives', 'purpose', 'aim', 'goals',
            'implementation', 'execution', 'deployment', 'rollout',
            'rights', 'entitlements', 'benefits', 'provisions',
            'impact', 'effects', 'consequences', 'outcomes',
            'analysis', 'assessment', 'evaluation', 'review',
            'summary', 'overview', 'conclusion', 'findings',
            'recommendations', 'suggestions', 'proposals',
            'eligibility', 'criteria', 'requirements', 'conditions',
            'procedure', 'process', 'steps', 'methodology',
            'timeline', 'schedule', 'phases', 'milestones'
        ]
        
        # Split text into potential sections
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        current_section = "POLICY OVERVIEW"
        current_content = []
        
        for line in lines:
            # Check if line is likely a section header
            is_header = False
            line_lower = line.lower()
            
            # Header detection logic
            if (len(line.split()) <= 8 and 
                (any(keyword in line_lower for keyword in section_keywords) or
                 line.isupper() or
                 line.endswith(':') or
                 (len(line) < 100 and any(char in line for char in [':', '•', '-'])))):
                is_header = True
            
            if is_header and current_content:
                # Save previous section
                sections[current_section] = '\n'.join(current_content)
                current_section = line.replace(':', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def parse_subsections(self, content: str) -> dict:
        """Parse section content into subsections"""
        subsections = {}
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        current_subsection = ""
        current_content = []
        
        for line in lines:
            # Check if line is a subsection header
            is_subsection = (
                len(line.split()) <= 6 and
                (line.endswith(':') or 
                 line.isupper() or
                 any(word in line.lower() for word in ['key', 'main', 'important', 'critical', 'essential']))
            )
            
            if is_subsection and current_content:
                subsections[current_subsection] = '\n'.join(current_content)
                current_subsection = line.replace(':', '').strip()
                current_content = []
            elif is_subsection:
                current_subsection = line.replace(':', '').strip()
            else:
                current_content.append(line)
        
        # Add final subsection
        subsections[current_subsection] = '\n'.join(current_content)
        
        return subsections

    def extract_bullet_points(self, text: str) -> list:
        """Extract and format bullet points from text"""
        bullet_points = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Clean up existing bullet markers
            line_clean = re.sub(r'^[-•*➤►▪▫◦‣⁃]\s*', '', line)
            line_clean = re.sub(r'^\d+\.\s*', '', line_clean)
            
            # Split long sentences into bullet points
            if len(line_clean) > 20:  # Only process substantial content
                # Split on common separators that indicate separate points
                if any(separator in line_clean for separator in [';', ':', ',']):
                    # Split on semicolons first (strongest separator)
                    parts = line_clean.split(';')
                    for part in parts:
                        part = part.strip()
                        if len(part) > 10:  # Avoid very short fragments
                            bullet_points.append(part)
                else:
                    bullet_points.append(line_clean)
        
        return bullet_points

    def generate_executive_summary(self, text: str) -> list:
        """Generate executive summary bullet points"""
        summary_points = []
        
        # Extract key sentences based on common policy indicators
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        key_indicators = [
            'objective', 'aim', 'purpose', 'goal',
            'implement', 'establish', 'create', 'develop',
            'benefit', 'impact', 'effect', 'result',
            'citizen', 'public', 'people', 'individual',
            'policy', 'scheme', 'program', 'initiative'
        ]
        
        for sentence in sentences[:10]:  # Limit to top 10 sentences
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in key_indicators):
                summary_points.append(sentence.strip())
        
        # If no key sentences found, use first few sentences
        if not summary_points:
            summary_points = sentences[:5]
        
        return summary_points[:7]  # Limit to 7 key points

    def create_pdf_analysis(self, summary: str, source: str) -> bytes:
        """Create a professionally formatted PDF from the analysis with enhanced structure"""
        try:
            buffer = io.BytesIO()
            
            # Create document with Indian government themed styling
            doc = SimpleDocTemplate(
                buffer, 
                pagesize=A4,
                rightMargin=50,
                leftMargin=50,
                topMargin=60,
                bottomMargin=60
            )
            
            # Define enhanced styles with Indian government colors
            styles = getSampleStyleSheet()
            
            # Custom title style
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=26,
                spaceAfter=24,
                spaceBefore=12,
                textColor=HexColor('#000080'),  # Navy blue
                alignment=1,  # Center alignment
                fontName='Helvetica-Bold'
            )
            
            # Custom subtitle style
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=16,
                spaceAfter=16,
                spaceBefore=8,
                textColor=HexColor('#FF9933'),  # Saffron
                alignment=1,  # Center alignment
                fontName='Helvetica-Bold'
            )
            
            # Custom main heading style (for major sections)
            main_heading_style = ParagraphStyle(
                'MainHeading',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12,
                spaceBefore=20,
                textColor=HexColor('#000080'),  # Navy blue
                fontName='Helvetica-Bold',
                borderWidth=1,
                borderColor=HexColor('#FF9933'),
                borderPadding=8,
                backColor=HexColor('#F8F8FF')  # Light blue background
            )
            
            # Custom sub-heading style
            sub_heading_style = ParagraphStyle(
                'SubHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=12,
                textColor=HexColor('#138808'),  # Green
                fontName='Helvetica-Bold',
                leftIndent=10
            )
            
            # Custom body style
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=6,
                spaceBefore=2,
                textColor=HexColor('#000000'),
                fontName='Helvetica',
                leading=14,
                leftIndent=20
            )
            
            # Custom bullet point style
            bullet_style = ParagraphStyle(
                'BulletStyle',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=4,
                spaceBefore=2,
                textColor=HexColor('#000000'),
                fontName='Helvetica',
                leading=13,
                leftIndent=30,
                bulletIndent=10,
                bulletFontName='Symbol'
            )
            
            # Custom metadata style
            meta_style = ParagraphStyle(
                'MetaStyle',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=4,
                textColor=HexColor('#138808'),  # Green
                fontName='Helvetica-Oblique',
                alignment=1  # Center alignment
            )
            
            # Header style for document info
            header_style = ParagraphStyle(
                'HeaderStyle',
                parent=styles['Normal'],
                fontSize=10,
                spaceAfter=2,
                textColor=HexColor('#666666'),
                fontName='Helvetica',
                alignment=0
            )
            
            # Build document content
            story = []
            
            # Title and Header
            story.append(Paragraph("🇮🇳 GOVERNMENT OF INDIA", title_style))
            story.append(Paragraph("Smart Policy ThinkTank", subtitle_style))
            story.append(Paragraph("Comprehensive Policy Analysis Report", sub_heading_style))
            story.append(Spacer(1, 20))
            
            # Document Information Box
            story.append(Paragraph("DOCUMENT INFORMATION", main_heading_style))
            current_time = time.strftime("%B %d, %Y at %I:%M %p")
            story.append(Paragraph(f"<b>Generated:</b> {current_time}", header_style))
            story.append(Paragraph(f"<b>Source Document:</b> {source}", header_style))
            story.append(Paragraph(f"<b>Analysis Model:</b> {st.session_state.current_model.upper()}", header_style))
            story.append(Paragraph(f"<b>Report ID:</b> SPT-{int(time.time())}", header_style))
            story.append(Spacer(1, 20))
            
            # Clean and process the summary text
            clean_summary = self.clean_text_for_pdf(summary)
            
            # Enhanced text processing with better structure detection
            sections = self.parse_policy_sections(clean_summary)
            
            for section_title, section_content in sections.items():
                # Add main section heading
                story.append(Paragraph(section_title.upper(), main_heading_style))
                
                # Process section content
                subsections = self.parse_subsections(section_content)
                
                for subsection_title, subsection_content in subsections.items():
                    if subsection_title:
                        # Add subsection heading
                        story.append(Paragraph(subsection_title, sub_heading_style))
                    
                    # Process content into bullet points or paragraphs
                    bullet_points = self.extract_bullet_points(subsection_content)
                    
                    if bullet_points:
                        for point in bullet_points:
                            if point.strip():
                                story.append(Paragraph(f"• {point.strip()}", bullet_style))
                    else:
                        # Add as regular paragraph if no bullet points detected
                        paragraphs = [p.strip() for p in subsection_content.split('\n') if p.strip()]
                        for para in paragraphs:
                            if para:
                                story.append(Paragraph(para, body_style))
                    
                    story.append(Spacer(1, 8))
            
            # Summary section if not already included
            if not any('summary' in section.lower() for section in sections.keys()):
                story.append(Paragraph("EXECUTIVE SUMMARY", main_heading_style))
                summary_points = self.generate_executive_summary(clean_summary)
                for point in summary_points:
                    story.append(Paragraph(f"• {point}", bullet_style))
                story.append(Spacer(1, 15))
            
            # Footer section
            story.append(Spacer(1, 30))
            story.append(Paragraph("DISCLAIMER & LEGAL NOTICE", main_heading_style))
            
            disclaimer_points = [
                "This analysis is generated by AI and is intended for informational purposes only.",
                "Users should verify all information with official government sources.",
                "This report does not constitute legal advice or official government policy interpretation.",
                "For authoritative guidance, please consult relevant government departments and officials.",
                "The analysis may not reflect the most current policy updates or amendments."
            ]
            
            for point in disclaimer_points:
                story.append(Paragraph(f"• {point}", bullet_style))
            
            story.append(Spacer(1, 20))
            story.append(Paragraph("Generated by Smart Policy ThinkTank | Powered by Powerpuff Girls", meta_style))
            story.append(Paragraph("For technical support and feedback, contact the development team", meta_style))
            
            # Build PDF
            doc.build(story)
            
            # Get PDF data
            pdf_data = buffer.getvalue()
            buffer.close()
            
            return pdf_data
            
        except Exception as e:
            logger.error(f"Enhanced PDF creation error: {e}")
            raise e
    
    def render_sidebar(self):
        """Render sidebar with settings and info"""
        with st.sidebar:
            st.title("⚙️ Settings")
            
            # Model selection
            st.subheader("🤖 Model Selection")
            available_models = st.session_state.llm_handler.get_available_models()
            model_options = list(available_models.keys())
            
            selected_model = st.selectbox(
                "Choose Model:",
                options=model_options,
                index=model_options.index(st.session_state.current_model) if st.session_state.current_model in model_options else 0,
                help="Different models have varying capabilities and response times"
            )
            
            if selected_model != st.session_state.current_model:
                if st.session_state.llm_handler.change_model(selected_model):
                    st.session_state.current_model = selected_model
                    st.success(f"✅ Switched to {selected_model}")
            
            st.divider()
            
            # Analysis history
            st.subheader("📚 Analysis History")
            if st.session_state.analysis_history:
                for i, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
                    with st.expander(f"📄 {entry['source'][:30]}{'...' if len(entry['source']) > 30 else ''}"):
                        st.write(f"**Source:** {entry['source']}")
                        st.write(f"**Timestamp:** {entry['timestamp']}")
                        st.write(f"**Length:** {entry['length']:,} characters")
                        st.write(f"**Words:** {entry.get('word_count', 'N/A'):,}")
                        
                        # Show file type icon
                        source_lower = entry['source'].lower()
                        if 'pdf' in source_lower:
                            st.write("**Type:** 📄 PDF Document")
                        elif 'txt' in source_lower or 'file' in source_lower:
                            st.write("**Type:** 📝 Text File")
                        else:
                            st.write("**Type:** ✏️ Text Input")
            else:
                st.info("No analysis history yet")
            
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.success("History cleared!")
            
            st.divider()
            
            # Usage statistics
            if st.session_state.analysis_history:
                st.subheader("📊 Usage Stats")
                total_analyses = len(st.session_state.analysis_history)
                total_chars = sum(entry['length'] for entry in st.session_state.analysis_history)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Analyses", total_analyses)
                with col2:
                    st.metric("Characters Processed", f"{total_chars:,}")
            
            st.divider()
            
            # Info section
            st.subheader("ℹ️ About")
            st.markdown("""
            **Smart Policy ThinkTank**
            
            This tool helps you understand Government policies by:
            - 🎯 Extracting policy objectives
            - 🔄 Explaining the implementation 
            - ⚖️ Outlining citizen rights
            - ⚠️ Highlighting policy impacts
            
            **Supported Files:**
            - 📄 PDF documents
            - 📝 Text files (.txt)
            - ✏️ Direct text input
            
            **Tips for Best Results:**
            - Upload complete policy documents
            - Ensure PDFs contain readable text
            - For long documents, be patient during processing
            """)
    
    def render_main_content(self):
        """Render main content area"""
        st.markdown("<h1 style='color: #000080; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>📃Smart Policy ThinkTank</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: left; color: #138808;'>A smarter way to digest policies</h2>", unsafe_allow_html=True)
        st.markdown("Upload or paste a Government policy to get a comprehensive analysis")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["📄 File Upload", "📝 Text Input"])
        
        with tab1:
            self.render_file_upload()
        
        with tab2:
            self.render_text_input()
    
    def render_file_upload(self):
        """Render file upload section"""
        st.subheader("Upload Policy Document")
        
        # Get supported extensions from LLM handler
        supported_extensions = list(st.session_state.llm_handler.get_supported_extensions())
        supported_types = [ext.replace('.', '') for ext in supported_extensions]
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=supported_types,
            help=f"Supported formats: {', '.join(supported_types).upper()}"
        )
        
        if uploaded_file is not None:
            try:
                file_extension = f".{uploaded_file.name.split('.')[-1].lower()}"
                
                # Handle different file types
                if uploaded_file.type == "text/plain" or file_extension == ".txt":
                    content = uploaded_file.read().decode('utf-8')
                    self.display_file_info(uploaded_file, content)
                    
                    if st.button("🚀 Analyze Document", key="analyze_file"):
                        self.process_analysis(content, f"File: {uploaded_file.name}")
                        
                elif uploaded_file.type == "application/pdf" or file_extension == ".pdf":
                    self.handle_pdf_upload(uploaded_file)
                    
                else:
                    st.error(f"❌ Unsupported file type: {file_extension}")
                    st.info(f"Supported formats: {', '.join(supported_types).upper()}")
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                logger.error(f"File upload error: {e}")
    
    def handle_pdf_upload(self, uploaded_file):
        """Handle PDF file upload and processing"""
        st.info("📄 PDF file detected - extracting text...")
        
        try:
            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            
            # Show PDF info
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{len(pdf_bytes):,} bytes")
            
            # Extract text using LLM handler
            with st.spinner("🔄 Extracting text from PDF..."):
                try:
                    extracted_text = st.session_state.llm_handler.extract_text_from_pdf(pdf_bytes)
                    
                    # Show extraction results
                    st.success("✅ Text extraction successful!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Characters", f"{len(extracted_text):,}")
                    with col2:
                        st.metric("Words", f"{len(extracted_text.split()):,}")
                    with col3:
                        st.metric("Pages", "Multiple" if "[Page" in extracted_text else "1")
                    
                    # Show preview
                    with st.expander("📖 Text Preview (First 500 characters)"):
                        preview_text = extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")
                        st.text_area("Extracted text preview:", preview_text, height=150, disabled=True)
                    
                    # Analysis button
                    if st.button("🚀 Analyze PDF Document", key="analyze_pdf"):
                        self.process_analysis(extracted_text, f"PDF: {uploaded_file.name}")
                        
                except Exception as e:
                    st.error(f"❌ Failed to extract text from PDF: {str(e)}")
                    st.info("💡 **Possible reasons:**")
                    st.info("• PDF contains only images (needs OCR)")
                    st.info("• PDF is password protected")
                    st.info("• PDF file is corrupted")
                    st.info("• Complex PDF layout")
                    
        except Exception as e:
            st.error(f"❌ Error reading PDF file: {str(e)}")
    
    def render_text_input(self):
        """Render text input section"""
        st.subheader("Paste Policy Text")
        
        policy_text = st.text_area(
            "Paste your Government policy here:",
            height=300,
            placeholder="Copy and paste the Government policy text here...",
            help="Paste the full text of the Government policy you want to analyze"
        )
        
        if policy_text:
            self.display_text_info(policy_text)
            
            if st.button("🚀 Analyze Text", key="analyze_text"):
                self.process_analysis(policy_text, "Text Input")
    
    def display_file_info(self, file, content: str):
        """Display file information"""
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", file.name)
        with col2:
            st.metric("File Size", f"{len(content):,} chars")
        with col3:
            st.metric("Estimated Words", f"{len(content.split()):,}")
    
    def display_text_info(self, text: str):
        """Display text information"""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Character Count", f"{len(text):,}")
        with col2:
            st.metric("Estimated Words", f"{len(text.split()):,}")
    
    def process_analysis(self, content: str, source: str):
        """Process the policy analysis"""
        if not content.strip():
            st.error("❌ Please provide policy text to analyze")
            return
        
        if len(content) < 100:
            st.warning("⚠️ The text seems quite short. Are you sure this is a complete policy?")
        
        # Show processing state
        with st.spinner("🔄 Analyzing policy... This may take a moment."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Simulate progress with status updates
                status_text.text("📝 Preparing analysis...")
                for i in range(20):
                    time.sleep(0.05)
                    progress_bar.progress((i + 1) / 100)
                
                status_text.text("🤖 Processing with AI model...")
                for i in range(20, 80):
                    time.sleep(0.02)
                    progress_bar.progress((i + 1) / 100)
                
                # Perform analysis
                status_text.text("🔍 Generating insights...")
                summary = st.session_state.llm_handler.summarize_policy(content)
                
                # Complete progress
                for i in range(80, 100):
                    time.sleep(0.01)
                    progress_bar.progress((i + 1) / 100)
                
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)  # Brief pause to show completion
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                self.display_analysis_results(summary, content, source)
                
                # Add to history
                self.add_to_history(source, content, summary)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                
                # Show helpful error info
                if "API" in str(e).upper():
                    st.info("💡 **API Error Tips:**")
                    st.info("• Check your GROQ_API_KEY in .env file")
                    st.info("• Verify your API key is valid")
                    st.info("• Check your internet connection")
                elif "TOKEN" in str(e).upper() or "LENGTH" in str(e).upper():
                    st.info("💡 **Content Too Long:**")
                    st.info("• Try analyzing a shorter document")
                    st.info("• Split large documents into sections")
                else:
                    st.info("💡 **General Troubleshooting:**")
                    st.info("• Try again in a moment")
                    st.info("• Check the error details above")
                    st.info("• Try a different model in settings")
                
                logger.error(f"Analysis error: {e}")
    
    def display_analysis_results(self, summary: str, original_text: str, source: str):
        """Display analysis results"""
        st.success("✅ Analysis Complete!")
        
        # Create tabs for results
        result_tab1, result_tab2 = st.tabs(["📊 Analysis Results", "📄 Original Text"])
        
        with result_tab1:
            st.markdown("## 🔍 Policy Analysis")
            st.markdown(summary)
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.download_button(
                    "📥 Download TXT",
                    data=summary,
                    file_name=f"policy_analysis_{int(time.time())}.txt",
                    mime="text/plain"
                ):
                    st.success("TXT downloaded!")
            
            with col2:
                try:
                    pdf_data = self.create_pdf_analysis(summary, source)
                    if st.download_button(
                        "📄 Download PDF",
                        data=pdf_data,
                        file_name=f"policy_analysis_{int(time.time())}.pdf",
                        mime="application/pdf"
                    ):
                        st.success("PDF downloaded!")
                except Exception as e:
                    if st.button("📄 Download PDF"):
                        st.error(f"PDF generation failed: {str(e)}")
                        st.info("💡 Try installing: pip install reportlab")
            
            with col3:
                if st.button("📋 Copy to Clipboard"):
                    st.info("Use your browser's copy function on the analysis text above")
            
            with col4:
                if st.button("🔄 Analyze Again"):
                    st.rerun()
        
        with result_tab2:
            st.markdown("## 📄 Original Policy Text")
            st.text_area(
                "Original content:",
                value=original_text[:5000] + ("..." if len(original_text) > 5000 else ""),
                height=400,
                disabled=True
            )
    
    def add_to_history(self, source: str, content: str, summary: str):
        """Add analysis to history"""
        entry = {
            'source': source,
            'content': content[:500] + "..." if len(content) > 500 else content,
            'summary': summary,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'length': len(content),
            'word_count': len(content.split())
        }
        st.session_state.analysis_history.append(entry)
        
        # Keep only last 10 entries
        if len(st.session_state.analysis_history) > 10:
            st.session_state.analysis_history = st.session_state.analysis_history[-10:]
    
    def run(self):
        """Run the Streamlit app"""
        self.render_sidebar()
        self.render_main_content()
        
        # Footer
        st.divider()
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FF9933, #138808); 
                   color: white; padding: 1rem; border-radius: 10px; text-align: center;
                   box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            👩🏽‍💻 Powered by Powerpuff Girls ❤️| 
            ⚠️ <strong>Disclaimer:</strong> This tool provides general analysis only. 
            Consult Government officials for specific advice.
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the app"""
    try:
        app = PolicyAnalyzerApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"App error: {e}")

if __name__ == "__main__":
    main()