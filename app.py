import streamlit as st
import io
import time
from typing import Optional
import logging
from main import LLMHandler

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
            "Paste your Governmen policy here:",
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
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.download_button(
                    "📥 Download Analysis",
                    data=summary,
                    file_name=f"policy_analysis_{int(time.time())}.txt",
                    mime="text/plain"
                ):
                    st.success("Analysis downloaded!")
            
            with col2:
                if st.button("📋 Copy to Clipboard"):
                    st.info("Use your browser's copy function on the analysis text above")
            
            with col3:
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