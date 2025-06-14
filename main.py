import os
import logging
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
import openai
from openai import OpenAI
import PyPDF2
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for LLM operations using Groq API with PDF support"""
    
    # Available models
    MODELS = {
        "llama3-8b": "llama3-8b-8192",
        "deepseek":"deepseek-r1-distill-llama-70b",
        "Qwen":"qwen/qwen3-32b"
    }
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}
    
    def __init__(self, model_name: str = "llama3-8b"):
        """Initialize LLM handler
        
        Args:
            model_name: Name of the model to use (default: llama3-8b)
        """
        self.model_name = self._validate_model(model_name)
        self.client = self._initialize_client()
        
    def _validate_model(self, model_name: str) -> str:
        """Validate and return full model name"""
        if model_name in self.MODELS:
            return self.MODELS[model_name]
        elif model_name in self.MODELS.values():
            return model_name
        else:
            logger.warning(f"Unknown model {model_name}, using default llama3-8b-8192")
            return self.MODELS["llama3-8b"]
    
    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client with Groq endpoint"""
        load_dotenv()
        api_key = os.getenv("api_key")
        
        if not api_key:
            raise ValueError("API_KEY not found in environment variables")
        
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            logger.info(f"LLM client initialized successfully with model: {self.model_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    def summarize_policy(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """Summarize privacy policy or legal document
        
        Args:
            text: The policy text to summarize
            custom_prompt: Optional custom prompt override
            
        Returns:
            Summarized policy text
            
        Raises:
            ValueError: If text is empty
            Exception: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Policy text cannot be empty")
        
        prompt = custom_prompt or self._get_default_prompt(text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert Indian Government policy analyst with deep knowledge of:
- Indian constitutional law and governance structures
- Public policy implementation and bureaucratic processes
- Social welfare schemes and their effectiveness
- Economic policies and their impact on different demographics
- Legal frameworks and regulatory compliance
- Beneficiary identification and targeting mechanisms
- Implementation challenges in Indian context

Provide comprehensive, structured analysis that helps citizens understand policy implications and helps policymakers identify potential issues."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2048,
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Policy summarization completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error during policy summarization: {e}")
            raise Exception(f"Failed to summarize policy: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_file: Union[str, bytes, io.BytesIO]) -> str:
        """Extract text from PDF file
        
        Args:
            pdf_file: PDF file path, bytes, or BytesIO object
            
        Returns:
            Extracted text from PDF
            
        Raises:
            Exception: If PDF extraction fails
        """
        try:
            text_content = []
            
            # Handle different input types
            if isinstance(pdf_file, str):
                # File path
                if not os.path.exists(pdf_file):
                    raise FileNotFoundError(f"PDF file not found: {pdf_file}")
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = self._extract_pages(pdf_reader)
            elif isinstance(pdf_file, bytes):
                # Bytes object
                pdf_stream = io.BytesIO(pdf_file)
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                text_content = self._extract_pages(pdf_reader)
            elif isinstance(pdf_file, io.BytesIO):
                # BytesIO object
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = self._extract_pages(pdf_reader)
            else:
                raise ValueError("Unsupported PDF input type")
            
            # Join all pages
            full_text = '\n\n'.join(text_content)
            
            if not full_text.strip():
                raise Exception("No text could be extracted from PDF (might be image-based)")
            
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_pages(self, pdf_reader: PyPDF2.PdfReader) -> list:
        """Extract text from all pages of PDF
        
        Args:
            pdf_reader: PyPDF2 PdfReader object
            
        Returns:
            List of text content from each page
        """
        text_content = []
        num_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing PDF with {num_pages} pages")
        
        for page_num in range(num_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    logger.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
                else:
                    logger.warning(f"No text found on page {page_num + 1}")
                    
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        return text_content
    
    def process_file(self, file_path: str) -> str:
        """Process file and extract text based on file type
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file type is not supported
            Exception: If file processing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        
        try:
            if file_extension == '.pdf':
                return self.extract_text_from_pdf(str(file_path))
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Successfully read text file: {len(content)} characters")
                return content
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            raise
    
    def analyze_file(self, file_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a file (PDF or text) and return comprehensive results
        
        Args:
            file_path: Path to the file to analyze
            custom_prompt: Optional custom prompt override
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            # Extract text from file
            text_content = self.process_file(file_path)
            
            # Get file info
            file_path_obj = Path(file_path)
            file_info = {
                'filename': file_path_obj.name,
                'file_type': file_path_obj.suffix.lower(),
                'file_size': file_path_obj.stat().st_size,
                'character_count': len(text_content),
                'word_count': len(text_content.split()),
                'extraction_successful': True
            }
            
            # Perform analysis
            summary = self.summarize_policy(text_content, custom_prompt)
            
            return {
                'success': True,
                'file_info': file_info,
                'extracted_text': text_content,
                'analysis': summary,
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_info': {'filename': Path(file_path).name if Path(file_path).exists() else 'unknown'},
                'model_used': self.model_name
            }
    
    def _get_default_prompt(self, text: str) -> str:
        """Generate enhanced government policy analysis prompt"""
        return f"""
You are an expert assistant trained to analyze Indian government policies. Carefully study the following document and generate a structured summary covering each section in detail. Use simple, precise language suitable for policymakers, civil society, and common citizens.

📌 Instructions:
Read the policy text below and generate an organized response structured into these key sections:

🏛️ Policy Summary

Name and type of the policy (e.g., scheme, mission, yojana)

Ministry/department responsible for implementation

Objective: Main goals and focus areas

🎯 Target Beneficiaries

Who benefits from this policy? (e.g., farmers, women, youth, SC/ST, etc.)

Eligibility criteria and any exclusions

Coverage scope (national, state-wise, rural/urban)

💵 Government Support & Aid

Nature of support (cash, subsidies, services, assets, etc.)

Delivery mechanism (e.g., DBT, in-kind, digital portals)

Budget allocation or cost structure (if available)

✅ Benefits & Key Features

Strengths or innovations of the scheme specified in the policy. Be specific as given in the policy and don't be generic.

How it improves on previous initiatives or fills a gap

Notable success stories or early feedback (if mentioned)

🌍 Social & Economic Impact

How this policy improves quality of life, livelihoods, or equity

Expected economic benefits (e.g., employment, productivity, savings)

Gender, regional, or demographic-specific impact

⚠️ Implementation Challenges (if applicable)

Any identified barriers or risks (e.g., delays, corruption, awareness)

Suggestions for improving impact and outreach

📝 Policy Document (analyze the content below):

{text[:12000]}
{ '[TRUNCATED - content exceeds model input limit]' if len(text) > 12000 else '' }

📎 Note:

Be factual and avoid speculation.

Avoid the sections if it's not applicable to the policy. 

If information is missing in the text, do not assume it.

Make the summary informative, actionable, and structured."""
    
    def get_available_models(self) -> Dict[str, str]:
        """Return available models"""
        return self.MODELS.copy()
    
    def get_supported_extensions(self) -> set:
        """Return supported file extensions"""
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def change_model(self, model_name: str) -> bool:
        """Change the current model
        
        Args:
            model_name: New model name
            
        Returns:
            True if model changed successfully
        """
        try:
            old_model = self.model_name
            self.model_name = self._validate_model(model_name)
            logger.info(f"Model changed from {old_model} to {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False

# Example usage
if __name__ == "__main__":
    try:
        # Initialize handler
        llm = LLMHandler()
        
        # Test with sample files
        test_files = ["sbi_privacy.txt", "sample_policy.pdf"]
        
        for sample_file in test_files:
            if os.path.exists(sample_file):
                print(f"\n{'='*60}")
                print(f"ANALYZING: {sample_file}")
                print('='*60)
                
                # Use the new analyze_file method
                result = llm.analyze_file(sample_file)
                
                if result['success']:
                    print(f"✅ File processed successfully!")
                    print(f"📄 File: {result['file_info']['filename']}")
                    print(f"📊 Type: {result['file_info']['file_type']}")
                    print(f"📏 Size: {result['file_info']['file_size']} bytes")
                    print(f"🔤 Characters: {result['file_info']['character_count']:,}")
                    print(f"📝 Words: {result['file_info']['word_count']:,}")
                    print(f"🤖 Model: {result['model_used']}")
                    print("\n" + "="*50)
                    print("POLICY ANALYSIS")
                    print("="*50)
                    print(result['analysis'])
                else:
                    print(f"❌ Error processing {sample_file}: {result['error']}")
            else:
                print(f"⚠️ Sample file {sample_file} not found")
                
        # Test PDF text extraction directly
        print(f"\n{'='*60}")
        print("TESTING PDF EXTRACTION")
        print('='*60)
        
        # You can test with any PDF file
        test_pdf = "test_policy.pdf"
        if os.path.exists(test_pdf):
            try:
                extracted_text = llm.extract_text_from_pdf(test_pdf)
                print(f"✅ PDF extraction successful!")
                print(f"📏 Extracted {len(extracted_text)} characters")
                print(f"📄 Preview: {extracted_text[:200]}...")
            except Exception as e:
                print(f"❌ PDF extraction failed: {e}")
        else:
            print(f"ℹ️ No test PDF found at {test_pdf}")
            
    except Exception as e:
        print(f"❌ Error: {e}")