import os
import logging
import hashlib
import json
import time
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass
from dotenv import load_dotenv
import openai
from openai import OpenAI
import PyPDF2
import io
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    name: str
    api_name: str
    base_url: str
    has_thinking_tags: bool = False
    max_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9
    requires_content_filtering: bool = False

class CacheManager:
    """Simple file-based cache manager for LLM responses"""
    
    def __init__(self, cache_dir: str = "llm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_age_hours = 24  # Cache expires after 24 hours
    
    def _get_cache_key(self, text: str, model: str, prompt: str) -> str:
        """Generate cache key from text, model, and prompt"""
        content = f"{text}{model}{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get_cached_response(self, text: str, model: str, prompt: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        try:
            cache_key = self._get_cache_key(text, model, prompt)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            # Check cache age
            cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if cache_age_hours > self.max_cache_age_hours:
                cache_path.unlink()  # Remove expired cache
                logger.info(f"Removed expired cache: {cache_key}")
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                logger.info(f"Using cached response: {cache_key}")
                return cache_data['response']
                
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    def cache_response(self, text: str, model: str, prompt: str, response: str):
        """Cache the response"""
        try:
            cache_key = self._get_cache_key(text, model, prompt)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'timestamp': time.time(),
                'model': model,
                'response': response,
                'text_length': len(text),
                'prompt_length': len(prompt)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Cached response: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def clear_cache(self):
        """Clear all cached responses"""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                'cache_entries': len(cache_files),
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir)
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}

class ResponseProcessor:
    """Process responses from different model types"""
    
    @staticmethod
    def clean_thinking_tags(response: str) -> str:
        """Remove thinking tags from model responses"""
        # Remove <think>...</think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove <thinking>...</thinking> tags and their content
        response = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove any other thinking-related tags
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean up multiple newlines and whitespace
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        
        return response.strip()
    
    @staticmethod
    def extract_main_content(response: str) -> str:
        """Extract main content from response, handling various formats"""
        # First clean thinking tags
        cleaned = ResponseProcessor.clean_thinking_tags(response)
        
        # If response seems to have structured sections, try to extract the main analysis
        if '##' in cleaned or '###' in cleaned:
            # Response has markdown structure, return as-is
            return cleaned
        
        # Split by double newlines and filter out very short segments
        segments = [seg.strip() for seg in cleaned.split('\n\n') if len(seg.strip()) > 50]
        
        if segments:
            return '\n\n'.join(segments)
        
        return cleaned
    
    @staticmethod
    def validate_analysis_content(response: str, original_text: str) -> bool:
        """Validate that the analysis actually references the input text"""
        # Check if response mentions key indicators that it processed the document
        indicators = [
            'policy', 'document', 'text', 'analysis', 'provision', 
            'section', 'clause', 'implementation', 'beneficiar'
        ]
        
        response_lower = response.lower()
        text_lower = original_text.lower()
        
        # Check if response has policy-related content
        has_policy_content = any(indicator in response_lower for indicator in indicators)
        
        # Check if response is not just a generic template
        generic_phrases = [
            'i cannot analyze', 'please provide', 'i need more information',
            'upload a document', 'no document provided'
        ]
        
        is_generic = any(phrase in response_lower for phrase in generic_phrases)
        
        return has_policy_content and not is_generic and len(response.strip()) > 200

class LLMHandler:
    """Enhanced handler for LLM operations with improved model support and caching"""
    
    # Model configurations
    MODEL_CONFIGS = {
        
        "deepseek": ModelConfig(
            name="deepseek",
            api_name="deepseek-r1-distill-llama-70b",
            base_url="https://api.groq.com/openai/v1",
            has_thinking_tags=True,
            max_tokens=8192,
            temperature=0.1
        ),
        "qwen": ModelConfig(
            name="qwen",
            api_name="qwen/qwen3-32b",
            base_url="https://api.groq.com/openai/v1",
            has_thinking_tags=True,
            max_tokens=8192,
            temperature=0.2
        )
    }
    
    # Supported file types
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}
    
    def __init__(self, model_name: str = "qwen/qwen3-32b", enable_cache: bool = True):
        """Initialize LLM handler
        
        Args:
            model_name: Name of the model to use
            enable_cache: Whether to enable response caching
        """
        self.model_config = self._get_model_config(model_name)
        self.client = self._initialize_client()
        self.cache_manager = CacheManager() if enable_cache else None
        self.response_processor = ResponseProcessor()
        
        logger.info(f"Initialized LLM handler with model: {self.model_config.name}")
        
    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration"""
        if model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[model_name]
        else:
            logger.warning(f"Unknown model {model_name}, using default qwen/qwen3-32b")
            return self.MODEL_CONFIGS["qwen/qwen3-32b"]
    
    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client based on model configuration"""
        load_dotenv()
        
        # Get appropriate API key based on model
        if self.model_config.base_url == "https://openrouter.ai/api/v1":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        else:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_key")
            if not api_key:
                raise ValueError("GROQ_API_KEY or api_key not found in environment variables")
        
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=self.model_config.base_url
            )
            logger.info(f"Client initialized for {self.model_config.name}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            raise

    def summarize_policy(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """Summarize policy with enhanced error handling and model-specific processing
        
        Args:
            text: The policy text to summarize
            custom_prompt: Optional custom prompt override
            
        Returns:
            Processed and cleaned policy analysis
        """
        if not text or not text.strip():
            raise ValueError("Policy text cannot be empty")
        
        # Check cache first
        prompt = custom_prompt or self._get_default_prompt()
        full_prompt = self._build_analysis_prompt(text, prompt)
        
        if self.cache_manager:
            cached_response = self.cache_manager.get_cached_response(
                text, self.model_config.api_name, full_prompt
            )
            if cached_response:
                return cached_response
        
        try:
            # Make API call with model-specific settings
            response = self.client.chat.completions.create(
                model=self.model_config.api_name,
                messages=[
                    {
                        "role": "system", 
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user", 
                        "content": full_prompt
                    }
                ],
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
                top_p=self.model_config.top_p
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Process response based on model type
            if self.model_config.has_thinking_tags:
                processed_response = self.response_processor.clean_thinking_tags(raw_response)
            else:
                processed_response = self.response_processor.extract_main_content(raw_response)
            
            # Validate the response
            if not self.response_processor.validate_analysis_content(processed_response, text):
                logger.warning("Generated response may not properly reference the input document")
                # Retry with more explicit prompt
                retry_prompt = self._build_explicit_analysis_prompt(text, prompt)
                return self._retry_analysis(retry_prompt, text)
            
            # Cache the processed response
            if self.cache_manager:
                self.cache_manager.cache_response(
                    text, self.model_config.api_name, full_prompt, processed_response
                )
            
            logger.info(f"Policy analysis completed successfully with {self.model_config.name}")
            return processed_response
            
        except Exception as e:
            logger.error(f"Error during policy analysis: {e}")
            raise Exception(f"Failed to analyze policy: {str(e)}")
    
    def _retry_analysis(self, prompt: str, original_text: str) -> str:
        """Retry analysis with more explicit prompt"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.api_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are analyzing a specific document. Focus ONLY on the content provided by the user. Do not provide generic templates."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Lower temperature for more focused response
                max_tokens=self.model_config.max_tokens,
                top_p=0.8
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            if self.model_config.has_thinking_tags:
                return self.response_processor.clean_thinking_tags(raw_response)
            else:
                return self.response_processor.extract_main_content(raw_response)
                
        except Exception as e:
            logger.error(f"Retry analysis failed: {e}")
            raise
    
    def _build_analysis_prompt(self, text: str, base_prompt: str) -> str:
        """Build comprehensive analysis prompt with document content"""
        return f"""
DOCUMENT TO ANALYZE:
==================
{text[:8000]}...
==================

{base_prompt}

IMPORTANT: Base your analysis ONLY on the document content provided above. Do not provide generic templates or placeholder text.
"""
    
    def _build_explicit_analysis_prompt(self, text: str, base_prompt: str) -> str:
        """Build more explicit prompt for retry attempts"""
        return f"""
You must analyze the specific document provided below. Extract real information from this exact document.

DOCUMENT CONTENT:
{text[:10000]}

Instructions:
1. Read the document content carefully
2. Identify the actual policy name, provisions, and details from the text
3. Provide specific analysis based on what you read
4. Do NOT use generic templates or placeholder text
5. Quote specific sections where relevant

{base_prompt}
"""

    def extract_text_from_pdf(self, pdf_file: Union[str, bytes, io.BytesIO]) -> str:
        """Extract text from PDF with improved error handling"""
        try:
            text_content = []
            
            if isinstance(pdf_file, str):
                if not os.path.exists(pdf_file):
                    raise FileNotFoundError(f"PDF file not found: {pdf_file}")
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = self._extract_pages(pdf_reader)
            elif isinstance(pdf_file, bytes):
                pdf_stream = io.BytesIO(pdf_file)
                pdf_reader = PyPDF2.PdfReader(pdf_stream)
                text_content = self._extract_pages(pdf_reader)
            elif isinstance(pdf_file, io.BytesIO):
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = self._extract_pages(pdf_reader)
            else:
                raise ValueError("Unsupported PDF input type")
            
            full_text = '\n\n'.join(text_content)
            
            if not full_text.strip():
                raise Exception("No text could be extracted from PDF (might be image-based)")
            
            # Clean up the extracted text
            full_text = self._clean_extracted_text(full_text)
            
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between joined words
        text = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        
        # Normalize line breaks
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'\n+', '\n\n', text)
        
        return text.strip()
    
    def _extract_pages(self, pdf_reader: PyPDF2.PdfReader) -> List[str]:
        """Extract text from all pages with improved error handling"""
        text_content = []
        num_pages = len(pdf_reader.pages)
        
        logger.info(f"Processing PDF with {num_pages} pages")
        
        for page_num in range(num_pages):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text.strip():
                    # Clean and format page text
                    clean_text = self._clean_extracted_text(page_text)
                    text_content.append(f"[Page {page_num + 1}]\n{clean_text}")
                    logger.debug(f"Extracted {len(clean_text)} characters from page {page_num + 1}")
                else:
                    logger.warning(f"No text found on page {page_num + 1}")
                    
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        return text_content
    
    def process_file(self, file_path: str) -> str:
        """Process file with improved error handling"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        
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
        """Analyze file with comprehensive error handling and validation"""
        try:
            # Extract text from file
            text_content = self.process_file(file_path)
            
            # Validate text content
            if len(text_content.strip()) < 100:
                raise ValueError("Document too short for meaningful analysis")
            
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
            analysis = self.summarize_policy(text_content, custom_prompt)
            
            # Validate analysis quality
            quality_score = self._assess_analysis_quality(analysis, text_content)
            
            return {
                'success': True,
                'file_info': file_info,
                'extracted_text': text_content,
                'analysis': analysis,
                'model_used': self.model_config.name,
                'quality_score': quality_score,
                'cached': False  # This would be set by cache manager
            }
            
        except Exception as e:
            logger.error(f"File analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_info': {'filename': Path(file_path).name if Path(file_path).exists() else 'unknown'},
                'model_used': self.model_config.name
            }
    
    def _assess_analysis_quality(self, analysis: str, original_text: str) -> float:
        """Assess the quality of the analysis"""
        score = 0.0
        
        # Length check
        if len(analysis) > 500:
            score += 0.2
        
        # Structure check (has headings/sections)
        if '##' in analysis or 'Policy' in analysis:
            score += 0.2
        
        # Content relevance check
        if self.response_processor.validate_analysis_content(analysis, original_text):
            score += 0.4
        
        # Specific details check
        if any(keyword in analysis.lower() for keyword in ['beneficiar', 'implement', 'provision', 'target']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for policy analysis"""
        return """You are an expert Indian Government policy analyst with deep knowledge of:
- Indian constitutional law and governance structures
- Public policy implementation and bureaucratic processes
- Social welfare schemes and their effectiveness
- Economic policies and their impact on different demographics
- Legal frameworks and regulatory compliance
- Beneficiary identification and targeting mechanisms
- Implementation challenges in Indian context

Provide comprehensive, structured analysis that helps citizens understand policy implications and helps policymakers identify potential issues. Always base your analysis on the specific document provided."""
    
    def _get_default_prompt(self) -> str:
        """Get enhanced default prompt for policy analysis"""
        return """
Analyze this government policy document and provide a comprehensive structured analysis. Include:

1. **Policy Overview**
   - Policy name and issuing authority
   - Objectives and scope
   - Target beneficiaries

2. **Key Provisions**
   - Main features and benefits
   - Eligibility criteria
   - Implementation mechanism

3. **Implementation Framework**
   - Timeline and phases
   - Responsible agencies
   - Budget allocation

4. **Impact Assessment**
   - Expected outcomes
   - Beneficiary coverage
   - Social and economic impact

5. **Challenges and Recommendations**
   - Potential implementation challenges
   - Suggestions for improvement

Focus on the specific content of this document. Provide actionable insights based on the actual policy details.
"""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            'current_model': self.model_config.name,
            'api_name': self.model_config.api_name,
            'base_url': self.model_config.base_url,
            'has_thinking_tags': self.model_config.has_thinking_tags,
            'max_tokens': self.model_config.max_tokens,
            'available_models': list(self.MODEL_CONFIGS.keys())
        }
    
    def change_model(self, model_name: str) -> bool:
        """Change the current model"""
        try:
            old_model = self.model_config.name
            self.model_config = self._get_model_config(model_name)
            self.client = self._initialize_client()
            logger.info(f"Model changed from {old_model} to {self.model_config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            return False
    
    def clear_cache(self):
        """Clear the response cache"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return {'cache_disabled': True}
    
    def get_available_models(self) -> Dict[str, str]:
        """Return available models (for backward compatibility)"""
        return {name: config.api_name for name, config in self.MODEL_CONFIGS.items()}
    
    def get_supported_extensions(self) -> set:
        """Return supported file extensions"""
        return self.SUPPORTED_EXTENSIONS.copy()

# Example usage and testing
if __name__ == "__main__":
    try:
        # Test different models
        models_to_test = ["deepseek", "qwen"]
        
        for model in models_to_test:
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model}")
            print('='*60)
            
            try:
                # Initialize handler
                llm = LLMHandler(model_name=model, enable_cache=True)
                
                print(f"✅ Model initialized: {llm.get_model_info()}")
                
                # Test with sample files
                test_files = ["sample_policy.pdf", "policy_doc.txt"]
                
                for sample_file in test_files:
                    if os.path.exists(sample_file):
                        print(f"\n📄 Analyzing: {sample_file}")
                        
                        result = llm.analyze_file(sample_file)
                        
                        if result['success']:
                            print(f"✅ Analysis successful!")
                            print(f"📊 Quality Score: {result['quality_score']:.2f}")
                            print(f"📏 Analysis Length: {len(result['analysis'])} chars")
                            print("\n" + "="*50)
                            print("ANALYSIS PREVIEW")
                            print("="*50)
                            print(result['analysis'][:500] + "..." if len(result['analysis']) > 500 else result['analysis'])
                        else:
                            print(f"❌ Analysis failed: {result['error']}")
                    else:
                        print(f"⚠️ Test file {sample_file} not found")
                
                # Display cache stats
                cache_stats = llm.get_cache_stats()
                print(f"\n📊 Cache Stats: {cache_stats}")
                
            except Exception as e:
                print(f"❌ Error testing {model}: {e}")
                
    except Exception as e:
        print(f"❌ Test execution failed: {e}")