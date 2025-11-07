"""
Text Preprocessing Module for Toxic Comment Detection

This module handles all text preprocessing tasks including:
- Text cleaning (removing URLs, emails, special characters)
- Tokenization
- Stopword removal
- Lemmatization
- Feature extraction
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessing class for toxic comment detection.
    
    Attributes:
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to apply lemmatization
        lowercase (bool): Whether to convert to lowercase
    """
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True
    ):
        """
        Initialize the TextPreprocessor.
        
        Args:
            remove_stopwords: If True, remove stopwords from text
            lemmatize: If True, apply lemmatization
            lowercase: If True, convert text to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        logger.info("TextPreprocessor initialized successfully")
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages."""
        required_packages = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
        
        for package in required_packages:
            try:
                nltk.data.find(f'corpora/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, emails, special characters, etc.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove IP addresses
        text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)
        
        # Remove numbers (optional - keep if you think they're relevant)
        # text = re.sub(r'\d+', '', text)
        
        # Remove newlines and tabs
        text = re.sub(r'[\n\t]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation(self, text: str, keep_chars: str = "") -> str:
        """
        Remove punctuation from text.
        
        Args:
            text: Input text
            keep_chars: Punctuation characters to keep (e.g., "!?")
            
        Returns:
            Text without punctuation
        """
        # Create translation table
        chars_to_remove = string.punctuation
        for char in keep_chars:
            chars_to_remove = chars_to_remove.replace(char, '')
        
        translator = str.maketrans('', '', chars_to_remove)
        return text.translate(translator)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            # Fallback to simple split
            return text.split()
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        if not self.lemmatizer:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, return_tokens: bool = False) -> str | List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Input text
            return_tokens: If True, return list of tokens; if False, return string
            
        Returns:
            Preprocessed text (string or list of tokens)
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove punctuation (keep ! and ? as they may indicate toxicity)
        text = self.remove_punctuation(text, keep_chars="!?")
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Filter empty tokens
        tokens = [t for t in tokens if t.strip()]
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_batch(
        self, 
        texts: List[str], 
        return_tokens: bool = False
    ) -> List[str | List[str]]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            return_tokens: If True, return list of tokens for each text
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, return_tokens) for text in texts]


class FeatureExtractor:
    """
    Extract features from text for toxicity detection.
    """
    
    @staticmethod
    def extract_text_features(text: str) -> dict:
        """
        Extract basic text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = (
            features['char_count'] / features['word_count'] 
            if features['word_count'] > 0 else 0
        )
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_count'] = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = (
            features['caps_count'] / features['char_count'] 
            if features['char_count'] > 0 else 0
        )
        
        # Special character features
        features['special_char_count'] = sum(
            1 for c in text if c in string.punctuation
        )
        
        return features
    
    @staticmethod
    def has_toxic_patterns(text: str) -> dict:
        """
        Check for common toxic patterns.
        
        Args:
            text: Input text (lowercase)
            
        Returns:
            Dictionary of pattern flags
        """
        patterns = {}
        
        # Pattern lists (simplified - expand in production)
        insult_words = ['idiot', 'stupid', 'moron', 'dumb', 'fool']
        threat_words = ['kill', 'die', 'hurt', 'attack', 'destroy']
        obscene_words = ['damn', 'hell', 'crap']  # Add more as needed
        
        text_lower = text.lower()
        
        patterns['has_insult'] = any(word in text_lower for word in insult_words)
        patterns['has_threat'] = any(word in text_lower for word in threat_words)
        patterns['has_obscene'] = any(word in text_lower for word in obscene_words)
        patterns['has_you_attack'] = bool(re.search(r'\byou\s+(are|re)\s+\w+', text_lower))
        
        return patterns


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True,
        lowercase=True
    )
    
    # Test preprocessing
    test_texts = [
        "You are such an IDIOT! I hate you!!!",
        "This is a normal comment about the article.",
        "Visit http://malicious-site.com for more info"
    ]
    
    print("Text Preprocessing Examples:")
    print("=" * 80)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        preprocessed = preprocessor.preprocess(text)
        print(f"Preprocessed: {preprocessed}")
        
        # Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_text_features(text)
        print(f"Features: {features}")
        
        patterns = extractor.has_toxic_patterns(text)
        print(f"Toxic patterns: {patterns}")
