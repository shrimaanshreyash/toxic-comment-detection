"""
Toxic Comment Detector

Main detection framework for identifying and categorizing toxic comments.
Uses rule-based and pattern matching approaches with confidence scoring.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from preprocessor import TextPreprocessor, FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToxicCommentDetector:
    """
    Main class for toxic comment detection and categorization.
    
    This detector uses a rule-based approach with pattern matching,
    keyword detection, and feature analysis to identify and categorize
    toxic comments into 6 categories:
    - toxic: General toxicity
    - severe_toxic: Extremely toxic content
    - obscene: Obscene language
    - threat: Threatening language
    - insult: Personal insults
    - identity_hate: Identity-based hate speech
    """
    
    # Toxicity categories
    CATEGORIES = [
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate'
    ]
    
    # Detection thresholds (can be tuned)
    THRESHOLDS = {
        'toxic': 0.3,
        'severe_toxic': 0.5,
        'obscene': 0.4,
        'threat': 0.6,
        'insult': 0.4,
        'identity_hate': 0.5
    }
    
    def __init__(self, threshold_multiplier: float = 1.0):
        """
        Initialize the detector.
        
        Args:
            threshold_multiplier: Adjust detection sensitivity (higher = stricter)
        """
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.threshold_multiplier = threshold_multiplier
        
        # Load keyword patterns
        self._load_patterns()
        
        logger.info("ToxicCommentDetector initialized")
    
    def _load_patterns(self) -> None:
        """Load keyword patterns for each toxicity category."""
        
        # INSULT patterns
        self.insult_patterns = {
            'direct_insults': [
                'idiot', 'stupid', 'moron', 'dumb', 'fool', 'loser',
                'pathetic', 'worthless', 'useless', 'incompetent'
            ],
            'intellectual_insults': [
                'ignorant', 'clueless', 'brainless', 'mindless'
            ],
            'you_patterns': [
                r'\byou\s+(are|re)\s+(an?\s+)?(idiot|stupid|moron|fool)',
                r'\byou\s+(suck|fail)',
                r'you\s+have\s+no\s+(brain|clue|idea)'
            ]
        }
        
        # THREAT patterns
        self.threat_patterns = {
            'violence': [
                'kill', 'murder', 'hurt', 'harm', 'beat', 'attack',
                'destroy', 'shoot', 'stab', 'punch'
            ],
            'intent': [
                r'(i|we)\s+(will|gonna|going\s+to)\s+(kill|hurt|destroy)',
                r'(i|we)\s+(hope|wish)\s+you\s+(die|get\s+hurt)',
                r'you\s+(should|deserve\s+to)\s+(die|be\s+killed)'
            ],
            'warnings': [
                'watch out', 'be careful', 'or else', 'regret'
            ]
        }
        
        # OBSCENE patterns
        self.obscene_patterns = {
            'profanity': [
                'damn', 'hell', 'crap', 'shit', 'fuck', 'ass',
                'bitch', 'bastard'
            ],
            'sexual': [
                'sex', 'sexy', 'porn', 'nude'
            ]
        }
        
        # IDENTITY_HATE patterns
        self.identity_hate_patterns = {
            'slurs': [
                # Note: In production, use comprehensive lists from hate speech databases
                # This is simplified for demonstration
            ],
            'group_attacks': [
                r'all\s+(women|men|muslims|jews|blacks|whites)\s+are',
                r'(hate|kill)\s+all\s+(women|men|muslims|jews|blacks|whites)'
            ],
            'stereotypes': [
                'typical', 'always', 'never', 'every single'
            ]
        }
        
        # TOXIC general patterns
        self.toxic_patterns = {
            'aggression': [
                'hate', 'angry', 'furious', 'pissed', 'rage'
            ],
            'dismissive': [
                'shut up', 'get lost', 'go away', 'get out'
            ],
            'attacks': [
                'you are', "you're", 'your stupid', 'your pathetic'
            ]
        }
    
    def _calculate_pattern_score(
        self,
        text: str,
        patterns: Dict[str, List[str]]
    ) -> float:
        """
        Calculate pattern matching score for a category.
        
        Args:
            text: Input text (lowercase)
            patterns: Dictionary of pattern lists
            
        Returns:
            Score between 0 and 1
        """
        text_lower = text.lower()
        matches = 0
        total_patterns = 0
        
        for pattern_type, pattern_list in patterns.items():
            total_patterns += len(pattern_list)
            
            for pattern in pattern_list:
                # Check if pattern is regex or simple keyword
                if pattern.startswith(r'\b') or '(' in pattern:
                    # Regex pattern
                    if re.search(pattern, text_lower):
                        matches += 2  # Regex matches are more specific
                else:
                    # Simple keyword
                    if pattern in text_lower:
                        matches += 1
        
        if total_patterns == 0:
            return 0.0
        
        # Normalize score
        score = min(matches / (total_patterns * 0.5), 1.0)
        return score
    
    def _analyze_features(self, text: str) -> Dict[str, float]:
        """
        Analyze text features that correlate with toxicity.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of feature scores
        """
        features = self.feature_extractor.extract_text_features(text)
        
        scores = {}
        
        # Caps ratio (shouting)
        scores['caps_score'] = min(features['caps_ratio'] * 2, 1.0)
        
        # Exclamation points (aggression)
        scores['exclamation_score'] = min(features['exclamation_count'] / 3, 1.0)
        
        # Length score (toxic comments tend to be longer)
        if features['word_count'] > 50:
            scores['length_score'] = 0.3
        elif features['word_count'] > 100:
            scores['length_score'] = 0.5
        else:
            scores['length_score'] = 0.1
        
        return scores
    
    def detect_insult(self, text: str) -> float:
        """
        Detect insult toxicity.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        pattern_score = self._calculate_pattern_score(text, self.insult_patterns)
        
        # Boost score if "you" is mentioned (personal attack)
        if re.search(r'\byou\b', text.lower()):
            pattern_score *= 1.3
        
        return min(pattern_score, 1.0)
    
    def detect_threat(self, text: str) -> float:
        """
        Detect threatening language.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        pattern_score = self._calculate_pattern_score(text, self.threat_patterns)
        
        # Threats are serious - boost violence keywords
        text_lower = text.lower()
        violence_count = sum(
            1 for word in self.threat_patterns['violence'] 
            if word in text_lower
        )
        
        if violence_count > 0:
            pattern_score += violence_count * 0.2
        
        return min(pattern_score, 1.0)
    
    def detect_obscene(self, text: str) -> float:
        """
        Detect obscene language.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        pattern_score = self._calculate_pattern_score(text, self.obscene_patterns)
        
        # Multiple profanity words increase score
        text_lower = text.lower()
        profanity_count = sum(
            1 for word in self.obscene_patterns['profanity']
            if word in text_lower
        )
        
        if profanity_count > 1:
            pattern_score *= 1.2
        
        return min(pattern_score, 1.0)
    
    def detect_identity_hate(self, text: str) -> float:
        """
        Detect identity-based hate speech.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        pattern_score = self._calculate_pattern_score(text, self.identity_hate_patterns)
        
        # Look for group generalizations
        text_lower = text.lower()
        if re.search(r'\ball\s+(women|men|muslims|jews|blacks|whites|gays)', text_lower):
            pattern_score += 0.3
        
        return min(pattern_score, 1.0)
    
    def detect_toxic(self, text: str) -> float:
        """
        Detect general toxicity.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        # General toxic score is based on overall patterns
        pattern_score = self._calculate_pattern_score(text, self.toxic_patterns)
        
        # Analyze features
        feature_scores = self._analyze_features(text)
        
        # Combine scores
        combined_score = (
            pattern_score * 0.6 +
            feature_scores['caps_score'] * 0.2 +
            feature_scores['exclamation_score'] * 0.2
        )
        
        return min(combined_score, 1.0)
    
    def detect_severe_toxic(self, text: str) -> float:
        """
        Detect severe toxicity (combination of multiple toxic elements).
        
        Args:
            text: Input text
            
        Returns:
            Confidence score (0-1)
        """
        # Severe toxic is when multiple categories are present
        scores = {
            'insult': self.detect_insult(text),
            'threat': self.detect_threat(text),
            'obscene': self.detect_obscene(text),
            'identity_hate': self.detect_identity_hate(text)
        }
        
        # Count how many categories are positive
        positive_categories = sum(1 for score in scores.values() if score > 0.3)
        
        if positive_categories >= 2:
            # Multiple toxic elements = severe
            return min(max(scores.values()) * 1.2, 1.0)
        else:
            # Very high score in any category can also be severe
            max_score = max(scores.values())
            if max_score > 0.7:
                return max_score * 0.9
            else:
                return 0.0
    
    def predict(self, text: str) -> Dict:
        """
        Predict toxicity for a comment.
        
        Args:
            text: Input comment text
            
        Returns:
            Dictionary containing:
            - is_toxic: Boolean
            - categories: List of detected categories
            - scores: Dictionary of category scores
            - confidence: Overall confidence score
            - severity: 'none', 'mild', 'moderate', 'severe'
        """
        if not text or not isinstance(text, str):
            return {
                'is_toxic': False,
                'categories': [],
                'scores': {},
                'confidence': 0.0,
                'severity': 'none'
            }
        
        # Calculate scores for each category
        scores = {
            'toxic': self.detect_toxic(text),
            'severe_toxic': self.detect_severe_toxic(text),
            'obscene': self.detect_obscene(text),
            'threat': self.detect_threat(text),
            'insult': self.detect_insult(text),
            'identity_hate': self.detect_identity_hate(text)
        }
        
        # Apply thresholds
        detected_categories = []
        for category, score in scores.items():
            threshold = self.THRESHOLDS[category] * self.threshold_multiplier
            if score >= threshold:
                detected_categories.append(category)
        
        # Determine overall toxicity
        is_toxic = len(detected_categories) > 0
        
        # Calculate confidence (max score among detected categories)
        if is_toxic:
            confidence = max(scores[cat] for cat in detected_categories)
        else:
            confidence = max(scores.values())  # Confidence even if not toxic
        
        # Determine severity
        severity = self._calculate_severity(scores, detected_categories)
        
        return {
            'is_toxic': is_toxic,
            'categories': detected_categories,
            'scores': scores,
            'confidence': round(confidence, 3),
            'severity': severity
        }
    
    def _calculate_severity(
        self,
        scores: Dict[str, float],
        detected_categories: List[str]
    ) -> str:
        """
        Calculate severity level.
        
        Args:
            scores: Dictionary of category scores
            detected_categories: List of detected categories
            
        Returns:
            Severity level: 'none', 'mild', 'moderate', 'severe'
        """
        if not detected_categories:
            return 'none'
        
        max_score = max(scores.values())
        num_categories = len(detected_categories)
        
        # Severe: multiple categories or very high scores
        if num_categories >= 3 or max_score > 0.8 or 'severe_toxic' in detected_categories:
            return 'severe'
        
        # Moderate: 2 categories or high score
        elif num_categories == 2 or max_score > 0.6:
            return 'moderate'
        
        # Mild: 1 category with moderate score
        else:
            return 'mild'
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict toxicity for multiple comments.
        
        Args:
            texts: List of comment texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ToxicCommentDetector()
    
    # Test comments
    test_comments = [
        "This is a great article, thank you for sharing!",
        "You are such an idiot, this is completely wrong!",
        "I will destroy you and your family!",
        "This is fucking stupid and obscene",
        "All Muslims are terrorists and should be banned",
        "Hey! Check out this cool website!"
    ]
    
    print("=" * 80)
    print("TOXIC COMMENT DETECTION EXAMPLES")
    print("=" * 80)
    
    for i, comment in enumerate(test_comments, 1):
        print(f"\n{i}. Comment: \"{comment}\"")
        print("-" * 80)
        
        result = detector.predict(comment)
        
        print(f"   Toxic: {result['is_toxic']}")
        print(f"   Severity: {result['severity']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if result['categories']:
            print(f"   Categories: {', '.join(result['categories'])}")
            print(f"   Scores:")
            for cat in result['categories']:
                print(f"     - {cat}: {result['scores'][cat]:.3f}")
