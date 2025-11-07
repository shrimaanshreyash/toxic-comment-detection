"""
Pattern Analyzer for Toxic Comment Detection

This module identifies patterns, trends, and escalation pathways in toxic comments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """
    Analyze patterns and trends in toxic comments.
    """
    
    TOXICITY_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the pattern analyzer.
        
        Args:
            df: DataFrame with toxic comment data
        """
        self.df = df
        logger.info("PatternAnalyzer initialized")
    
    def find_escalation_patterns(self) -> Dict:
        """
        Identify escalation patterns between toxicity categories.
        
        Returns:
            Dictionary containing escalation patterns and probabilities
        """
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        patterns = {}
        
        # Calculate co-occurrence probabilities
        for cat1 in self.TOXICITY_COLS:
            for cat2 in self.TOXICITY_COLS:
                if cat1 != cat2:
                    # P(cat2 | cat1) = P(cat1 AND cat2) / P(cat1)
                    cat1_count = self.df[cat1].sum()
                    both_count = ((self.df[cat1] == 1) & (self.df[cat2] == 1)).sum()
                    
                    if cat1_count > 0:
                        prob = both_count / cat1_count
                        patterns[f"{cat1} → {cat2}"] = {
                            'probability': prob,
                            'count': int(both_count),
                            'base_count': int(cat1_count)
                        }
        
        # Sort by probability
        patterns = dict(sorted(
            patterns.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        ))
        
        return patterns
    
    def analyze_multi_label_patterns(self) -> Dict:
        """
        Analyze patterns in multi-label toxic comments.
        
        Returns:
            Dictionary with multi-label statistics and patterns
        """
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        # Get toxic comments only
        toxic_df = self.df[self.df[self.TOXICITY_COLS].sum(axis=1) > 0].copy()
        
        # Count labels per comment
        toxic_df['num_labels'] = toxic_df[self.TOXICITY_COLS].sum(axis=1)
        
        # Find most common combinations
        label_combinations = []
        for _, row in toxic_df.iterrows():
            active_labels = [col for col in self.TOXICITY_COLS if row[col] == 1]
            if len(active_labels) > 1:
                label_combinations.append(tuple(sorted(active_labels)))
        
        combo_counts = Counter(label_combinations)
        
        return {
            'avg_labels_per_toxic': toxic_df['num_labels'].mean(),
            'max_labels': toxic_df['num_labels'].max(),
            'label_distribution': toxic_df['num_labels'].value_counts().to_dict(),
            'top_combinations': dict(combo_counts.most_common(10))
        }
    
    def analyze_temporal_patterns(
        self,
        datetime_col: Optional[str] = None
    ) -> Dict:
        """
        Analyze temporal patterns in toxicity (if datetime available).
        
        Args:
            datetime_col: Name of datetime column
            
        Returns:
            Dictionary with temporal patterns
        """
        if self.df is None or datetime_col is None:
            return {'message': 'Temporal analysis requires datetime column'}
        
        if datetime_col not in self.df.columns:
            return {'message': f'Column {datetime_col} not found'}
        
        # Convert to datetime
        self.df[datetime_col] = pd.to_datetime(self.df[datetime_col])
        
        # Extract time features
        self.df['hour'] = self.df[datetime_col].dt.hour
        self.df['day_of_week'] = self.df[datetime_col].dt.dayofweek
        self.df['month'] = self.df[datetime_col].dt.month
        
        # Analyze by hour
        hourly_toxic = self.df.groupby('hour')['toxic'].mean()
        
        # Analyze by day of week
        daily_toxic = self.df.groupby('day_of_week')['toxic'].mean()
        
        return {
            'hourly_pattern': hourly_toxic.to_dict(),
            'daily_pattern': daily_toxic.to_dict(),
            'peak_hour': int(hourly_toxic.idxmax()),
            'peak_day': int(daily_toxic.idxmax())
        }
    
    def find_toxic_keywords(self, top_n: int = 50) -> Dict[str, List[Tuple[str, int]]]:
        """
        Find most common keywords in toxic vs non-toxic comments.
        
        Args:
            top_n: Number of top keywords to return
            
        Returns:
            Dictionary with toxic and non-toxic keywords
        """
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        from collections import Counter
        import re
        
        def get_words(text):
            """Extract words from text."""
            if pd.isna(text):
                return []
            # Simple tokenization
            words = re.findall(r'\b[a-z]{3,}\b', str(text).lower())
            return words
        
        # Get toxic and non-toxic comments
        toxic_comments = self.df[self.df['toxic'] == 1]['comment_text']
        clean_comments = self.df[self.df['toxic'] == 0]['comment_text']
        
        # Extract words
        toxic_words = []
        for comment in toxic_comments:
            toxic_words.extend(get_words(comment))
        
        clean_words = []
        for comment in clean_comments:
            clean_words.extend(get_words(comment))
        
        # Count words
        toxic_counter = Counter(toxic_words)
        clean_counter = Counter(clean_words)
        
        # Get top words
        top_toxic = toxic_counter.most_common(top_n)
        top_clean = clean_counter.most_common(top_n)
        
        # Find words unique to toxic comments (relative frequency)
        toxic_unique = []
        for word, count in top_toxic:
            toxic_freq = count / len(toxic_words) if toxic_words else 0
            clean_freq = clean_counter.get(word, 0) / len(clean_words) if clean_words else 0
            
            if toxic_freq > clean_freq * 2:  # At least 2x more common in toxic
                toxic_unique.append((word, count, toxic_freq / max(clean_freq, 0.0001)))
        
        return {
            'top_toxic_words': top_toxic[:20],
            'top_clean_words': top_clean[:20],
            'toxic_unique_words': sorted(toxic_unique, key=lambda x: x[2], reverse=True)[:20]
        }
    
    def analyze_length_correlation(self) -> Dict:
        """
        Analyze correlation between comment length and toxicity.
        
        Returns:
            Dictionary with length statistics
        """
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        # Calculate lengths
        self.df['text_length'] = self.df['comment_text'].astype(str).str.len()
        self.df['word_count'] = self.df['comment_text'].astype(str).str.split().str.len()
        
        # Group by toxicity
        toxic_lengths = self.df[self.df['toxic'] == 1]['text_length']
        clean_lengths = self.df[self.df['toxic'] == 0]['text_length']
        
        toxic_words = self.df[self.df['toxic'] == 1]['word_count']
        clean_words = self.df[self.df['toxic'] == 0]['word_count']
        
        return {
            'toxic_avg_length': float(toxic_lengths.mean()),
            'clean_avg_length': float(clean_lengths.mean()),
            'toxic_avg_words': float(toxic_words.mean()),
            'clean_avg_words': float(clean_words.mean()),
            'length_difference_ratio': float(toxic_lengths.mean() / clean_lengths.mean()),
            'word_difference_ratio': float(toxic_words.mean() / clean_words.mean())
        }
    
    def generate_insights(self) -> List[Dict]:
        """
        Generate actionable insights from pattern analysis.
        
        Returns:
            List of insight dictionaries
        """
        insights = []
        
        # Insight 1: Escalation patterns
        escalations = self.find_escalation_patterns()
        top_escalation = list(escalations.items())[0] if escalations else None
        
        if top_escalation:
            pattern, data = top_escalation
            insights.append({
                'title': 'Primary Escalation Pattern',
                'description': f"Comments with {pattern.split(' → ')[0]} are {data['probability']:.1%} likely to also contain {pattern.split(' → ')[1]}",
                'impact': 'high',
                'action': f"Monitor for {pattern.split(' → ')[0]} as early warning sign",
                'data': data
            })
        
        # Insight 2: Multi-label clustering
        multi_label = self.analyze_multi_label_patterns()
        insights.append({
            'title': 'Multi-Label Toxicity Clustering',
            'description': f"Toxic comments average {multi_label['avg_labels_per_toxic']:.1f} toxicity types",
            'impact': 'medium',
            'action': 'Implement multi-label detection rather than single-category',
            'data': multi_label
        })
        
        # Insight 3: Length correlation
        length_corr = self.analyze_length_correlation()
        insights.append({
            'title': 'Comment Length Indicator',
            'description': f"Toxic comments are {length_corr['length_difference_ratio']:.1f}x longer than clean comments",
            'impact': 'medium',
            'action': 'Use length as additional signal in detection algorithm',
            'data': length_corr
        })
        
        # Insight 4: Keyword patterns
        keywords = self.find_toxic_keywords()
        if keywords['toxic_unique_words']:
            top_word = keywords['toxic_unique_words'][0]
            insights.append({
                'title': 'Toxic Language Markers',
                'description': f"Identified {len(keywords['toxic_unique_words'])} words significantly more common in toxic content",
                'impact': 'high',
                'action': 'Build keyword-based early detection system',
                'data': {'top_indicators': keywords['toxic_unique_words'][:10]}
            })
        
        return insights
    
    def visualize_escalation_heatmap(
        self,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create heatmap visualization of escalation patterns.
        
        Args:
            save_path: Path to save the figure
        """
        if self.df is None:
            raise ValueError("DataFrame not provided")
        
        # Calculate co-occurrence matrix
        matrix = np.zeros((len(self.TOXICITY_COLS), len(self.TOXICITY_COLS)))
        
        for i, cat1 in enumerate(self.TOXICITY_COLS):
            for j, cat2 in enumerate(self.TOXICITY_COLS):
                if i != j:
                    cat1_count = self.df[cat1].sum()
                    both_count = ((self.df[cat1] == 1) & (self.df[cat2] == 1)).sum()
                    
                    if cat1_count > 0:
                        matrix[i, j] = both_count / cat1_count
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            xticklabels=self.TOXICITY_COLS,
            yticklabels=self.TOXICITY_COLS,
            cbar_kws={'label': 'Conditional Probability'}
        )
        plt.title('Toxicity Escalation Patterns\nP(Column | Row)', fontsize=16, fontweight='bold')
        plt.ylabel('If comment has...', fontsize=12)
        plt.xlabel('...probability it also has:', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'comment_text': [f"Comment {i}" for i in range(1000)],
        'toxic': np.random.binomial(1, 0.1, 1000),
        'severe_toxic': np.random.binomial(1, 0.02, 1000),
        'obscene': np.random.binomial(1, 0.05, 1000),
        'threat': np.random.binomial(1, 0.01, 1000),
        'insult': np.random.binomial(1, 0.05, 1000),
        'identity_hate': np.random.binomial(1, 0.01, 1000),
    })
    
    # Initialize analyzer
    analyzer = PatternAnalyzer(sample_data)
    
    # Run analyses
    print("=== PATTERN ANALYSIS ===\n")
    
    escalations = analyzer.find_escalation_patterns()
    print("Top 5 Escalation Patterns:")
    for pattern, data in list(escalations.items())[:5]:
        print(f"  {pattern}: {data['probability']:.1%}")
    
    print("\nMulti-Label Analysis:")
    multi_label = analyzer.analyze_multi_label_patterns()
    print(f"  Average labels per toxic comment: {multi_label['avg_labels_per_toxic']:.2f}")
    
    print("\nLength Correlation:")
    length_corr = analyzer.analyze_length_correlation()
    print(f"  Toxic/Clean length ratio: {length_corr['length_difference_ratio']:.2f}x")
    
    print("\nGenerated Insights:")
    insights = analyzer.generate_insights()
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight['title']}")
        print(f"   {insight['description']}")
        print(f"   Action: {insight['action']}")