#!/usr/bin/env python3
"""
Data augmentation for existing elephant corpus.
Creates variations through paraphrasing, question generation, and context expansion.
"""

import random
import re
from pathlib import Path
from typing import List, Dict, Tuple
import json

class ElephantDataAugmenter:
    """Augment existing elephant data through various techniques"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.paraphrase_templates = self._load_paraphrase_templates()
        self.question_starters = self._load_question_starters()
        
    def _load_paraphrase_templates(self) -> Dict[str, List[str]]:
        """Templates for paraphrasing existing content"""
        return {
            'fact_starters': [
                "It is known that",
                "Research shows that", 
                "Scientists have discovered that",
                "Studies indicate that",
                "Observations reveal that",
                "Experts note that",
                "Field research demonstrates that",
                "Wildlife biologists report that"
            ],
            'explanation_starters': [
                "This happens because",
                "The reason for this is that", 
                "This behavior occurs when",
                "The explanation is that",
                "This is due to the fact that"
            ],
            'comparison_phrases': [
                "In contrast to",
                "Unlike", 
                "Compared to",
                "Different from",
                "In comparison with"
            ]
        }
    
    def _load_question_starters(self) -> List[str]:
        """Question patterns for generating new Q&A content"""
        return [
            "What makes",
            "How do",
            "Why are", 
            "When do",
            "Where can",
            "Which type of",
            "What causes",
            "How long do",
            "What happens when",
            "Why do"
        ]
    
    def paraphrase_sentences(self, sentences: List[str], num_variations: int = 3) -> List[str]:
        """Create paraphrased versions of existing sentences"""
        paraphrased = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:  # Skip very short sentences
                continue
                
            for _ in range(num_variations):
                # Choose a random paraphrasing technique
                technique = random.choice(['fact_starter', 'reorder', 'expand'])
                
                if technique == 'fact_starter':
                    starter = random.choice(self.paraphrase_templates['fact_starters'])
                    # Make first word lowercase if adding starter
                    modified = sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence
                    paraphrased.append(f"{starter} {modified}")
                    
                elif technique == 'reorder':
                    # Simple sentence reordering (move clauses around)
                    if ',' in sentence:
                        parts = sentence.split(',', 1)
                        if len(parts) == 2 and len(parts[1].strip()) > 10:
                            reordered = f"{parts[1].strip()}, {parts[0].lower()}"
                            paraphrased.append(reordered)
                    
                elif technique == 'expand':
                    # Add explanatory context
                    explanations = [
                        "This adaptation helps them survive in their natural habitat.",
                        "This behavior is essential for their species' survival.",
                        "This characteristic distinguishes them from other animals.",
                        "This trait has evolved over millions of years.",
                    ]
                    expanded = f"{sentence} {random.choice(explanations)}"
                    paraphrased.append(expanded)
        
        return paraphrased
    
    def generate_questions_from_statements(self, statements: List[str]) -> List[str]:
        """Convert statements into question-answer pairs"""
        qa_pairs = []
        
        # Keywords that indicate good facts to turn into questions
        question_keywords = [
            'weigh', 'eat', 'live', 'communicate', 'use', 'trunk', 
            'ears', 'memory', 'family', 'habitat', 'behavior'
        ]
        
        for statement in statements:
            if len(statement.strip()) < 30:
                continue
                
            # Find statements that contain question-worthy information
            for keyword in question_keywords:
                if keyword.lower() in statement.lower():
                    # Generate different types of questions
                    questions = self._generate_question_variants(statement, keyword)
                    
                    for question in questions:
                        qa_pair = f"Q: {question}\nA: {statement}\n"
                        qa_pairs.append(qa_pair)
                    
                    break  # Only generate questions for first matching keyword
        
        return qa_pairs
    
    def _generate_question_variants(self, statement: str, keyword: str) -> List[str]:
        """Generate different question variants from a statement"""
        questions = []
        
        # Pattern-based question generation
        if 'weigh' in statement.lower():
            questions.extend([
                "How much do elephants weigh?",
                "What is the average weight of an elephant?",
                "How heavy can elephants get?"
            ])
        elif 'eat' in statement.lower():
            questions.extend([
                "What do elephants eat?",
                "How much do elephants eat per day?",
                "What is an elephant's diet like?"
            ])
        elif 'trunk' in statement.lower():
            questions.extend([
                "How do elephants use their trunks?",
                "What can an elephant do with its trunk?",
                "Why do elephants have trunks?"
            ])
        elif 'communicate' in statement.lower():
            questions.extend([
                "How do elephants communicate?",
                "What methods do elephants use to communicate?",
                "Can elephants talk to each other?"
            ])
        else:
            # Generic questions
            questions.extend([
                f"What should I know about elephant {keyword}?",
                f"Can you tell me about elephant {keyword}?",
                f"How does {keyword} relate to elephants?"
            ])
        
        return questions
    
    def create_context_variations(self, text_chunks: List[str]) -> List[str]:
        """Create variations by combining and expanding context"""
        variations = []
        
        # Combine related chunks
        for i in range(len(text_chunks) - 1):
            chunk1 = text_chunks[i].strip()
            chunk2 = text_chunks[i + 1].strip()
            
            if (len(chunk1) > 30 and len(chunk2) > 30 and 
                'elephant' in chunk1.lower() and 'elephant' in chunk2.lower()):
                
                # Create connecting variations
                connectors = [
                    "Additionally,", "Furthermore,", "Moreover,", 
                    "In related news,", "Similarly,", "On the other hand,"
                ]
                
                connector = random.choice(connectors)
                combined = f"{chunk1} {connector} {chunk2.lower()}"
                variations.append(combined)
        
        return variations
    
    def augment_corpus(self, input_file: Path, output_file: Path, 
                      augmentation_factor: float = 2.0) -> int:
        """Augment an existing corpus file"""
        
        print(f"📖 Reading corpus from: {input_file}")
        
        # Read existing content
        with input_file.open('r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences and chunks
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        print(f"📝 Found {len(sentences)} sentences to augment")
        
        all_augmented = []
        
        # 1. Paraphrase existing sentences (40% of augmentation)
        print("  🔄 Creating paraphrases...")
        paraphrases = self.paraphrase_sentences(sentences[:200], num_variations=2)
        all_augmented.extend(paraphrases)
        
        # 2. Generate Q&A from statements (30% of augmentation) 
        print("  ❓ Generating Q&A pairs...")
        qa_pairs = self.generate_questions_from_statements(sentences[:150])
        all_augmented.extend(qa_pairs)
        
        # 3. Create context variations (30% of augmentation)
        print("  🔗 Creating context variations...")
        context_vars = self.create_context_variations(sentences)
        all_augmented.extend(context_vars)
        
        # Shuffle and write
        random.shuffle(all_augmented)
        
        with output_file.open('w', encoding='utf-8') as f:
            for item in all_augmented:
                f.write(item + "\n\n")
        
        size_bytes = output_file.stat().st_size
        print(f"✅ Augmentation complete!")
        print(f"📁 Generated {len(all_augmented)} augmented items")
        print(f"📊 Output size: {size_bytes / 1024:.1f} KB")
        
        return len(all_augmented)


def main():
    """Main augmentation function"""
    
    # Input: your existing corpus
    corpus_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/out"
    input_file = corpus_dir / "elephant_human_90_10_corpus.txt"
    
    # Output: augmented data
    output_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw" 
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "augmented_corpus.txt"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Augment the corpus
    augmenter = ElephantDataAugmenter(seed=42)
    num_items = augmenter.augment_corpus(input_file, output_file, augmentation_factor=2.0)
    
    print(f"\n🎉 Data augmentation complete!")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Generated: {num_items} augmented items")


if __name__ == "__main__":
    main()
