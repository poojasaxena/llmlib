#!/usr/bin/env python3
"""
Advanced Synthetic Data Generation for Elephant Domain
Creates diverse, high-quality synthetic elephant data using multiple techniques.
"""

import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re

class AdvancedElephantDataGenerator:
    """Generate diverse synthetic elephant data using multiple techniques"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.elephant_facts = self._load_base_facts()
        self.conversation_templates = self._load_conversation_templates()
        self.question_patterns = self._load_question_patterns()
        
    def _load_base_facts(self) -> List[str]:
        """Core elephant facts for variation generation"""
        return [
            "African elephants are larger than Asian elephants.",
            "Elephants use their trunks for breathing, drinking, and communication.",
            "Baby elephants are called calves and weigh about 250 pounds at birth.",
            "Elephants have excellent memories and can remember other elephants for decades.",
            "Asian elephants have smaller ears than African elephants.",
            "Elephants are herbivores and eat up to 300 pounds of vegetation daily.",
            "Elephant tusks are actually elongated incisor teeth.",
            "Elephants live in matriarchal societies led by the oldest female.",
            "Elephants communicate through infrasonic sounds below human hearing range.",
            "Wild elephants can live 60-70 years in the wild.",
            "Elephants are found in Africa and Asia in grasslands and forests.",
            "Elephants play a crucial role in their ecosystem as keystone species.",
            "Poaching for ivory remains the biggest threat to elephant populations.",
            "Elephants show empathy and mourn their dead companions.",
            "A group of elephants is called a herd or parade.",
        ]
    
    def _load_conversation_templates(self) -> List[Dict]:
        """Templates for natural conversations about elephants"""
        return [
            {
                "pattern": "Did you know that {fact}?",
                "responses": ["That's amazing!", "I had no idea!", "Really? Tell me more!", "Fascinating!"]
            },
            {
                "pattern": "I'm curious about {topic}. What can you tell me?",
                "responses": ["Great question! {explanation}", "Here's what I know: {explanation}"]
            },
            {
                "pattern": "Why do elephants {behavior}?",
                "responses": ["Elephants {behavior} because {reason}", "This behavior is important for {purpose}"]
            }
        ]
    
    def _load_question_patterns(self) -> List[str]:
        """Question patterns for generating Q&A pairs"""
        return [
            "What is the difference between {type1} and {type2} elephants?",
            "How do elephants use their {body_part}?",
            "Why are elephants important to {ecosystem_type}?",
            "What do elephants eat in {habitat}?",
            "How do elephants {behavior} in the wild?",
            "What threats do {elephant_type} elephants face?",
            "How long do elephants {life_stage}?",
            "What makes elephant {characteristic} unique?",
        ]

    def generate_fact_variations(self, base_facts: List[str], num_variations: int = 5) -> List[str]:
        """Generate variations of existing facts"""
        variations = []
        
        variation_templates = [
            "Did you know that {}",
            "It's interesting that {}",
            "One fascinating fact is that {}",
            "Researchers have found that {}",
            "Studies show that {}",
            "Many people don't realize that {}",
        ]
        
        for fact in base_facts:
            for _ in range(num_variations):
                template = random.choice(variation_templates)
                # Ensure fact ends with period
                clean_fact = fact.rstrip('.') + '.'
                variation = template.format(clean_fact.lower())
                variations.append(variation)
                
        return variations

    def generate_conversations(self, num_conversations: int = 100) -> List[str]:
        """Generate natural conversations about elephants"""
        conversations = []
        
        topics = [
            "elephant behavior", "elephant habitat", "elephant conservation", 
            "elephant anatomy", "elephant intelligence", "elephant families",
            "elephant migration", "elephant communication", "elephant diet",
            "elephant threats", "elephant reproduction", "elephant evolution"
        ]
        
        for _ in range(num_conversations):
            topic = random.choice(topics)
            num_turns = random.randint(3, 6)
            
            conversation = []
            conversation.append(f"Human: I'm interested in learning about {topic}.")
            
            # Generate conversation turns
            for turn in range(num_turns - 1):
                if turn % 2 == 0:  # AI response
                    fact = random.choice(self.elephant_facts)
                    response = f"Assistant: {fact} Would you like to know more about this?"
                else:  # Human follow-up
                    follow_ups = [
                        "That's really interesting! Can you tell me more?",
                        "How does that compare to other animals?", 
                        "What are the implications of this?",
                        "Are there any conservation concerns related to this?",
                        "How do researchers study this behavior?"
                    ]
                    response = f"Human: {random.choice(follow_ups)}"
                
                conversation.append(response)
            
            conversations.append("\n".join(conversation) + "\n")
            
        return conversations

    def generate_qa_pairs(self, num_pairs: int = 200) -> List[str]:
        """Generate question-answer pairs"""
        qa_pairs = []
        
        # Question templates with specific elephant content
        templates = [
            ("What is the average lifespan of {elephant_type} elephants?", 
             "{elephant_type} elephants typically live {lifespan} years in the wild."),
            ("How much do {elephant_type} elephants weigh?", 
             "Adult {elephant_type} elephants weigh between {weight_range}."),
            ("What do elephants eat in {season}?", 
             "During {season}, elephants primarily eat {food_items}."),
            ("How do elephants communicate over long distances?", 
             "Elephants use infrasonic calls that can travel several kilometers."),
            ("Why do elephants have such large ears?", 
             "Large ears help elephants regulate body temperature in hot climates."),
        ]
        
        # Content variations
        elephant_types = ["African", "Asian", "forest", "savanna"]
        lifespans = ["50-70", "60-70", "40-60"]
        weight_ranges = ["4,000-7,000 kg", "3,000-5,000 kg", "2,700-4,000 kg"]
        seasons = ["dry season", "wet season", "winter", "summer"]
        food_items = ["grasses and bark", "fruits and leaves", "roots and branches"]
        
        for _ in range(num_pairs):
            template_q, template_a = random.choice(templates)
            
            # Fill in variables
            qa_text = f"Q: {template_q}\nA: {template_a}\n"
            qa_text = qa_text.replace("{elephant_type}", random.choice(elephant_types))
            qa_text = qa_text.replace("{lifespan}", random.choice(lifespans))
            qa_text = qa_text.replace("{weight_range}", random.choice(weight_ranges))
            qa_text = qa_text.replace("{season}", random.choice(seasons))
            qa_text = qa_text.replace("{food_items}", random.choice(food_items))
            
            qa_pairs.append(qa_text)
            
        return qa_pairs

    def generate_descriptive_passages(self, num_passages: int = 50) -> List[str]:
        """Generate longer descriptive passages about elephants"""
        passages = []
        
        passage_starters = [
            "In the African savanna, elephants roam freely across vast territories.",
            "Asian elephants, smaller than their African cousins, inhabit dense forests.",
            "The matriarchal structure of elephant herds is fascinating to observe.",
            "Conservation efforts for elephants face numerous challenges worldwide.",
            "Elephant intelligence continues to amaze researchers and wildlife enthusiasts.",
        ]
        
        for _ in range(num_passages):
            starter = random.choice(passage_starters)
            
            # Add 2-4 more sentences
            facts_sample = random.sample(self.elephant_facts, random.randint(2, 4))
            passage = starter + " " + " ".join(facts_sample)
            
            passages.append(passage + "\n")
            
        return passages

    def generate_all_synthetic_data(self, output_file: Path, total_target: int = 50000):
        """Generate comprehensive synthetic dataset"""
        
        print(f"🤖 Generating synthetic elephant data...")
        
        all_data = []
        
        # 1. Fact variations (30%)
        print("  📝 Generating fact variations...")
        variations = self.generate_fact_variations(self.elephant_facts, num_variations=8)
        all_data.extend(variations)
        
        # 2. Conversations (25%)
        print("  💬 Generating conversations...")
        conversations = self.generate_conversations(num_conversations=200)
        all_data.extend(conversations)
        
        # 3. Q&A pairs (25%) 
        print("  ❓ Generating Q&A pairs...")
        qa_pairs = self.generate_qa_pairs(num_pairs=300)
        all_data.extend(qa_pairs)
        
        # 4. Descriptive passages (20%)
        print("  📖 Generating descriptive passages...")
        passages = self.generate_descriptive_passages(num_passages=100)
        all_data.extend(passages)
        
        # Shuffle and write
        random.shuffle(all_data)
        
        with output_file.open('w', encoding='utf-8') as f:
            for item in all_data:
                f.write(item + "\n")
        
        print(f"✅ Generated {len(all_data)} synthetic items")
        print(f"📁 Saved to: {output_file}")
        
        # Calculate size
        size_bytes = output_file.stat().st_size
        print(f"📊 File size: {size_bytes / 1024:.1f} KB")
        
        return len(all_data)


def main():
    """Generate advanced synthetic elephant data"""
    
    # Output path
    output_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_advanced.txt"
    
    # Generate data
    generator = AdvancedElephantDataGenerator(seed=42)
    num_items = generator.generate_all_synthetic_data(output_file)
    
    print(f"\n🎉 Synthetic data generation complete!")
    print(f"Generated {num_items} items in {output_file}")


if __name__ == "__main__":
    main()
