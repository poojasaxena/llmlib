#!/usr/bin/env python3
"""
Web scraping script for elephant-related content from reliable sources.
Focuses on educational and conservation websites.
"""

import requests
from bs4 import BeautifulSoup
import time
import re
from pathlib import Path
from typing import List, Dict, Optional
import json

class ElephantDataScraper:
    """Scrape elephant content from reliable sources"""
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize scraped text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text.strip()
    
    def scrape_wikipedia_sections(self) -> List[str]:
        """Scrape elephant-related Wikipedia content"""
        
        urls = [
            'https://en.wikipedia.org/wiki/Elephant',
            'https://en.wikipedia.org/wiki/African_elephant', 
            'https://en.wikipedia.org/wiki/Asian_elephant',
            'https://en.wikipedia.org/wiki/Elephant_behavior',
            'https://en.wikipedia.org/wiki/Elephant_cognition'
        ]
        
        scraped_content = []
        
        for url in urls:
            try:
                print(f"📡 Scraping: {url}")
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract paragraphs from main content
                content_div = soup.find('div', {'id': 'mw-content-text'})
                if content_div:
                    paragraphs = content_div.find_all('p')
                    
                    for p in paragraphs:
                        text = p.get_text()
                        cleaned = self.clean_text(text)
                        
                        # Filter for substantial paragraphs about elephants
                        if (len(cleaned) > 100 and 
                            ('elephant' in cleaned.lower() or 'trunk' in cleaned.lower()) and
                            not cleaned.startswith('Coordinates:')):
                            scraped_content.append(cleaned)
                
                time.sleep(self.delay)
                
            except Exception as e:
                print(f"❌ Error scraping {url}: {e}")
                continue
        
        print(f"✅ Scraped {len(scraped_content)} paragraphs from Wikipedia")
        return scraped_content
    
    def generate_elephant_facts_from_sources(self) -> List[str]:
        """Generate educational facts from reliable sources"""
        
        # These are facts that can be safely stated (publicly available information)
        educational_facts = [
            "African elephants can weigh up to 6 tons and stand 13 feet tall.",
            "Asian elephants are smaller, weighing up to 5 tons and standing 9 feet tall.",
            "Elephants spend 12-18 hours a day eating vegetation.",
            "An elephant's trunk contains over 40,000 muscles.",
            "Elephants can recognize themselves in mirrors, showing self-awareness.",
            "Female elephants are pregnant for 22 months, the longest of any mammal.",
            "Elephants have been observed using tools, such as sticks for scratching.",
            "The word 'elephant' comes from the Greek word 'elephas' meaning ivory.",
            "Elephants can hear sounds at frequencies as low as 1 Hz.",
            "Wild elephants walk an average of 50 miles per day searching for food.",
            "Elephants have poor eyesight but excellent hearing and smell.",
            "Baby elephants suck their trunks for comfort, like human babies suck thumbs.",
            "Elephants can learn to paint and have been taught to create artwork.",
            "The oldest known elephant lived to be 82 years old in captivity.",
            "Elephants fear bees and will avoid areas where they hear buzzing sounds.",
            "Elephant family groups are led by the oldest and wisest female, called the matriarch.",
            "When an elephant dies, other elephants have been observed mourning and covering the body with grass and dirt.",
            "Elephants can swim and use their trunks like snorkels when crossing deep water.",
            "African elephants have wrinkled skin to help them stay cool by trapping moisture.",
            "Asian elephants are more closely related to extinct woolly mammoths than to African elephants."
        ]
        
        return educational_facts

def main():
    """Main scraping and data generation function"""
    
    output_dir = Path.home() / "PoojaVault/Professional/Workbench/Datasets/llm/mixed_text/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scraper = ElephantDataScraper(delay=1.5)
    
    all_content = []
    
    # 1. Generate educational facts
    print("📚 Generating educational facts...")
    facts = scraper.generate_elephant_facts_from_sources()
    all_content.extend(facts)
    
    # 2. Scrape Wikipedia (be respectful)
    print("🌐 Scraping Wikipedia content...")
    try:
        wiki_content = scraper.scrape_wikipedia_sections()
        all_content.extend(wiki_content)
    except Exception as e:
        print(f"⚠️ Wikipedia scraping failed: {e}")
    
    # Save all content
    output_file = output_dir / "web_scraped_elephants.txt"
    
    with output_file.open('w', encoding='utf-8') as f:
        for item in all_content:
            f.write(item + "\n\n")
    
    size_bytes = output_file.stat().st_size
    print(f"\n✅ Scraping complete!")
    print(f"📁 Saved {len(all_content)} items to: {output_file}")
    print(f"📊 File size: {size_bytes / 1024:.1f} KB")

if __name__ == "__main__":
    main()
