# Enhanced Data Pipeline Documentation

## Overview

The enhanced `run_full_data_pipeline.py` now automatically incorporates multiple data sources for comprehensive LLM training data preparation:

1. **Base corpus preparation** (raw text files)
2. **Basic synthetic expansion** (existing functionality)
3. **Advanced synthetic generation** (new script: `advanced_synth_expand.py`)
4. **Web scraping** (new script: `web_scrape_elephants.py`)
5. **Data augmentation** (new script: `augment_corpus.py`)
6. **Intelligent file combination** (automatically finds and combines all data sources)
7. **Deduplication and filtering**
8. **Train/val/test splitting**
9. **Statistics and cleanup**

## Usage

### Basic Usage

```bash
# Ensure GLOBAL_DATASETS_DIR is set
export GLOBAL_DATASETS_DIR="/path/to/your/datasets"

# Run the enhanced pipeline
python src/llmlib/data/pipeline/run_full_data_pipeline.py
```

### Pipeline Steps in Detail

#### Step 1: Base Corpus Preparation
- Reads all `.txt` files from `$GLOBAL_DATASETS_DIR/llm/mixed_text/raw/`
- Cleans and combines into `elephant_human_90_10_corpus.txt`
- Shuffles content but doesn't dedupe (dedup happens later)

#### Step 2: Basic Synthetic Expansion
- Uses existing `synth_expand.py` to generate variations
- Creates `synthetic_generated.txt` with up to 20,000 lines

#### Step 3: Advanced Data Generation
The pipeline automatically runs these scripts if they exist:

**Advanced Synthetic Expansion** (`advanced_synth_expand.py`):
```bash
# Manually run: 
python src/llmlib/data/advanced_synth_expand.py /path/to/output/dir /path/to/source.txt 10000
```
- Creates contextually rich synthetic examples
- Generates educational content, dialogues, stories
- Output: `advanced_synthetic_YYYYMMDD_HHMMSS.txt`

**Web Scraping** (`web_scrape_elephants.py`):
```bash
# Manually run:
python src/llmlib/data/web_scrape_elephants.py /path/to/output/dir 1000
```
- Scrapes elephant-related content from educational websites
- Respectful scraping with rate limiting
- Output: `web_scraped_YYYYMMDD_HHMMSS.txt`

**Data Augmentation** (`augment_corpus.py`):
```bash
# Manually run:
python src/llmlib/data/augment_corpus.py /path/to/output/dir /path/to/source.txt
```
- Applies NLP transformations: paraphrasing, style transfer, etc.
- Creates variations while preserving meaning
- Output: `augmented_YYYYMMDD_HHMMSS.txt`

#### Step 4: Intelligent File Combination
The pipeline automatically discovers and combines:
- Base corpus files
- All synthetic generation outputs
- Web scraped content  
- Augmented data files
- Any files matching patterns: `*_synthetic.txt`, `*_augmented.txt`, `*_scraped.txt`

#### Step 5: Deduplication & Filtering
- Removes exact duplicates
- Filters by line length (10-500 characters)
- Reports reduction statistics

#### Step 6: Train/Val/Test Split
- 80%/10%/10% split by default
- Deterministic with seed=42
- Creates `train.txt`, `val.txt`, `test.txt`

#### Step 7: Statistics & Cleanup
- Generates corpus statistics
- Optionally removes intermediate files
- Preserves key files for reference

## File Organization

### Input Files (Raw Data)
```
$GLOBAL_DATASETS_DIR/llm/mixed_text/raw/
├── elephant_facts.txt
├── conservation_info.txt
└── ...other .txt files
```

### Generated Data Files
```
$GLOBAL_DATASETS_DIR/llm/mixed_text/out/
├── elephant_human_90_10_corpus.txt           # Base cleaned corpus
├── synthetic_generated.txt                    # Basic synthetic
├── advanced_synthetic_YYYYMMDD_HHMMSS.txt    # Advanced synthetic
├── web_scraped_YYYYMMDD_HHMMSS.txt          # Web scraped content
├── augmented_YYYYMMDD_HHMMSS.txt            # Augmented content
├── corpus_all_sources_combined.txt           # All sources combined
├── corpus_final_dedup.txt                    # Final deduplicated
├── train.txt                                 # Training set
├── val.txt                                   # Validation set
└── test.txt                                  # Test set
```

## Configuration

### Environment Variables
- `GLOBAL_DATASETS_DIR`: Root directory for datasets (required)

### Pipeline Parameters
Edit `run_full_data_pipeline.py` to adjust:
- `max_lines=20_000`: Synthetic generation limit
- `min_len=10, max_len=500`: Text filtering bounds
- `splits=(0.8, 0.1, 0.1)`: Train/val/test ratios
- `seed=42`: Random seed for reproducibility

### Data Generation Scripts
Each script accepts command-line arguments:
- Output directory (required)
- Source file or count parameters
- Additional options (see script help)

## Monitoring Pipeline Progress

The enhanced pipeline provides detailed logging:
```
🐘 Enhanced Elephant LLM Data Pipeline Starting...
📁 Data directory: /path/to/output

=== Step 1: Prepare raw corpus ===
...processing raw files...

=== Step 2: Generate basic synthetic expansions ===
Wrote 20,000 synthetic lines to synthetic_generated.txt

=== Step 3: Advanced data generation ===
Running advanced_synth_expand.py...
✅ advanced_synth_expand.py completed successfully
Running web_scrape_elephants.py...
✅ web_scrape_elephants.py completed successfully
...

=== Step 4: Combine all available data sources ===
📊 Combining 5 data sources:
  1. elephant_human_90_10_corpus.txt (1,234 lines, 0.5MB)
  2. synthetic_generated.txt (20,000 lines, 2.1MB)
  3. advanced_synthetic_20241201_143022.txt (10,000 lines, 1.8MB)
  ...

📊 Deduplication results:
   Original lines: 45,678
   Unique lines: 41,234
   Final filtered: 39,876
   Reduction: 12.7%

🎉 Enhanced pipeline completed successfully!
```

## Troubleshooting

### Common Issues

**Missing environment variable:**
```
KeyError: 'GLOBAL_DATASETS_DIR'
```
Solution: Set the environment variable before running

**No raw data found:**
```
⚠️ Base corpus not found
```
Solution: Ensure `.txt` files exist in the raw directory

**Script execution errors:**
```
⚠️ advanced_synth_expand.py failed: ...
```
Solution: Check script dependencies and permissions

**Empty output files:**
```
❌ No data files found to combine!
```
Solution: Verify raw data exists and base corpus generation succeeds

### Debug Mode
Add print statements or modify the `run_script_if_exists()` function to see detailed subprocess output.

## Performance Considerations

- **Large datasets**: The pipeline handles files of any size, but memory usage scales with deduplication
- **Synthetic generation**: Can be time-consuming; adjust `max_lines` parameter
- **Web scraping**: Respectful rate limiting may slow down data collection
- **Parallel processing**: Scripts run sequentially; consider parallel execution for large-scale data generation

## Integration with Training

After the pipeline completes, use the generated splits for model training:

```bash
# Train with the enhanced dataset
python src/llmlib/cli/modern_gpt_train_cli.py \
    --train_data "$GLOBAL_DATASETS_DIR/llm/mixed_text/out/train.txt" \
    --val_data "$GLOBAL_DATASETS_DIR/llm/mixed_text/out/val.txt" \
    --config "experiments/v4/config.toml"
```

The enhanced pipeline ensures your model trains on a diverse, high-quality dataset combining human-written content, synthetic variations, web-sourced material, and augmented examples.
