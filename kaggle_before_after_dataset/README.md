# AI vs Human Voice Dataset (Before/After Split)

## Unique Structure
This dataset is organized in a unique BEFORE/AFTER training structure:

### BEFORE_TRAIN/
Contains all 200 samples in raw format:
- `ai/` - 100 AI-generated voice samples
- `human/` - 100 Human-like voice samples

### AFTER_TRAIN/
Contains the same data split for ML:
- `train/` - 140 samples (70 AI + 70 Human)
- `val/` - 30 samples (15 AI + 15 Human)
- `test/` - 30 samples (15 AI + 15 Human)

## Perfect for Teaching
This structure helps students understand:
1. Raw data organization
2. Train/validation/test splitting
3. Complete ML pipeline
