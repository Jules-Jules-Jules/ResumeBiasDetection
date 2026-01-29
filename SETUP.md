# Setup Instructions

## Environment Setup

1. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run smoke test**
   ```bash
   ./run.sh smoke
   ```

## Verify Installation

```bash
# Check that all paths exist and libraries are installed
python -m src.cli info

# Run tests
pytest
```
You should see confirmation that all required directories exist and libraries are properly installed.

## How to Run Everything

The project is organized into phases. You can run everything at once or each phase separately:

```bash
# run everything (all phases from start to finish - this should take around 5 minutes)
./run.sh all

# to see all available commands
./run.sh

# or run each phase individually:
./run.sh phase1 # data processing
./run.sh phase2 # scoring & evaluation  
./run.sh phase3 # bias audit
./run.sh phase4 # train + eval both models
./run.sh phase5 # mitigation experiments

# or run specific parts:
./run.sh phase4-train # just train classifiers
./run.sh phase4-eval # just eval classifiers
./run.sh test # run tests
./run.sh smoke # smoke test
./run.sh info # system info
```

## Analyzing Results

After running `./run.sh all`, you should open and run the Jupyter notebooks to analyze the results:

1. `notebooks/01_phase1_data_build.ipynb` - data processing overview
2. `notebooks/02_phase2_retrieval_eval.ipynb` - retrieval performance analysis
3. `notebooks/03_phase3_bias_audit.ipynb` - bias detection results
4. `notebooks/04_phase4_classifier_head.ipynb` - classifier training analysis
5. `notebooks/05_phase5_mitigations.ipynb` - bias mitigation effectiveness

To open the notebooks:
```bash
jupyter notebook notebooks/
```

**Run all cells in each notebook** for analysis and reporting. A copy of this report is also available in `reports/`.

### If you run out of memory

- make the batch size smaller in `src/config.py`
- close other programs

## Updating packages

```bash
pip install --upgrade -r requirements.txt
```

## Changing Settings
Edit `src/config.py` to change:
- where files are saved
- train/test split ratios
- random seed
- model settings
- bias thresholds


