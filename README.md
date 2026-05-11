# HFT Backtester

Install dependencies:

pip install -r requirements.txt


## Data

The `data/` folder contains a sample dataset ready to run. To use the full datasets from the assignment, replace the samples with them. The filenames must stay the same.

## Running the strategies

Run 2008 only:

python run_backtest.py --config configs/as_2008.yaml


Run microprice only:

python run_backtest.py --config configs/as_microprice.yaml


Run both and compare:

python run_backtest.py --config configs/as_2008.yaml --compare-with configs/as_microprice.yaml


Results are saved to `results/`.
