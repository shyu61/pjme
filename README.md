# pjme

## Dependencies
- `rye` for package manager
  - [Installation](https://rye-up.com/guide/installation/)
- python3.10.13

## Setup
1. Install dependencies and create virtual environment.
```bash
rye sync
```
2. download data to `./data/`
  - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
  - baseline notebook of this dataset
    - https://www.kaggle.com/code/robikscube/time-series-forecasting-with-prophet

## Usage
```bash
# lightgbm
python -m src.lgb --add-holiday-feats --add-temerature-feats

# prophet
python -m src.prophet
```
