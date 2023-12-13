import datetime
from pathlib import Path

import polars as pl
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    df = pl.read_csv(DATA_DIR / "PJME_hourly.csv")
    df = df.with_columns(pl.col("Datetime").str.to_datetime(time_unit="ns")).rename(
        {"Datetime": "ds", "PJME_MW": "y"}
    )
    # train/test split
    # 2015/1/1以降をテストデータとする
    split_date = datetime.datetime(2015, 1, 1)
    train = df.filter(pl.col("ds") <= split_date)
    test = df.filter(pl.col("ds") > split_date)

    model = Prophet(yearly_seasonality=2, weekly_seasonality=1)
    model.fit(train.to_pandas())
    mae = mean_absolute_error(
        y_true=test["y"], y_pred=model.predict(test.to_pandas())["yhat"]
    )
    print(f"{mae:.5f}")


if __name__ == "__main__":
    main()

# 5100.53516
