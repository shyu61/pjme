import datetime
from pathlib import Path
from lightgbm import LGBMRegressor
import polars as pl
from sklearn.metrics import mean_squared_error

DATA_DIR = Path(__file__).parent.parent / "data"


def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("date").alias("date"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.week().alias("week_of_year"),
        pl.col("date").dt.hour().alias("hour"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.day().alias("day_of_month"),
        pl.col("date").dt.ordinal_day().alias("day_of_year"),
    )
    return df


def main():
    df = pl.read_csv(DATA_DIR / "PJME_hourly.csv")
    df = df.with_columns(pl.col("Datetime").str.to_datetime()).rename(
        {"Datetime": "date"}
    )
    df = feature_engineering(df)

    # train/test split
    # 2015/1/1以降をテストデータとする
    split_date = datetime.datetime(2015, 1, 1)
    train = df.filter(pl.col("date") <= split_date)
    test = df.filter(pl.col("date") > split_date)

    lgb_params = {
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
    }
    model = LGBMRegressor(**lgb_params)
    model.fit(train.drop("date", "PJME_MW"), train["PJME_MW"])

    pred = model.predict(test.drop("date", "PJME_MW"))
    output = mean_squared_error(y_true=test["PJME_MW"], y_pred=pred, squared=False)
    print(f"{output:.5f}")  # => 3896.71370


if __name__ == "__main__":
    main()
