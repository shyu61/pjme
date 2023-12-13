import datetime
from pathlib import Path

import click
import polars as pl
from lightgbm import LGBMRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import mean_absolute_error

DATA_DIR = Path(__file__).parent.parent / "data"


def feature_engineering(
    df: pl.DataFrame, add_holiday_feats: bool, add_temerature_feats: bool
) -> pl.DataFrame:
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
    if add_holiday_feats:
        df = __add_holiday(df)
    if add_temerature_feats:
        df = __add_temperature(df)
    return df


def __add_holiday(df: pl.DataFrame) -> pl.DataFrame:
    cal = calendar()
    holidays = cal.holidays(start=df["date"].min(), end=df["date"].max())
    holidays_sr = pl.Series(holidays).cast(pl.Date)
    df = df.with_columns(
        pl.col("date")
        .cast(pl.Date)
        .is_in(holidays_sr)
        .cast(pl.UInt8)
        .alias("is_holiday"),
    )
    return df


def __add_temperature(df: pl.DataFrame) -> pl.DataFrame:
    temperature_df = pl.read_csv(
        DATA_DIR / "temperature" / "chicago.csv",
        dtypes={"high": pl.Float32, "low": pl.Float32},
    )
    temperature_df = temperature_df.with_columns(
        pl.col("DATE").str.to_datetime(time_unit="ns").alias("date")
    ).drop("NAME", "DATE")
    df = df.join(temperature_df, on="date", how="left")
    df = df.with_columns(
        pl.col("high").forward_fill(),
        pl.col("low").forward_fill(),
    )
    return df


@click.command()
@click.option("--add-holiday-feats", "-h", is_flag=True)
@click.option("--add-temerature-feats", "-t", is_flag=True)
def main(add_holiday_feats: bool, add_temerature_feats: bool):
    df = pl.read_csv(DATA_DIR / "PJME_hourly.csv")
    df = df.with_columns(pl.col("Datetime").str.to_datetime(time_unit="ns")).rename(
        {"Datetime": "date"}
    )
    df = feature_engineering(df, add_holiday_feats, add_temerature_feats)

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
    model.fit(train.drop("date", "PJME_MW").to_pandas(), train["PJME_MW"].to_pandas())

    pred = model.predict(test.drop("date", "PJME_MW"))
    mae = mean_absolute_error(y_true=test["PJME_MW"], y_pred=pred)
    print(f"{mae:.5f}")


if __name__ == "__main__":
    main()

# baseline: 2948.49838
# --add-holiday-feats: 2939.00323
# --add-temerature-feats: 2837.51423
