import os
import pickle
import datetime
import pandas as pd
import yfinance as yf
from copy import deepcopy


def download_data(tickers, date = None, from_date = None, to_date = None):
    if date != None:
        from_date = date
        to_date = date
    # Add 1 day to the to_date for inclusivity
    to_date = datetime.datetime.fromisoformat(to_date) + datetime.timedelta(days=1)
    raw_data = yf.download(tickers, start = from_date, end = to_date)
    no_duplicate_data = raw_data[~raw_data.index.duplicated(keep='first')]
    # Rearrange data into a more standardized and accessible form
    if len(tickers) == 1:
        rearranged_data = {"Data" : {tickers[0] : no_duplicate_data}}
    else:
        # The data is under "Data" to seperate it from things like "Formatting"
        rearranged_data = {ticker : pd.DataFrame() for ticker in tickers}
        for column in no_duplicate_data.columns:
            rearranged_data[column[1]][column[0]] = no_duplicate_data[column]
        rearranged_data = {"Data" : rearranged_data}
    # Mostly for the human and to solve issues easier
    rearranged_data["Formatting"] = []

    # print(rearranged_data.keys())

    return rearranged_data


def format_into_percentages(data, formatting_basis = "first open"):
    # Do not format already formtted data
    if len(data["Formatting"]) != 0:
        raise Exception("This data has already been formatted")
    # Make sure we're not changing the original data
    formatted_data = deepcopy(data)
    formatted_data["Formatting"] += [formatting_basis, "percentages"]

    for ticker in formatted_data["Data"].keys():
        df = formatted_data["Data"][ticker]

        if formatting_basis == "first open":
            first_date = df.index[0]
            first_open = df["Open"][first_date]
            for column, datapoint in df.items():
                datapoint /= first_open
        # TODO: daily close and daily open are similar, should they be combined?
        elif formatting_basis == "daily close":
            # WARNING: This cuts off the very first datapoint
            new_df = df.copy() # TODO: This may be could be optimized
            dates = df.index
            for date_index in range(len(dates)):
                last_close = df["Close"][dates[date_index - 1]]
                for column in new_df.columns:
                    new_df.loc[dates[date_index], column] /= last_close
            formatted_data["Data"][ticker] = new_df.iloc[1:]

        elif formatting_basis == "daily open":
            # WARNING: This cuts off the very first datapoint
            new_df = df.copy() # TODO: this may be could be optimized
            dates = df.index
            for date_index in range(len(dates)):
                last_close = df["Close"][dates[date_index - 1]]
                latest_open = df["Open"][dates[date_index]]
                for column in new_df.columns:
                    new_df.loc[dates[date_index], column] /= latest_open if (
                        column != "Open"
                    ) else last_close
            formatted_data["Data"][ticker] = new_df.iloc[1:]
        else:
            raise Exception("Invalid formatting basis given")

    return formatted_data

def adjust_for_volatility(data, volatility_type = "global v"):
    if "v adjusted" in data["Formatting"]:
        raise Exception("The volatility of this data has already been adjusted")
    # Make sure we're not changing the original data
    adjusted_data = deepcopy(data)
    adjusted_data["Formatting"] += ["v adjusted", volatility_type + " adjusted"]

    for ticker in adjusted_data["Data"].keys():
        df = adjusted_data["Data"][ticker]
        df_ohlc = df.drop(["Volume"], axis=1)
        # Slightly different thing based on volatility type being adjusted for
        if volatility_type == "global v":
            global_max_value = df_ohlc.max().max()
            for column, datapoint in df_ohlc.items():
                datapoint /= global_max_value
        elif volatility_type == "daily v":
            dates = df_ohlc.index
            for date_index in range(len(dates)):
                day = df_ohlc.iloc[date_index]
                day_max_value = day.max()
                for column in df_ohlc:
                    df_ohlc.loc[dates[date_index]][column] /= day_max_value
        else:
            raise Exception("Invalid volatility type provided")
        # Volume is adjusted seperately to the rest
        df_volume = df["Volume"].copy()
        max_volume = df_volume.max()
        for index in df_volume.index:
            df_volume.loc[index] /= max_volume
        new_df = df_ohlc
        new_df["Volume"] = df_volume
        adjusted_data["Data"][ticker] = new_df

    return adjusted_data
