import io
import pickle
import datetime
import contextlib
import numpy as np
import pandas as pd
import yfinance as yf
from copy import deepcopy
from datetime import datetime as dt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from alchemist.data.data_utils import *


def download_data(tickers, date = None, from_date = None, to_date = None):
    if date != None:
        from_date = date
        to_date = date
    # Add 1 day to the to_date for inclusivity
    if type(to_date) == str:
        to_date = dt.fromisoformat(to_date)
    to_date += datetime.timedelta(days=1)
    # Download data, silently
    with contextlib.redirect_stdout(io.StringIO()):
        raw_data = yf.download(tickers, start = from_date, end = to_date)
    no_null_data = raw_data.dropna(how="any", axis=0)
    no_duplicate_data = no_null_data[~no_null_data.index.duplicated(keep='first')]
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


def get_data(path, tickers, date = None, from_date = None, to_date = None,
             backup = False):
    if date != None:
        from_date = date
        to_date = date
        # A slightly hacky fix, but shouldn't break anything
        if type(to_date) == str:
            to_date = dt.fromisoformat(to_date)
        to_date += datetime.timedelta(days=1)

    try:
        loaded_data = load_data(path)
        needed_data = {"Formatting" : loaded_data["Formatting"],
                        "Data" : {}}
        # Check that the data matches the dates needed;
        for ticker in tickers:
            try:
                needed_data["Data"][ticker] = loaded_data["Data"][ticker]
                index_list = list(needed_data["Data"][ticker].index)
                # Download data after the stored date if needed
                # NOTE: from_date and to_date must be unchanged for multiple tickers
                f_date, sf_date = squeeze_dates(from_date, index_list[0])
                if f_date != None:
                    before_data = download_data([ticker], from_date = f_date,
                                                to_date = sf_date)
                    before_data["Data"][ticker] = before_data["Data"][ticker].iloc[2:]
                    needed_data["Data"][ticker] = pd.concat([before_data["Data"][ticker], 
                                                     needed_data["Data"][ticker]])
                # Download data before the stored date if needed
                st_date, t_date = squeeze_dates(index_list[-1], to_date)
                if t_date != None:
                    after_data = download_data([ticker], from_date = st_date,
                                               to_date = t_date)
                    after_data["Data"][ticker] = after_data["Data"][ticker].iloc[2:]
                    needed_data["Data"][ticker] = pd.concat([needed_data["Data"][ticker], 
                                                     after_data["Data"][ticker]])
            except KeyError:
                needed_data["Data"][ticker] = download_data([ticker], from_date, 
                                            to_date)["Data"][ticker]

            # data["Data"][ticker] = data["Data"][ticker].dropna(how="any", axis=0)

        # Save the data
        save_data(needed_data, path, backup = backup)

    except FileNotFoundError:
        # If no save file exists, download new data
        needed_data = download_data(tickers = tickers, from_date = from_date, 
                             to_date = to_date)
        save_data(needed_data, path, backup = backup)

    # data = data["Data"].pop(t for t in data["Data"].keys() not in tickers)

    return needed_data


def format_into_percentages(data, formatting_basis = "first open"):
    # Do not format already formtted data
    if len(data["Formatting"]) != 0:
        raise Exception("This data has already been formatted")
    # Make sure we're not changing the original data
    formatted_data = deepcopy(data)
    formatted_data["Formatting"] += [formatting_basis, "percentages"]

    for ticker in formatted_data["Data"].keys():
        df_ohlc = formatted_data["Data"][ticker].drop(["Volume"], axis=1)
        df_volume = formatted_data["Data"][ticker]["Volume"].copy()

        if formatting_basis == "first open":
            first_date = df_ohlc.index[0]
            first_open = df_ohlc["Open"][first_date]
            for column, datapoint in df_ohlc.items():
                datapoint /= first_open
        # TODO: daily close and daily open are similar, should they be combined?
        elif formatting_basis == "daily close":
            # WARNING: This cuts off the very first datapoint
            new_df = df_ohlc.copy() # TODO: This may be could be optimized
            dates = df_ohlc.index
            for date_index in range(len(dates)):
                last_close = df_ohlc["Close"][dates[date_index - 1]]
                for column in new_df.columns:
                    new_df.loc[dates[date_index], column] /= last_close
            df_ohlc = new_df.iloc[1:]

        elif formatting_basis == "daily open":
            # WARNING: This cuts off the very first datapoint
            new_df = df_ohlc.copy() # TODO: this may be could be optimized
            dates = df_ohlc.index
            for date_index in range(len(dates)):
                last_close = df_ohlc["Close"][dates[date_index - 1]]
                latest_open = df_ohlc["Open"][dates[date_index]]
                for column in new_df.columns:
                    new_df.loc[dates[date_index], column] /= latest_open if (
                        column != "Open"
                    ) else last_close
            df_ohlc = new_df.iloc[1:]
        else:
            raise Exception("Invalid formatting basis given")
        # Format volume column seperately
        dates = df_volume.index
        new_df_volume = df_volume.copy()
        for date_index in range(len(dates)):
            last_vol = df_volume[dates[date_index - 1]]
            new_df_volume.iloc[date_index] /= last_vol
        df_volume = new_df_volume

        new_df = df_ohlc
        new_df["Volume"] = df_volume
        formatted_data["Data"][ticker] = new_df

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


def format_into_xy(data, label_var = "Close", num_features = 1, label_type = "float",
        label_type_vars = {"divider" : 0, "balance" : False}):
    x_data = []
    y_data = []

    for ticker in data["Data"].keys():
        df = data["Data"][ticker]
        for date_index in range(len(df.index))[num_features-1:]:
            x_data.append([l.tolist() for l in df.iloc[
                date_index-num_features : date_index
            ].values])
            y_data.append(df.loc[df.index[date_index], [label_var]].values[0])

    # Remove empty values
    while [] in x_data:
        i = x_data.index([])
        del y_data[i]
        x_data.remove([])

    # Labels may need to be changed for classification etc.
    # TODO: Storing label vars in a dict like this may be inadvisable, maybe fix
    if label_type == "bin":
        # "bin" for binary classification
        divider = label_type_vars["divider"]
        for i in range(len(y_data)):
            y_data[i] = 1 if y_data[i] >= divider else 0
        try: balance = label_type_vars["balance"]
        except: balance = False
        if balance:
            while y_data.count(1) > y_data.count(0):
                bad_index = y_data.index(1)
                y_data.pop(bad_index)
                x_data.pop(bad_index)
            while y_data.count(1) < y_data.count(0):
                bad_index = y_data.index(0)
                y_data.pop(bad_index)
                x_data.pop(bad_index)
    
    return x_data, y_data


class TickerDataset (Dataset):
    # TickerDataset interacts correctly with data_loader
    def __init__(self, x_data, y_data):
        # Set parameters and things
        self.x_data = x_data
        self.y_data = y_data
        self.length = len(self.x_data)

    def __getitem__(self, index):
        return np.array([self.x_data[index]]), self.y_data[index]

    def __len__(self):
        return self.length


# Returns a train and test TickerDataset; feels a bit wrong because
# it acts kinda like a class constructor, but should be fine really
def train_test_datasets(x_data, y_data, train_size = 0.8):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                            train_size = train_size)
    train_dataset = TickerDataset(x_train, y_train)
    test_dataset = TickerDataset(x_test, y_test)
    return train_dataset, test_dataset


def squeeze_dates(from_date, to_date):
    # NOTE: These are NASDAQ holidays, last updated 2021-06-18
    holidays = ["01-01", "01-18", "02-15", "04-02", "05-31",
                "07-05", "08-06", "11-25", "12-24"]
    
    if type(from_date) == str:
        from_date = dt.fromisoformat(from_date)
    if type(to_date) == str:
        to_date = dt.fromisoformat(to_date)
    
    # print(from_date, to_date)

    while from_date.weekday() > 4 or str(from_date)[5:10] in holidays:
        from_date += datetime.timedelta(days = 1)
    while to_date.weekday() > 4 or str(to_date)[5:10] in holidays:
        to_date -= datetime.timedelta(days = 1)

    if (to_date - from_date).days <= 0:
        return None, None

    return from_date, to_date


