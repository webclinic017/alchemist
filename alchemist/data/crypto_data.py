import io
import math  
import contextlib
import pandas as pd
import yfinance as yf
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CryptoData():

    def __init__(self, pairs=None, from_date=None, to_date=None,
                 adjust_volatility=None, n_features=1, train_fraction=1,
                 backtest_dataset=False, divider=None, balance=False):
        # All necessary operations can be called by specifying variables
        # on initialisation
        if pairs != None:
            self.download_data(pairs, from_date, to_date)
        if adjust_volatility != None:
            self.format_data_into_percentages()
            self.generate_datasets(adjust_volatility, n_features, 
                                   train_fraction, backtest_dataset,
                                   divider, balance)

    def download_data(self, pairs, from_date, to_date):
        # Download data, silently
        with contextlib.redirect_stdout(io.StringIO()):
            new_raw_data = yf.download(pairs, start=from_date, end=to_date)
        # If there was only one pair, stack the columns like with multiple
        if len(pairs) == 1:
            multicol = pd.MultiIndex.from_tuples(
                    [(c, pairs[0]) for c in new_raw_data.columns])
            new_raw_data.columns = multicol
        # If we already have loaded data, combine it with the dowloaded data
        try:
            raw_data = self.raw_data.combine_first(new_raw_data)
            # Fill in missing days with nan if needed
            idx = pd.date_range(raw_data.index[0], raw_data.index[-1])
            self.raw_data = raw_data.reindex(idx, fill_value=math.nan) if (
                    len(idx) > len(raw_data.index)) else raw_data
        # Otherwise just set the raw data
        except AttributeError:
            self.raw_data = new_raw_data

    def format_data_into_percentages(self):
        self.percentage_data = self.raw_data.copy()
        for column, content in self.raw_data.items():
            last_index = None
            for index, item in content.items():
                try:
                    self.percentage_data.loc[index, column] = (
                        item / self.raw_data.loc[last_index, (
                            ("Volume" if column[0] == "Volume" else "Close"), 
                            column[1])])
                except:
                    self.percentage_data.loc[index, column] = math.nan
                last_index = index

    def generate_datasets(self, adjust_volatility=False, n_features=1,
                          train_fraction=1, backtest_dataset=False,
                          divider=None, balance=False):
        x_data = []
        y_data = []
        index_list = []
        
        # NOTE: For backtest_ds keep track of dates/indeces?
        # Arrange percentage data into x and y data
        reordered_df = self.percentage_data.reorder_levels([1, 0], 1)
        pairs = set(reordered_df.columns.get_level_values(0))
        for pair in pairs:
            relevant_df = reordered_df[pair]
            for date_index, date in enumerate(relevant_df.index, n_features-1):
                x_data.append([l.tolist() for l in relevant_df.iloc[
                    date_index-n_features : date_index].values])
                y_data.append(relevant_df.loc[date, ["Close"]].values[0])
                index_list.append(date_index)

        # Remove nan values from x and y data
        nan_indexes = []
        for i, x in enumerate(x_data):
            if math.isnan(y_data[i]):
                nan_indexes.append(i)
            for x_ in x:
                if math.isnan(x_[0]):
                    nan_indexes.append(i)
        nan_set = set(nan_indexes)
        x_data = [x for i, x in enumerate(x_data) if i not in nan_set]
        y_data = [y for i, y in enumerate(y_data) if i not in nan_set]
        index_list = [y for i, y in enumerate(index_list) if i not in nan_set]

        # Turn labels into 1's and 0's based on divider
        if divider != None:
            y_data = [(1 if y >= divider else 0) for y in y_data]

        # Balance Data, though this isn't the best solution to imbalanced data
        if balance:
            while y_data.count(1) > y_data.count(0):
                bad_index = y_data.index(1)
                y_data.pop(bad_index)
                x_data.pop(bad_index)
            while y_data.count(1) < y_data.count(0):
                bad_index = y_data.index(0)
                y_data.pop(bad_index)
                x_data.pop(bad_index)

        # Adjust every set of features to be between 0 and 1
        # NOTE: Maybe volume should be done seperately?
        if adjust_volatility:
            for i, x in enumerate(x_data):
                max_x = 0
                for _x in x:
                    max_x = max(_x) if max(_x) > max_x else max_x
                x_data[i] = [[__x/max_x for __x in _x] for _x in x]

        # Create datasets
        if backtest_dataset:
            # Rearrange the data to be clumps of days
            x_data = [[x_data[
                    [ind for ind, v in enumerate(index_list) if v == j][i]] 
                for i in range(index_list.count(j))] for j in set(index_list)]
            y_data = [[y_data[
                    [ind for ind, v in enumerate(index_list) if v == j][i]] 
                for i in range(index_list.count(j))] for j in set(index_list)]
            self.backtest_ds = CryptoDataset(x_data, y_data) 
        elif train_fraction == 1:
            self.train_ds = CryptoDataset(x_data, y_data)
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                    x_data, y_data, train_size = train_fraction)
            self.train_ds = CryptoDataset(x_train, y_train)
            self.test_ds = CryptoDataset(x_test, y_test)

class CryptoDataset (Dataset):
    # CryptoDataset interacts correctly with data_loader
    def __init__(self, x_data, y_data):
        # Set parameters and things
        self.x_data = x_data
        self.y_data = y_data
        self.length = len(self.x_data)

    def __getitem__(self, index):
        return np.array([self.x_data[index]]), self.y_data[index]

    def __len__(self):
        return self.length
