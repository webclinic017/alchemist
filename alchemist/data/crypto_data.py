import io
import math  
import contextlib
import pandas as pd
import yfinance as yf


class CryptoData():

    def __init__(self, pairs=None, from_date=None, to_date=None):
        if pairs != None:
            self.download_data(pairs, from_date, to_date)

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


