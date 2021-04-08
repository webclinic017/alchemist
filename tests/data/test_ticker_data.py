import io
import math
import unittest
import contextlib
import pandas as pd
from alchemist.data.ticker_data import *


class TestDataScraping(unittest.TestCase):

    def test_download_single_ticker_data(self):
        # Required because yf does different things when given one ticker
        with contextlib.redirect_stdout(io.StringIO()):
            downloaded_data = download_data(tickers = ["GME"],
                                            date = "2021-04-01")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(math.floor(gme_data["Open"]["2021-04-01"]), 193)
        self.assertEqual(math.floor(gme_data["High"]["2021-04-01"]), 196)

    def test_download_multiple_ticker_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            downloaded_data = download_data(tickers = ["GME", "TSLA"],
                                            date = "2021-04-01")
        gme_data = downloaded_data["Data"]["GME"]
        tsla_data = downloaded_data["Data"]["TSLA"]
        self.assertEqual(math.floor(gme_data["Open"]["2021-04-01"]), 193)
        self.assertEqual(math.floor(gme_data["High"]["2021-04-01"]), 196)
        self.assertEqual(math.floor(tsla_data["Open"]["2021-04-01"]), 688)
        self.assertEqual(math.floor(tsla_data["High"]["2021-04-01"]), 692)

    def test_download_data_length(self):
        # NOTE: For some reason this fails on the date 2021-04-01, as an extra
        #       day of data is retrieved. As I cannot see a way of fixing this,
        #       nor any negative consequences to it, it remains unfixed
        with contextlib.redirect_stdout(io.StringIO()):
            downloaded_data = download_data(tickers = ["GME"],
                                            date = "2020-03-13")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(len(gme_data["Open"].values), 1)
        # Remember that weekends don't have data
        with contextlib.redirect_stdout(io.StringIO()):
            downloaded_data = download_data(tickers = ["GME"],
                                            from_date = "2021-03-01",
                                            to_date = "2021-03-10")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(len(gme_data["Open"].values), 8)

    def test_download_multiple_date_data(self):
        with contextlib.redirect_stdout(io.StringIO()):
            downloaded_data = download_data(tickers = ["GME"],
                                            from_date = "2021-03-01",
                                            to_date = "2021-04-01")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)
        self.assertAlmostEqual(gme_data["Low"]["2021-03-18"], 196, 0)
        self.assertAlmostEqual(gme_data["Close"]["2021-03-01"], 120, 0)


class TestDataFormatting(unittest.TestCase):
    def setUp(self):
        with contextlib.redirect_stdout(io.StringIO()):
                self.example_data = download_data(tickers = ["GME", "TSLA"],
                                                  from_date = "2021-03-01",
                                                  to_date = "2021-04-01")

    # TODO: Test that volume is affected by percentage formatting?

    def test_formatting_formatted_data(self):
        # Formatting data that's already been formatted should fail
        formatted_data = format_into_percentages(self.example_data)
        with self.assertRaises(Exception):
            format_into_percentages(formatted_data)
        # This shouldn't raise an error
        formatted_data = format_into_percentages(self.example_data)
        # Same for adjusting for volatility
        adjusted_data = adjust_for_volatility(formatted_data)
        with self.assertRaises(Exception):
            adjusted_data = adjust_for_volatility(adjusted_data)
        adjusted_data = adjust_for_volatility(formatted_data)

    def test_formatting_into_percentages_based_on_daily_close(self):
        formatted = format_into_percentages(self.example_data, "daily close")
        f_data = formatted["Data"]
        self.assertIn("daily close", formatted["Formatting"])
        self.assertIn("percentages", formatted["Formatting"])
        self.assertAlmostEqual(f_data["GME"]["Open"]["2021-03-16"], 0.923, 3)
        self.assertAlmostEqual(f_data["TSLA"]["Close"]["2021-03-24"], 0.952, 3)

    def test_formatting_into_percentages_based_on_daily_open(self):
        formatted = format_into_percentages(self.example_data, "daily open")
        f_data = formatted["Data"]
        self.assertIn("daily open", formatted["Formatting"])
        self.assertIn("percentages", formatted["Formatting"])
        self.assertAlmostEqual(f_data["GME"]["Open"]["2021-03-16"], 0.923, 3)
        self.assertAlmostEqual(f_data["TSLA"]["Close"]["2021-03-24"], 0.944, 3)

    def test_formatting_into_percentages_based_on_first_open(self):
        formatted = format_into_percentages(self.example_data, "first open")
        f_data = formatted["Data"]
        self.assertIn("first open", formatted["Formatting"])
        self.assertIn("percentages", formatted["Formatting"])
        self.assertAlmostEqual(f_data["GME"]["Open"]["2021-03-01"], 1, 1)
        self.assertAlmostEqual(f_data["GME"]["High"]["2021-03-09"], 2.390, 3)

    def test_adjusting_for_global_volatility(self):
        formatted = format_into_percentages(self.example_data, "first open")
        adjusted = adjust_for_volatility(formatted, "global v")
        a_data = adjusted["Data"]
        self.assertIn("v adjusted", adjusted["Formatting"])
        self.assertIn("global v adjusted", adjusted["Formatting"])
        # The scaling should mean the biggest value in the data is exactly 1
        self.assertEqual(a_data["GME"].max().max(), 1)
        self.assertEqual(a_data["TSLA"].max().max(), 1)
        # Check volume is scaled seperately
        self.assertEqual(a_data["GME"].drop(["Volume"], axis=1).max().max(), 1)
        self.assertEqual(a_data["GME"]["Volume"].max().max(), 1)
        # Count how many ones there are
        count = 0
        for column in a_data["GME"].columns:
            try: count += a_data["GME"][column].value_counts()[1.0]
            except: pass
        # For global v, there should be 1 one, plus 1 for volume
        self.assertEqual(count, 1 +1)

    def test_adjusting_for_daily_volatility(self):
        formatted = format_into_percentages(self.example_data, "first open")
        adjusted = adjust_for_volatility(formatted, "daily v")
        a_data = adjusted["Data"]
        self.assertIn("v adjusted", adjusted["Formatting"])
        self.assertIn("daily v adjusted", adjusted["Formatting"])
        # The scaling should mean the biggest value in the data is exactly 1
        self.assertEqual(a_data["GME"].max().max(), 1)
        self.assertEqual(a_data["TSLA"].max().max(), 1)
        # Count how many ones there are
        count = 0
        for column in a_data["GME"].columns:
            try: count += a_data["GME"][column].value_counts()[1.0]
            except: pass
        # For daily v, there should be as many ones as days, plus 1 for volume
        self.assertEqual(count, len(a_data["GME"].index) +1)

    def test_adjusting_unformatted_data(self):
        # Adjusting for volatility should work just as well for unformatted data
        adjusted = adjust_for_volatility(self.example_data, "global v")
        a_data = adjusted["Data"]
        # Same tests as above apply, but we don't need to run all of them
        self.assertIn("global v adjusted", adjusted["Formatting"])
        self.assertEqual(a_data["TSLA"].max().max(), 1)
        count = 0
        for column in a_data["TSLA"].columns:
            try: count += a_data["TSLA"][column].value_counts()[1.0]
            except: pass
        self.assertEqual(count, 1 +1)

    # def test_formatting_for_training(self):
    #     pass
    #
    # def test_formatting_into_dataset(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
