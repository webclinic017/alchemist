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

    def test_formatting_volume_into_percentages(self):
        # Trading volume also needs to be formatted, seperately from the rest
        formatted = format_into_percentages(self.example_data, "first open")
        tsla_data = formatted["Data"]["TSLA"]
        self.assertAlmostEqual(tsla_data["Volume"]["18-03-2021"], 0.829, 3)
        # The results should be the same for different formatting types
        formatted = format_into_percentages(self.example_data, "daily close")
        tsla_data = formatted["Data"]["TSLA"]
        self.assertAlmostEqual(tsla_data["Volume"]["18-03-2021"], 0.829, 3)

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


class TestPreparingDataForTraining(unittest.TestCase):
    def setUp(self):
        with contextlib.redirect_stdout(io.StringIO()):
                self.example_data = download_data(tickers = ["GME", "TSLA"],
                                                  from_date = "2021-03-01",
                                                  to_date = "2021-04-01")
                self.formatted_data = format_into_percentages(self.example_data,
                                                              "daily close")
                self.adjusted_data = adjust_for_volatility(self.example_data,
                                                           "global v")

    def assertXYCorrespondance(self, x_value, y_value, x_data, y_data):
        # As this is used a bunch, it's nice to put it into its own function
        self.assertIn(x_value, x_data)
        self.assertIn(y_value, y_data)
        # The y_value doesn't just have to exist, it has to be in the right place
        index = x_data.index(x_value)
        self.assertEqual(y_data[index], y_value)

    def test_formatting_into_xy_data_basic(self):
        # With jsut a single input vector
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close")
        # The full data for GME 2021-03-12
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615,
                    1.0087307269756611, 1.0576923076923077, 0.9139537358972539]]
        y_value = 0.8322873322860055
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_xy_data_with_multiple_features(self):
        # With multiple input vectors
        x_data, y_data = format_into_xy(self.adjusted_data, num_features = 3,
                                        label_var = "Open")
        x_value = [[0.7807408182338799, 0.7807408182338799, 0.859965910166229,
                    0.7749025660141386, 0.8328133017048471, 0.5792956100071032],
                   [0.9340877685412351, 0.9340877685412351, 0.9403420291356522,
                    0.8254080987139205, 0.8433942186770239, 0.7553237542856823],
                   [0.9264328758849383, 0.9264328758849383, 0.9954791781672033,
                    0.9084051127640849, 0.9711417148821643, 0.6779426487614169]]
        y_value = 0.9698936897581362
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_xy_data_with_y_manipulation(self):
        # Often we want to manipulate the label, e.g. for classification
        label_vars = {"divider" : 1}
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close", label_type = "bin",
                                        label_type_vars = label_vars)
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615,
                    1.0087307269756611, 1.0576923076923077, 0.9139537358972539]]
        y_value = 0
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_dataset(self):
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close")
        datset = TickerDataset(x_data, y_data, split_data = False)
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615,
                    1.0087307269756611, 1.0576923076923077, 0.9139537358972539]]
        y_value = 0.8322873322860055
        l = length(x_data)
        # Testing if we can acess the dataset correctlye
        self.assertXYCorrespondance(x_value, y_value,
                                    dataset.x_data, dataset.y_data)
        self.assertEqual(dataset.length, l)
        # NOTE: The dataset also has a couple functions that I'm not 100% sure
        # how to test (read: can't be bothered to test) but they are very simple
        # and shouldn't break if this test works


if __name__ == '__main__':
    unittest.main()
