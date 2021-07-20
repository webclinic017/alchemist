import os
import math
import unittest
import pandas as pd
import pytest_socket
from alchemist.data.ticker_data import *
# pytest_socket.SocketBlockedError = Exception


class TestDataScraping(unittest.TestCase):

    def test_download_single_ticker_data(self):
        # Required because yf does different things when given one ticker
        downloaded_data = download_data(tickers = ["GME"],
                                        date = "2021-04-01")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(math.floor(gme_data["Open"]["2021-04-01"]), 193)
        self.assertEqual(math.floor(gme_data["High"]["2021-04-01"]), 196)

    def test_download_multiple_ticker_data(self):
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
        downloaded_data = download_data(tickers = ["GME"],
                                        date = "2020-03-13")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(len(gme_data["Open"].values), 1)
        # Remember that weekends don't have data
        downloaded_data = download_data(tickers = ["GME"],
                                        from_date = "2021-03-01",
                                        to_date = "2021-03-10")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(len(gme_data["Open"].values), 8)

    def test_download_multiple_date_data(self):
        downloaded_data = download_data(tickers = ["GME"],
                                        from_date = "2021-03-01",
                                        to_date = "2021-04-01")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)
        self.assertAlmostEqual(gme_data["Low"]["2021-03-18"], 196, 0)
        self.assertAlmostEqual(gme_data["Close"]["2021-03-01"], 120, 0)

    def test_providing_non_string_date_formats(self):
        # To avoid formatting 50 times, and because sometimes we work with
        # dates as datetime objects rather than strings, download_data
        # must at least accept datetime objects as dates
        date = datetime.datetime.fromisoformat("2021-04-01")
        data = download_data(["GME"], date = date)
        gme_data = data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)

    def test_downloading_weekend_data(self):
        downloaded_data = download_data(tickers = ["GME"], date = "2021-05-30")
        gme_data = downloaded_data["Data"]["GME"]
        self.assertEqual(type(gme_data), pd.core.frame.DataFrame)
        self.assertEqual(len(gme_data.index), 0)

    def test_not_download_nan_data(self):
        downloaded_data = download_data(tickers = ["FB"], from_date = "2012-01-01",
                                          to_date = "2013-01-01")
        self.assertFalse(downloaded_data["Data"]["FB"].isnull().values.any())
        # Acts differently with several tickers
        downloaded_data = download_data(tickers = ["GOOG", "FB"], from_date = "2012-01-01",
                                          to_date = "2013-01-01")
        self.assertFalse(downloaded_data["Data"]["FB"].isnull().values.any())

    # NOTE: There's nothing that explicitly handles weekends and holidays;
    # they should just give an empty/nearly empty dataframe, which should be
    # handled fine by the rest of the code.


class TestDataConditionalLoading(unittest.TestCase):
    # get_data only downloads data if it isn't available in the saved file
    # This is useful if we want to access a lot of data, or have no wifi
    
    def test_get_data(self):
        # Test that save files are created, and that data is downloaded
        path = "cache/tests/data/test_get_data"
        if os.path.isfile(path):
            os.remove(path)
        data = get_data(tickers = ["GME"], date = "2021-04-01", path = path)
        self.assertAlmostEqual(data["Data"]["GME"]["Open"]["2021-04-01"], 193, 0)
        self.assertTrue(os.path.isfile(path))
        # Now change and save the data; get_data should now return this
        data["Data"]["GME"].loc["2021-04-01", "Open"] = 100
        save_data(data, path)
        data = get_data(tickers = ["GME"], date = "2021-04-01", path = path)
        self.assertEqual(data["Data"]["GME"]["Open"]["2021-04-01"], 100)

    def test_update_incomplete_data(self):
        path = "cache/tests/data/test_update_incomplete_data"
        if os.path.isfile(path):
            os.remove(path)
        get_data(tickers = ["GME"], date = "2021-04-14", path = path)
        data = get_data(tickers = ["GME"], date = "2021-04-15", path = path)
        self.assertAlmostEqual(data["Data"]["GME"]["Open"]["2021-04-14"], 144, 0)
        self.assertAlmostEqual(data["Data"]["GME"]["Open"]["2021-04-15"], 163, 0)
        
    def test_only_download_needed_data(self):
        path = "cache/tests/data/test_partially_download_data"
        if os.path.isfile(path):
            os.remove(path)
        old_data = get_data(tickers = ["GME"], date = "2021-04-14", path = path)
        old_data["Data"]["GME"].loc["2021-04-14", "Open"] = 100
        save_data(old_data, path)
        data = get_data(tickers = ["GME"], from_date = "2021-04-13", 
                        to_date = "2021-04-15", path = path)
        self.assertEqual(data["Data"]["GME"]["Open"]["2021-04-14"], 100)
        self.assertAlmostEqual(data["Data"]["GME"]["Open"]["2021-04-13"], 142, 0)
        self.assertAlmostEqual(data["Data"]["GME"]["Open"]["2021-04-15"], 163, 0)

    def test_get_data_with_weekend_time(self):
        path = "cache/tests/data/test_partially_download_data"
        if os.path.isfile(path):
            os.remove(path)
        d1 = get_data(path, ["GME"], from_date = "2021-03-29", to_date = "2021-04-01")
        # 2021-04-02 and -03 is a weekend, so no data needs to be downloaded
        # NOTE: as yfinance is multithreaded, this failing actually crashes the code,
        # and doesn't give a nice error message
        pytest_socket.disable_socket()
        d2 = get_data(path, ["GME"], from_date = "2021-03-29", to_date = "2021-04-03")
        pytest_socket.enable_socket()
        self.assertListEqual(list(d1["Data"]["GME"].index), list(d2["Data"]["GME"].index))

    def test_not_getting_nan_data(self):
        path = "cache/tests/data/test_not_get_nan_data"
        if os.path.isfile(path): os.remove(path)
        data = get_data(path = path, tickers = ["GOOG", "FB"], 
                        from_date = "2012-01-01", to_date = "2013-01-01")
        self.assertFalse(data["Data"]["FB"].isnull().values.any())

    def test_only_get_required_data(self):
        path = "cache/tests/data/test_only_get_required_data"
        if os.path.isfile(path): os.remove(path)
        data = get_data(path = path, tickers = ["GOOG", "FB", "AMZN"], 
                        from_date = "2012-01-01", to_date = "2013-01-01")
        new_data = get_data(path = path, tickers = ["GOOG"], 
                            from_date = "2012-05-01", to_date = "2012-09-01")
        self.assertNotIn("FB", new_data["Data"].keys())
        self.assertNotIn("AMZN", new_data["Data"].keys())
        self.assertNotIn("2012-02-02", new_data["Data"]["GOOG"].index)
        self.assertNotIn("2012-11-11", new_data["Data"]["GOOG"].index)

    def test_dont_download_when_less_data_needed(self):
        path = "cache/tests/data/test_dont_donwload_when_less_data_needed"
        if os.path.isfile(path): os.remove(path)
        data = get_data(path = path, tickers = ["GOOG", "AMZN"], 
                        from_date = "2012-01-01", to_date = "2013-01-01")
        pytest_socket.disable_socket()
        new_data = get_data(path = path, tickers = ["GOOG"], 
                        from_date = "2012-02-01", to_date = "2012-07-01")
        pytest_socket.enable_socket()

    def test_squeeze_date_range(self):
        # chops off ends of date ranges if they include holidays or weekends
        from_date = "2021-01-01"
        to_date = "2021-02-07"
        new_from, new_to = squeeze_dates(from_date, to_date)
        new_from = str(new_from)[:10]
        new_to = str(new_to)[:10]
        self.assertEqual(new_from, "2021-01-04")
        self.assertEqual(new_to, "2021-02-05")
        # The dates can be squeezed to nothing
        from_date = "2021-01-01"
        to_date = "2021-01-03"
        new_from, new_to = squeeze_dates(from_date, to_date)
        self.assertIsNone(new_from)
        self.assertIsNone(new_to)

    def test_getting_data_for_new_ticker(self):
        path = "cache/tests/data/test_get_data_unknown_ticker"
        if os.path.isfile(path):
            os.remove(path)
        get_data(path = path, tickers = ["TSLA"], date = "2021-04-01")
        data = get_data(path = path, tickers = ["GME"], date = "2021-04-01")
        gme_data = data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)

    def test_get_data_backup(self):
        path = "cache/tests/data/test_backup_get_data"
        if os.path.isfile(path): os.remove(path)
        if os.path.isfile(path + "_backup"): os.remove(path + "_backup")
        get_data(tickers = ["GME"], date = "2021-04-01", path = path,
                 backup = True)
        self.assertTrue(os.path.isfile(path + "_backup"))
        # Retrieving from backup etc. should just be handled by load_data

    def test_not_needing_to_download_data(self):
        path = "cache/tests/data/test_not_downloading_data"
        if os.path.isfile(path):
            os.remove(path)
        data = get_data(path = path, tickers = ["GME"],
                        from_date = "2021-03-13", to_date = "2021-04-13")
        gme_data = data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)
        # Just repeating the thing should work perfectly fine
        pytest_socket.disable_socket()
        data = get_data(path = path, tickers = ["GME"],
                        from_date = "2021-03-13", to_date = "2021-04-13")
        pytest_socket.enable_socket()
        gme_data = data["Data"]["GME"]
        self.assertAlmostEqual(gme_data["High"]["2021-04-01"], 197, 0)

    def test_getting_multiple_ticker_data(self):
        path = "cache/tests/data/test_get_multiple_ticker_data"
        if os.path.isfile(path):
            os.remove(path)
        data = get_data(path = path, tickers = ["GME", "TSLA"],
                        from_date = "2021-03-13", to_date = "2021-04-13")
        gme_data = data["Data"]["GME"]
        tsla_data = data["Data"]["TSLA"]
        self.assertEqual(math.floor(gme_data["Open"]["2021-04-01"]), 193)
        self.assertEqual(math.floor(gme_data["High"]["2021-04-01"]), 196)
        self.assertEqual(math.floor(tsla_data["Open"]["2021-04-01"]), 688)
        self.assertEqual(math.floor(tsla_data["High"]["2021-04-01"]), 692)
        # Secong time should require no download and work fine
        pytest_socket.disable_socket()
        data = get_data(path = path, tickers = ["GME", "TSLA"],
                        from_date = "2021-03-13", to_date = "2021-04-13")
        pytest_socket.enable_socket()
        gme_data = data["Data"]["GME"]
        tsla_data = data["Data"]["TSLA"]
        self.assertEqual(math.floor(gme_data["High"]["2021-04-01"]), 196)
        self.assertEqual(math.floor(tsla_data["Open"]["2021-04-01"]), 688)


    # NOTE: get_data isn't perfect, and should not be used to store any and
    # all data in one file, especially if you need small patched of data
    # all over the place. With one data file per purpose, it should be fine.


class TestDataFormatting(unittest.TestCase):
    def setUp(self):
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
        self.assertAlmostEqual(tsla_data["Volume"]["18-03-2021"], 0.823, 3)
        # The results should be the same for different formatting types
        formatted = format_into_percentages(self.example_data, "daily close")
        tsla_data = formatted["Data"]["TSLA"]
        self.assertAlmostEqual(tsla_data["Volume"]["18-03-2021"], 0.823, 3)

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
        # Empty values mess things up, there shouldn't be any
        self.assertNotIn([], x_data)
        # The full data for GME 2021-03-12
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615, 
                    1.0087307269756611, 1.0576923076923077, 0.9128794701986755]]
        y_value = 0.8322873322860055
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_xy_data_offset_y(self):
        # With jsut a single input vector
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close", offset=5)
        # Empty values mess things up, there shouldn't be any
        self.assertNotIn([], x_data)
        # The full data for GME 2021-03-12
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615, 
                    1.0087307269756611, 1.0576923076923077, 0.9128794701986755]]
        y_value = 0.9711389691117529
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
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close", label_type = "bin",
                                        divider = 1)
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615, 
                    1.0087307269756611, 1.0576923076923077, 0.9128794701986755]]
        y_value = 0
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_xy_data_with_y_manipulation_offset_y(self):
        # Often we want to manipulate the label, e.g. for classification
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close", label_type = "bin",
                                        divider = 0.95, offset=5)
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615, 
                    1.0087307269756611, 1.0576923076923077, 0.9128794701986755]]
        y_value = 1
        self.assertXYCorrespondance(x_value, y_value, x_data, y_data)

    def test_formatting_into_balanced_xy(self):
        label_vars = {"divider" : 1, "balance" : True}
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close", label_type = "bin",
                                        divider = 1, balance = True)
        self.assertEqual(y_data.count(0), y_data.count(1))
        


    def test_formatting_into_dataset(self):
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close")
        dataset = TickerDataset(x_data, y_data)
        x_value = [[1.0173076923076922, 1.0173076923076922, 1.1365384615384615, 
                    1.0087307269756611, 1.0576923076923077, 0.9128794701986755]]
        y_value = 0.8322873322860055
        l = len(x_data)
        # Testing if we can acess the dataset correctlye
        self.assertXYCorrespondance(x_value, y_value,
                                    dataset.x_data, dataset.y_data)
        self.assertEqual(dataset.length, l)
        # NOTE: The dataset also has a couple functions that I'm not 100% sure
        # how to test (read: can't be bothered to test) but they are very simple
        # and shouldn't break if this test works

    def test_formatting_into_split_datasets(self):
        x_data, y_data = format_into_xy(self.formatted_data, num_features = 1,
                                        label_var = "Close")
        train_ds, test_ds = train_test_datasets(x_data, y_data, train_size = 0.7)
        train_l = train_ds.length
        test_l = test_ds.length
        self.assertAlmostEqual(train_l / 0.7, test_l / 0.3, -1)
        train_datapoint = train_ds.x_data[0]
        self.assertNotIn(train_datapoint, test_ds.x_data)


if __name__ == '__main__':
    unittest.main()

