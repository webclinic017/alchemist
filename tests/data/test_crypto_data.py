import math
import unittest
import pandas as pd
from torch.utils.data import Dataset
from alchemist.data.crypto_data import *


class TestDownloadingData(unittest.TestCase):

    def setUp(self):
        self.data = CryptoData()

    def test_download_single_pair_data(self):
        self.data.download_data(["BTC-USD"], from_date="2021-06-01",
                                to_date="2021-06-07")
        raw_data = self.data.raw_data
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertEqual(len(raw_data.index), 7)
        self.assertAlmostEqual(raw_data["High"]["BTC-USD"]["2021-06-05"],
                               37917.7, 0)
    
    def test_downloading_multiple_pair_data(self):
        self.data.download_data(["BTC-USD", "ETH-USD"], 
                                from_date="2021-06-01", 
                                to_date="2021-06-07")
        specific_day_data = self.data.raw_data.loc["2021-06-05", "High"]
        self.assertAlmostEqual(specific_day_data["BTC-USD"], 37917.7, 0)
        self.assertAlmostEqual(specific_day_data["ETH-USD"], 2817.5, 0)

    def test_downloading_overlapping_data(self):
        # Sometimes we have some data, and want to download more
        self.data.download_data(["BTC-USD"], from_date="2021-06-01",
                                to_date="2021-06-07")
        self.data.download_data(["BTC-USD", "ETH-USD"], 
                                from_date="2021-06-05", 
                                to_date="2021-06-10")
        raw_data = self.data.raw_data
        self.assertEqual(len(raw_data), 10)
        # There should be gaps in the dataframe in the form of nan values
        self.assertTrue(math.isnan(raw_data["High"]["ETH-USD"]["2021-06-01"]))
        # But old and new data should be accessible
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-01"]))
        self.assertFalse(math.isnan(raw_data["High"]["ETH-USD"]["2021-06-09"]))
        # Verify that the data is actually correct
        specific_day_data = raw_data.loc["2021-06-07", "High"]
        self.assertAlmostEqual(specific_day_data["BTC-USD"], 36790.6, 0)
        self.assertAlmostEqual(specific_day_data["ETH-USD"], 2845.2, 0)

    def test_download_non_overlapping_data(self):
        # Sometimes we download data from unconnected time periods
        # This is unadvisable, but the empty space should be filled with nan
        self.data.download_data(["BTC-USD"], from_date="2021-06-01",
                                to_date="2021-06-07")
        self.data.download_data(["BTC-USD"], from_date="2021-06-14", 
                                to_date="2021-06-20")
        # There should be data for the dates specified, nan in between
        raw_data = self.data.raw_data
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-05"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-15"]))
        self.assertTrue(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-10"]))
        
    def test_filling_in_gaps(self):
        self.data.download_data(["BTC-USD"], from_date="2021-06-01",
                                to_date="2021-06-07")
        self.data.download_data(["BTC-USD", "ETH-USD"], 
                                from_date="2021-06-10", 
                                to_date="2021-06-14")
        # The nans in the above data should be filled in with this
        self.data.download_data(["ETH-USD"], from_date="2021-06-01",
                                to_date="2021-06-07")
        # The empty space between the data should be filled with this
        self.data.download_data(["BTC-USD", "ETH-USD"], 
                                from_date="2021-06-07", 
                                to_date="2021-06-10")
        # Verify this worked 
        raw_data = self.data.raw_data
        self.assertFalse(math.isnan(raw_data["High"]["ETH-USD"]["2021-06-03"]))
        self.assertFalse(math.isnan(raw_data["High"]["ETH-USD"]["2021-06-08"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-08"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-03"]))
        # Check that the old data still exists, and that the order is right
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-03"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2021-06-12"]))
        self.assertEqual(str(raw_data.index[4])[:10], "2021-06-05")
        self.assertEqual(str(raw_data.index[8])[:10], "2021-06-09")
        self.assertEqual(str(raw_data.index[12])[:10], "2021-06-13")

    def test_downloading_non_existant_data(self):
        # As ETH went live 2015-08-07, this should have a chunk of nan data
        self.data.download_data(["BTC-USD", "ETH-USD"], "2015-08-01",
                                "2015-08-14")
        raw_data = self.data.raw_data
        self.assertTrue(math.isnan(raw_data["High"]["ETH-USD"]["2015-08-03"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2015-08-03"]))
        self.assertFalse(math.isnan(raw_data["High"]["ETH-USD"]["2015-08-09"]))
        self.assertFalse(math.isnan(raw_data["High"]["BTC-USD"]["2015-08-09"]))

    def test_downloading_data_on_init(self):
        # Data should be downloaded if dates etc are provided on initialisation
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"], from_date="2021-06-01",
                          to_date="2021-06-07")
        specific_day_data = data.raw_data.loc["2021-06-05", "High"]
        self.assertAlmostEqual(specific_day_data["BTC-USD"], 37917.7, 0)
        self.assertAlmostEqual(specific_day_data["ETH-USD"], 2817.5, 0)


class TestFormattingDataForTraining(unittest.TestCase):
    
    def setUp(self):
        # This should download some data with plenty of irregular gaps,
        # which formatting needs to be able to deal with
        self.data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                               from_date="2021-06-01", to_date="2021-06-06")
        self.data.download_data(["BTC-USD"], "2021-06-11", "2021-06-15")
        self.data.download_data(["ETH-USD"], "2021-06-06", "2021-06-08")

    def test_formatting_into_percentages(self):
        # Should just generate a pandas df similar to raw_data
        self.data.format_data_into_percentages()
        specific_day_data = self.data.percentage_data.loc["2021-06-05", "High"]
        self.assertAlmostEqual(specific_day_data["BTC-USD"], 1.028, 3)
        self.assertAlmostEqual(specific_day_data["ETH-USD"], 1.048, 3)
        # A bunch of the data should be nan gaps
        data = self.data.percentage_data
        self.assertTrue(math.isnan(data["High"]["BTC-USD"]["2021-06-08"]))
        self.assertFalse(math.isnan(data["High"]["ETH-USD"]["2021-06-08"]))
        self.assertTrue(math.isnan(data["High"]["BTC-USD"]["2021-06-11"]))
        self.assertTrue(math.isnan(data["High"]["BTC-USD"]["2021-06-01"]))
        # Volume must be formatted seperately, in relation to other volume
        self.assertAlmostEqual(
                data["Volume"]["BTC-USD"]["2021-06-04"], 1.18, 3)

    def test_basic_dataset(self):
        self.data.format_data_into_percentages()
        self.data.generate_datasets()
        self.assertIsInstance(self.data.train_ds, Dataset)
        self.assertNotIn(math.nan, self.data.train_ds.y_data)
        x_value = [[0.952272357743142, 0.952272357743142, 1.0019962111851022,
                    0.9301928746591981, 1.0001539088344926, 0.979960886160529]]
        y_value = 1.0997213189467518
        index = self.data.train_ds.x_data.index(x_value)
        self.assertEqual(self.data.train_ds.y_data[index], y_value)

    def test_train_test_split_datasets(self):
        self.data.format_data_into_percentages()
        self.data.generate_datasets(train_fraction=0.7)
        self.assertIsInstance(self.data.train_ds, Dataset)
        self.assertIsInstance(self.data.test_ds, Dataset)
        self.assertAlmostEqual(self.data.train_ds.length / 7,
                               self.data.test_ds.length / 3, 1)

    def test_volatility_adjusted_datasets(self):
        self.data.format_data_into_percentages()
        self.data.generate_datasets(adjust_volatility = True)
        # This is the same data checked above, with adjusted volatility
        x_value = [[0.9503752081226438, 0.9503752081226438, 1.0, 
                    0.9283397125414483, 0.9981613679472594, 
                    0.9780085745049764]]
        y_value = 1.0997213189467518
        index = self.data.train_ds.x_data.index(x_value)
        self.assertEqual(self.data.train_ds.y_data[index], y_value)

    def test_backtest_dataset(self):
        self.data.format_data_into_percentages()
        self.data.generate_datasets(adjust_volatility = True,
                                    backtest_dataset=True)
        # If we had data for more than one pair for a given day,
        # We should have a corresponding x with more than one set of data
        # As such backtest data's data will have an extra dimension
        self.assertEqual(len(self.data.backtest_ds.x_data[0]), 2)
        self.assertEqual(len(self.data.backtest_ds.x_data[-1]), 1)
        self.assertEqual(len(self.data.backtest_ds.y_data[0]), 2)
        self.assertEqual(len(self.data.backtest_ds.y_data[-1]), 1)

    def test_formatting_on_init(self):
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                          from_date="2021-06-05", to_date="2021-06-15",
                          adjust_volatility=False)
        # Specifying adjust_volatility on init should be enough to make the ds
        x_value = [[0.952272357743142, 0.952272357743142, 1.0019962111851022,
                    0.9301928746591981, 1.0001539088344926, 0.979960886160529]]
        y_value = 1.0997213189467518
        index = data.train_ds.x_data.index(x_value)
        self.assertEqual(data.train_ds.y_data[index], y_value)


class TestFormattingVariables(unittest.TestCase):

    def test_divider(self):
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                          from_date="2021-06-10", to_date="2021-06-15",
                          adjust_volatility=False, divider=1)
        x_value = [[0.952272357743142, 0.952272357743142, 1.0019962111851022,
                    0.9301928746591981, 1.0001539088344926, 0.979960886160529]]
        y_value = 1.0
        index = data.train_ds.x_data.index(x_value)
        self.assertEqual(data.train_ds.y_data[index], y_value)

    def test_balance(self):
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                          from_date="2021-06-01", to_date="2021-06-10",
                          adjust_volatility=False, divider=1, balance=True)
        self.assertEqual(data.train_ds.y_data.count(0),
                         data.train_ds.y_data.count(1))

    def test_num_features(self):
        data = CryptoData(pairs=["BTC-USD", "ETH-USD"],
                          from_date="2021-06-01", to_date="2021-06-10",
                          adjust_volatility=False, n_features=3)
        self.assertEqual(len(data.train_ds.x_data[0]), 3)

    # def test_formatting_basis(self):
        # pass

    # def test_volatility_type(self):
        # pass

    # def test_label_var(self):
        # pass


class TestSavingLoadingData(unittest.TestCase):
    # If given a filename, the data manager should load as much
    # data as possible from the file and download / format the rest

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
