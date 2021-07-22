import math
import unittest
import pandas as pd
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


class TestFormattingData(unittest.TestCase):

    def test_(self):
        pass


class TestPreparingDataForTraining(unittest.TestCase):

    def test_(self):
        pass


class TestSavingLoadingData(unittest.TestCase):
    # If given a filename, the data mmanager should load as much
    # data as possible from the file and download / format the rest

    def test_(self):
        pass


if __name__ == '__main__':
    unittest.main()
