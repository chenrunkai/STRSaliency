NTE_MODULE_PATH = "."
from .dataset import Dataset
from .real import (ComputersDataset, CricketXDataset, EarthquakesDataset,
                   EcgDataset, FordADataset, FordBDataset, GunPointDataset,
                   PTBHeartRateDataset, WaferDataset)
from .synth.blipv3.blipv3_dataset import BlipV3Dataset


def get_dataset(dataset_name: str):
    # Dataset Mapper
    if dataset_name == 'blip':
        dataset = BlipV3Dataset()
    elif dataset_name in ['wafer', "1"]:
        dataset = WaferDataset()
    elif dataset_name in ['cricket_x', "2"]:
        dataset = CricketXDataset()
    elif dataset_name in ['gun_point', "3"]:
        dataset = GunPointDataset()
    elif dataset_name in ['earthquakes', "4"]:
        dataset = EarthquakesDataset()
    elif dataset_name in ['computers', "5"]:
        dataset = ComputersDataset()
    elif dataset_name in ['ford_a', "6"]:
        dataset = FordADataset()
    elif dataset_name in ['ford_b', "7"]:
        dataset = FordBDataset()
    elif dataset_name in ['ptb', "8"]:
        dataset = PTBHeartRateDataset()
    elif dataset_name in ['ecg', "9"]:
        dataset = EcgDataset()
    else:
        raise Exception(f"Unknown Dataset: {dataset_name}")
    return dataset

def backgroud_data_configuration(BACKGROUND_DATA, BACKGROUND_DATA_PERC, dataset: Dataset):
    # Background Data Configuration
    if BACKGROUND_DATA == 'train':
        print("Using TRAIN data as background data")
        bg_data = dataset.train_data
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    elif BACKGROUND_DATA == 'test':
        print("Using TEST data as background data")
        bg_data = dataset.test_data
        bg_len = int(len(bg_data) * BACKGROUND_DATA_PERC / 100)
    else:
        print("Using Instance as background data (No BG Data)")
        bg_data = dataset.test_data
        bg_len = 0
    return bg_data, bg_len