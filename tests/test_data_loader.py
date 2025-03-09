import pytest
import torch
import anyconfig
from src.data_prep.data_loader import create_dataloader, COCODamageDataset

# Load config
CONFIG_PATH = "config/config.yaml"
config = anyconfig.load(CONFIG_PATH)

@pytest.fixture
def test_dataloader():
    return create_dataloader(dataset_type="train")

def test_dataloader_loads_data(test_dataloader):
    data_iter = iter(test_dataloader)
    images, labels = next(data_iter)

    assert isinstance(images, torch.Tensor), "Images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "Labels should be a Tensor"

    assert images.shape[0] == config["training"]["batch_size"], "Batch size should match config"
    assert images.shape[1] == 3, "Images should have 3 channels (RGB)"
    assert images.shape[2] == 224 and images.shape[3] == 224, "Images should be 224x224"

def test_dataset_length():
    dataset = COCODamageDataset(
        root_dir=config["data"]["train"]["img_dir"], 
        annotation_file=config["data"]["train"]["annotation_file"]
    )
    assert len(dataset) > 0, "Dataset should not be empty"
