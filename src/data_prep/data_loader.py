import os
import torch
import torchvision.transforms as transforms
import anyconfig
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image

# Load config
CONFIG_PATH = "config/config.yaml"
config = anyconfig.load(CONFIG_PATH)

class COCODamageDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to COCO annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Assigning label based on annotation (assuming one label per image)
        label = torch.tensor(annotations[0]['category_id']) if annotations else torch.tensor(0)

        return image, label

def create_dataloader(dataset_type="train"):
    """
    Creates a dataloader based on dataset type.

    Args:
        dataset_type (str): One of ["train", "val", "test"].

    Returns:
        DataLoader: PyTorch DataLoader object.
    """
    dataset_path = config["data"][dataset_type]["img_dir"]
    annotation_path = config["data"][dataset_type]["annotation_file"]
    batch_size = config["training"]["batch_size"]

    dataset = COCODamageDataset(dataset_path, annotation_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(dataset_type == "train"))
    
    return dataloader
