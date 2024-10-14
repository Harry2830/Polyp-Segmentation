import os
import yaml
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms
from source.dataset import DataSet

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    extn = config["extn"]
    mask_path = config["mask_path"]
    image_path = config["image_path"]
    extn_ = f"*{extn}"

    img_ids = glob(os.path.join(image_path, extn_))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2)

    train_transform = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = DataSet(
        img_ids=train_img_ids,
        img_dir=image_path,
        mask_dir=mask_path,
        img_ext=extn,
        mask_ext=extn,
        transform=train_transform
    )

    val_dataset = DataSet(
        img_ids=val_img_ids,
        img_dir=image_path,
        mask_dir=mask_path,
        img_ext=extn,
        mask_ext=extn,
        transform=val_transform
    )

    # Save preprocessed datasets for GPU environment
    pd.to_pickle(train_dataset, "train_dataset.pkl")
    pd.to_pickle(val_dataset, "val_dataset.pkl")
