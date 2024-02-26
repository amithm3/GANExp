import os
from pathlib import Path
from typing import Union, Generator
import xml.etree.ElementTree as ET

from utils.datasets import ImageDataset


class CelebADataset(ImageDataset):
    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "jessicali9530/celeba-dataset")

    def get_data(self) -> Generator[dict[str, ...], None, None]:
        files = os.listdir(folder := self.DIR / "img_align_celeba" / "img_align_celeba")
        for file in files: yield {
            "file": folder / file,
        }


class ITSDataset(ImageDataset):
    SETS = "hazy", "clear", "trans"

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "balraj98/indoor-training-set-its-residestandard")

    def get_data(self) -> Generator[dict[str, ...], None, None]:
        files = os.listdir(folder := self.DIR / self.SET)
        for file in files: yield {
            "file": folder / file,
        }


class DenseHazeCVPR2019Dataset(ImageDataset):
    SETS = "hazy", "GT"

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "rajat95gupta/hazing-images-dataset-cvpr-2019")

    def get_data(self) -> str:
        files = os.listdir(folder := self.DIR / self.SET)
        for file in files: yield {
            "file": folder / file,
        }


class VINSDataset(ImageDataset):
    SETS = "Android", "iphone", "Rico", "uplabs", "Wireframes"

    @classmethod
    def download(cls, DIR: Union[str, "Path"]):
        cls.kaggle_download(DIR, "raja25/vins-dataset")

    def get_data(self) -> str:
        for file in os.listdir((folder := self.DIR / "All Dataset" / self.SET) / "Annotations"):
            try:
                root = ET.parse(folder / "Annotations" / file).getroot()
                file = f"JPEGImages/{file.replace('.xml', '.jpg')}"
                labels = [element.text
                          for element in root.findall('.//name')]
                bboxes = [[int(float(box.find(bound).text)) for bound in ("xmin", "ymin", "xmax", "ymax")]
                          for box in root.findall('.//bndbox')]
                yield {
                    "file": folder / file,
                    "labels": labels,
                    "bboxes": bboxes,
                }
            except IndexError:
                break


__all__ = [
    "CelebADataset",
    "ITSDataset",
    "DenseHazeCVPR2019Dataset",
    "VINSDataset",
]
