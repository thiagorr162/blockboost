import pathlib
import zipfile

# The download is not automated

ZIP_DIR = pathlib.Path("research/data/raw/celeba")
IMGS_DIR = pathlib.Path("research/data/raw/celeba/imgs")

with zipfile.ZipFile(ZIP_DIR / "img_align_celeba.zip", "r") as zip_ref:
    zip_ref.extractall(IMGS_DIR)
