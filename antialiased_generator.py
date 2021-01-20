import glob
import os.path
import shutil
from PIL import Image, ImageFilter

DATASET_PATH = "./dataset/regular"
TARGET_PATH = "./dataset/antialiased"


def main():
    for meta_file in glob.glob(DATASET_PATH + "/*.json"):
        path = meta_file
        new_path = os.path.join(TARGET_PATH, os.path.basename(meta_file))
        print(path, new_path)
        shutil.copy(path, new_path)

    for image_file in glob.glob(DATASET_PATH + "/*.bmp"):
        path = os.path.join(DATASET_PATH, image_file)
        print(path)
        process_image(image_file)


def process_image(path):
    try:
        img = Image.open(path, 'r')
        if img is None:
            print(f"image is none: {path}")
            return

        size = img.size
        img = img.convert(mode="RGB")
        img = img.resize((size[0] * 2, size[1] * 2), Image.BICUBIC)
        img.save(os.path.join(TARGET_PATH, os.path.basename(path)))
    except:
        pass

if __name__ == "__main__":
    main()
