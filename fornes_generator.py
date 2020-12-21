import os
import os.path
import json
import shutil

DATASET_PATH = "./raw_datasets/Fornes/Music_Symbols/"
TARGET_PATH = "./dataset/regular"

classes_map = {
    'ACCIDENTAL_Natural': 'acc_natural',
    'ACCIDENTAL_Sharp': 'acc_sharp',
    'ACCIDENTAL_Flat': 'acc_flat',
    'ACCIDENTAL_DoubSharp': 'acc_double_sharp',
    'CLEF_Alto': 'clef_c',
    'CLEF_Bass': 'clef_f',
    'CLEF_Trebble': 'clef_g',
}


def main():
    metadata_map = {}

    img_id = 0
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            category = os.path.basename(dirpath)
            new_filename = f"f_{img_id}_{category}.bmp"

            path = os.path.join(dirpath, filename)
            new_path = os.path.join(TARGET_PATH, new_filename)
            shutil.copy(path, new_path)

            metadata_map[new_filename] = classes_map[category]
            img_id += 1
    meta_json = json.dumps(metadata_map)
    with open(os.path.join(TARGET_PATH, "f_meta.json"), "w") as file:
        file.write(meta_json)


if __name__ == "__main__":
    main()
