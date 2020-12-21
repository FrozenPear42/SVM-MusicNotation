import math
import os
import os.path
import json
from operator import itemgetter

from PIL import Image, ImageDraw

DATASET_PATH = "./raw_datasets/HOMUS/HOMUS"
TARGET_PATH = "./dataset/regular"

classes_map = {
    'Common-Time': 'time_common',
    'Cut-Time': 'time_cut',
    '2-2-Time': 'time_2_2',
    '2-4-Time': 'time_2_4',
    '3-4-Time': 'time_3_4',
    '4-4-Time': 'time_4_4',
    '3-8-Time': 'time_3_8',
    '6-8-Time': 'time_6_8',
    '9-8-Time': 'time_9_8',
    '12-8-Time': 'time_12_8',
    'Quarter-Note': 'note_quarter',
    'Quarter-Rest': 'rest_quarter',
    'Sixteenth-Note': 'note_sixteenth',
    'Sixteenth-Rest': 'rest_sixteenth',
    'Sixty-Four-Note': 'note_sixty_four',
    'Sixty-Four-Rest': 'rest_sixty_four',
    'Thirty-Two-Note': 'note_thirty_two',
    'Thirty-Two-Rest': 'rest_thirty_two',
    'Whole-Half-Rest': 'rest_whole_half',
    'Whole-Note': 'note_whole',
    'Eighth-Note': 'note_eighth',
    'Eighth-Rest': 'rest_eighth',
    'Half-Note': 'note_half',
    'Natural': 'acc_natural',
    'Sharp': 'acc_sharp',
    'Flat': 'acc_flat',
    'Double-Sharp': 'acc_double_sharp',
    'C-Clef': 'clef_c',
    'F-Clef': 'clef_f',
    'G-Clef': 'clef_g',
    'Barline': 'misc_barline',
    'Dot': 'misc_dot',
}


def main():
    metadata_map = {}

    img_id = 0
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            category, img, bounds = process_file(os.path.join(dirpath, filename))
            filename = f"h_{img_id}_{category}.bmp"
            img.save(os.path.join(TARGET_PATH, filename))
            metadata_map[filename] = classes_map[category]
            img_id += 1
    meta_json = json.dumps(metadata_map)
    with open(os.path.join(TARGET_PATH, "h_meta.json"), "w") as file:
        file.write(meta_json)


def process_file(path):
    with open(path, "r") as file:
        lines = file.readlines()
        category = lines[0].strip()
        strokes_data = lines[1:]

        strokes = []
        for stroke_data in strokes_data:
            points = [pair_to_point(p) for p in stroke_data.rstrip().split(";") if len(p) > 0]
            strokes.append(points)

        bounds = (10000, 10000, 0, 0)

        for stroke in strokes:
            min_x = min(min(stroke, key=itemgetter(0))[0], bounds[0])
            min_y = min(min(stroke, key=itemgetter(1))[1], bounds[1])
            max_x = max(max(stroke, key=itemgetter(0))[0], bounds[2])
            max_y = max(max(stroke, key=itemgetter(1))[1], bounds[3])
            bounds = (min_x, min_y, max_x, max_y)

        size = (max_x - min_x + 1, max_y - min_y + 1)
        img = Image.new('RGB', size, color=(0, 0, 0))
        g = ImageDraw.Draw(img)
        for stroke in strokes:
            points = [(p[0] - min_x, p[1] - min_y) for p in stroke]
            g.line(points, fill=(255, 255, 255), width=2)

    return category, img, bounds


def pair_to_point(text):
    data = text.split(",")
    return int(data[0]), int(data[1])


if __name__ == "__main__":
    main()
