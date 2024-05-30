from pathlib import Path
import json
import imagesize
from sklearn.model_selection import train_test_split
import argparse

def create_image_annotation(file_path: Path, image_id: int):
    w, h = imagesize.get(str(file_path))
    return {
        "file_name": file_path.name,
        "height": h,
        "width": w,
        "id": image_id,
    }

def create_annotation_from_yolo_format(line, image_id, annotation_id, w, h):
    parts = line.split()
    class_id = int(parts[0])  # �대옒�� ID�� 泥� 踰덉㎏ �붿냼�낅땲��.
    # parts[1]�� <name>�쇰줈, �ш린�쒕뒗 �ъ슜�섏� �딆뒿�덈떎.
    x_center, y_center, width, height = map(float, parts[2:])  # <x_center>遺��� �쒖옉�섎뒗 �섎㉧吏� 遺�遺꾩쓣 float�쇰줈 蹂��섑빀�덈떎.
    min_x = int(w * x_center - w * width / 2)
    min_y = int(h * y_center - h * height / 2)
    width = int(w * width)
    height = int(h * height)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": [min_x, min_y, width, height],
        "area": width * height,
        "iscrowd": 0,
        "category_id": class_id + 1, # COCO �곗씠�곗뀑�� 1遺��� �쒖옉�⑸땲��.
        "segmentation": [],
    }


def process_annotations(file_paths, txt_path,  output_path):
    annotations = []
    images = []
    image_id = annotation_id = 0
    total_files = len(file_paths)
    for idx, file_path in enumerate(file_paths):
        print(f"Processing {idx+1}/{total_files}...", end='\r')
        images.append(create_image_annotation(file_path, image_id))
        label_file_name = f"{file_path.stem}.txt"
        label_path = txt_path / label_file_name
        if label_path.exists():
            with open(label_path, encoding='cp949') as f:
                for line in f:
                    annotations.append(create_annotation_from_yolo_format(line, image_id, annotation_id, images[-1]["width"], images[-1]["height"]))
                    annotation_id += 1
        image_id += 1
    print("\nCompleted processing all files.")
    with open(output_path, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": [{"supercategory": "none", "id": 1, "name": "person"}]}, f, indent=4)
    

def split_data(file_paths, test_size=0.1, val_size=0.1):
    train_test_labels = [str(path).split('/')[1].rsplit('_', 1)[0] for path in file_paths]
    train_val_files, test_files = train_test_split(file_paths, test_size=test_size, stratify = train_test_labels, random_state=42)
    train_val_labels = [str(path).split('/')[1].rsplit('_', 1)[0] for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=val_size / (1 - test_size), stratify = train_val_labels, random_state=42)
    return train_files, val_files, test_files

def get_args():
    parser = argparse.ArgumentParser("Yolo format annotations to COCO dataset format")
    parser.add_argument("--musicbank_230923_231201", default='./musicbank_230923_231201',type=str, help="Root dir for jpgs.")
    parser.add_argument("--refine1", default='./refine1',type=str, help="Root dir for txts.")
    parser.add_argument("--annotations1", default="./annotations1", type=str, help="Directory where the output JSON files will be saved")
    parser.add_argument("--dataset_coco", default="dataset_coco", type=str, help="Base name for output json files")
    return parser.parse_args(args=[])

def main():
    opt = get_args()
    image_path = Path(opt.musicbank_230923_231201)
    txt_path = Path(opt.refine1)
    output_dir = Path(opt.annotations1)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = sorted(image_path.glob("*.jpg")) + sorted(image_path.glob("*.jpeg")) + sorted(image_path.glob("*.png"))
    train_files, val_files, test_files = split_data(file_paths)
    
    process_annotations(train_files, txt_path, output_dir / f"{opt.dataset_coco}_train.json")
    process_annotations(val_files, txt_path, output_dir / f"{opt.dataset_coco}_val.json")
    process_annotations(test_files, txt_path, output_dir / f"{opt.dataset_coco}_test.json")

if __name__ == "__main__":
    main()