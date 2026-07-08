from pathlib import Path
import cv2


def yolo_to_xyxy(box, img_w, img_h):
  xc, yc, w, h = box
  x1 = int((xc - w / 2) * img_w)
  y1 = int((yc - h / 2) * img_h)
  x2 = int((xc + w / 2) * img_w)
  y2 = int((yc + h / 2) * img_h)
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(img_w, x2), min(img_h, y2)
  return x1, y1, x2, y2

def extract_objects(images_dir, labels_dir, out_dir, classes):
  images_dir = Path(images_dir)
  labels_dir = Path(labels_dir)
  out_dir = Path(out_dir)

  for cls in classes:
    (out_dir / cls).mkdir(parents=True, exist_ok=True)

  image_files = [p for p in images_dir.iterdir() if p.suffix.lower() == '.jpg']

  counters = {cls: 0 for cls in classes}

  for img_path in image_files:
    label_path = labels_dir / (img_path.stem + ".txt")
    img = cv2.imread(str(img_path))

    h, w = img.shape[:2]

    gt_boxes_px, entries = [], []
    if label_path.exists():
      with open(label_path, "r") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          parts = line.split()
          cls_id = int(parts[0])
          box = tuple(float(x) for x in parts[1:5])
          x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
          if x2 <= x1 or y2 <= y1:
            continue
          gt_boxes_px.append((x1, y1, x2, y2))
          entries.append((cls_id, (x1, y1, x2, y2)))

    for cls_id, (x1, y1, x2, y2) in entries:
      crop = img[y1:y2, x1:x2]
      if crop.size == 0:
        continue
      cls_name = classes[cls_id]
      out_path = out_dir / cls_name / f"{img_path.stem}_{counters[cls_name]}.jpg"
      cv2.imwrite(str(out_path), crop)
      counters[cls_name] += 1


  print("Patches Completed")
  for cls, count in counters.items():
      print(f"  {cls}: {count} patches")

CLASSES = ["cardboard", "cloth", "egg-shell", "organic","plastic","plastic-bag","tea-waste"]

extract_objects("images","labels","crops", CLASSES)