from pathlib import Path
import shutil
import random
import yaml
import numpy as np

# =========================
# CONFIG
# =========================

DATASET_ROOT = Path("/mnt/d/EXAMENSARBETE_filer/DATASET")
OUTPUT_ROOT = Path("/mnt/d/EXAMENSARBETE_filer/YOLO_DATASET")

SOURCE_FOLDERS = [
    "ui_cd_d",
    "ui_fd_d",
    "ui_fhrd_d",
    "ui_hrd_d",
]

CAMERA_NAME = "camera0"
IMAGE_W = 1920
IMAGE_H = 1080

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "van": 2,
    "bicycle": 3,
    "motorcycle": 4,
    "walker": 5,
}

CLASS_NAMES = ["car", "truck", "van", "bicycle", "motorcycle", "walker"]


# =========================
# HELPERS
# =========================

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_camera_matrices(data, camera_name):
    cam = data[camera_name]
    K = np.array(cam["intrinsic"], dtype=np.float64)
    E = np.array(cam["extrinsic"], dtype=np.float64)
    return K, E


def world_to_camera(point_world, E):
    p = np.array(
        [point_world[0], point_world[1], point_world[2], 1.0],
        dtype=np.float64
    )
    return (np.linalg.inv(E) @ p)[:3]


def camera_to_image(point_cam, K):
    X, Y, Z = point_cam

    if Y <= 0.2:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (X / Y) + cx
    v = fy * (-Z / Y) + cy

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return float(u), float(v), float(Y)


def get_center_world(obj):
    loc = obj.get("location")
    if loc is None:
        return None

    center = obj.get("center", [0.0, 0.0, 0.0])

    return [
        loc[0] + center[0],
        loc[1] + center[1],
        loc[2] + center[2],
    ]


def get_extent(obj):
    extent = obj.get("extent")
    if extent is None or len(extent) != 3:
        return None
    return extent


def get_yaw(obj):
    angle = obj.get("angle", [0.0, 0.0, 0.0])
    if angle is None or len(angle) != 3:
        return 0.0
    return angle[2]


def get_3d_corners(center_world, extent, yaw_deg):
    cx, cy, cz = center_world
    ex, ey, ez = extent

    yaw = np.deg2rad(yaw_deg)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    corners = []

    for dx in (-ex, ex):
        for dy in (-ey, ey):
            for dz in (-ez, ez):
                x_rot = dx * cos_y - dy * sin_y
                y_rot = dx * sin_y + dy * cos_y
                z_rot = dz

                corners.append([
                    cx + x_rot,
                    cy + y_rot,
                    cz + z_rot,
                ])

    return corners


def make_bbox_from_projected(projected_points):
    if len(projected_points) < 2:
        return None

    xs = [p[0] for p in projected_points]
    ys = [p[1] for p in projected_points]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(IMAGE_W - 1), max(xs))
    y2 = min(float(IMAGE_H - 1), max(ys))

    w = x2 - x1
    h = y2 - y1

    if w <= 3 or h <= 3:
        return None

    aspect = w / (h + 1e-6)

    if aspect > 12 or aspect < 0.05:
        return None

    return float(x1), float(y1), float(x2), float(y2)


def bbox_to_yolo(x1, y1, x2, y2):
    xc = ((x1 + x2) / 2.0) / IMAGE_W
    yc = ((y1 + y2) / 2.0) / IMAGE_H
    w = (x2 - x1) / IMAGE_W
    h = (y2 - y1) / IMAGE_H
    return xc, yc, w, h


# =========================
# LABEL GENERATION
# =========================

def process_object(obj, K, E):
    cls_name = obj.get("class")

    if cls_name not in CLASS_MAP:
        return None

    center_world = get_center_world(obj)
    extent = get_extent(obj)
    yaw = get_yaw(obj)

    if center_world is None or extent is None:
        return None

    center_cam = world_to_camera(center_world, E)

    if center_cam[1] <= 0.5:
        return None

    corners_world = get_3d_corners(center_world, extent, yaw)

    projected = []

    for corner in corners_world:
        point_cam = world_to_camera(corner, E)
        point_img = camera_to_image(point_cam, K)

        if point_img is not None:
            projected.append(point_img)

    bbox = make_bbox_from_projected(projected)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    xc, yc, w, h = bbox_to_yolo(x1, y1, x2, y2)

    if cls_name in ["walker", "bicycle", "motorcycle"]:
        if w < 0.001 or h < 0.001:
            return None
    else:
        if w < 0.002 or h < 0.002:
            return None

    class_id = CLASS_MAP[cls_name]

    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def generate_labels_for_frame(yaml_path: Path):
    data = load_yaml(yaml_path)

    if CAMERA_NAME not in data:
        return []

    K, E = get_camera_matrices(data, CAMERA_NAME)

    labels = []

    for group in ["vehicles", "walkers"]:
        if group not in data:
            continue

        for _, obj in data[group].items():
            label = process_object(obj, K, E)
            if label is not None:
                labels.append(label)

    return labels


# =========================
# DATASET BUILDING
# =========================

def find_frames():
    frames = []

    for source_name in SOURCE_FOLDERS:
        source_root = DATASET_ROOT / source_name

        if not source_root.exists():
            print(f"WARNING: missing folder: {source_root}")
            continue

        sequence_dirs = [
            p for p in source_root.iterdir()
            if p.is_dir() and p.name not in ["-1", "-2"]
        ]

        for seq_dir in sequence_dirs:
            yaml_files = sorted(seq_dir.glob("*.yaml"))

            for yaml_path in yaml_files:
                frame_id = yaml_path.stem

                if "gnss" in frame_id:
                    continue

                image_path = seq_dir / f"{frame_id}_{CAMERA_NAME}.png"

                if image_path.exists():
                    frames.append((source_name, seq_dir.name, frame_id, yaml_path, image_path))

    return frames


def prepare_output_dirs():
    for split in ["train", "val"]:
        (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml():
    data_yaml = OUTPUT_ROOT / "data.yaml"

    content = f"""path: {OUTPUT_ROOT}
train: images/train
val: images/val

names:
"""

    for idx, name in enumerate(CLASS_NAMES):
        content += f"  {idx}: {name}\n"

    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(content)


def build_dataset():
    random.seed(RANDOM_SEED)

    prepare_output_dirs()
    write_data_yaml()

    frames = find_frames()
    random.shuffle(frames)

    split_idx = int(len(frames) * TRAIN_RATIO)
    train_frames = frames[:split_idx]
    val_frames = frames[split_idx:]

    total_images = 0
    total_labels = 0
    skipped_empty = 0

    for split, split_frames in [("train", train_frames), ("val", val_frames)]:
        for source_name, seq_name, frame_id, yaml_path, image_path in split_frames:
            labels = generate_labels_for_frame(yaml_path)

            if len(labels) == 0:
                skipped_empty += 1
                continue

            out_name = f"{source_name}_{seq_name}_{frame_id}_{CAMERA_NAME}"

            out_img = OUTPUT_ROOT / "images" / split / f"{out_name}.png"
            out_label = OUTPUT_ROOT / "labels" / split / f"{out_name}.txt"

            shutil.copyfile(image_path, out_img)

            with open(out_label, "w", encoding="utf-8") as f:
                for line in labels:
                    f.write(line + "\n")

            total_images += 1
            total_labels += len(labels)

    print("Dataset generation finished.")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Source folders: {SOURCE_FOLDERS}")
    print(f"Total source frames found: {len(frames)}")
    print(f"Images written: {total_images}")
    print(f"Labels written: {total_labels}")
    print(f"Skipped empty frames: {skipped_empty}")
    print(f"YOLO data file: {OUTPUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    build_dataset()