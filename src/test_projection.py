from pathlib import Path
from datetime import datetime
import yaml
import cv2
import numpy as np

IMAGE_W = 1920
IMAGE_H = 1080

BASE_PATH = Path("/mnt/d/EXAMENSARBETE_filer/DATASET/ui_cd_d/1263")
FRAME_ID = "000060"
CAMERA_NAME = "camera0"

CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "van": 2,
    "bicycle": 3,
    "motorcycle": 4,
    "walker": 5,
}

CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 0, 255),
    "van": (255, 0, 0),
    "bicycle": (0, 255, 255),
    "motorcycle": (255, 255, 0),
    "walker": (255, 0, 255),
}


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_camera_matrices(data, camera_name):
    cam = data[camera_name]
    K = np.array(cam["intrinsic"], dtype=np.float64)
    E = np.array(cam["extrinsic"], dtype=np.float64)
    return K, E


def world_to_camera(point_world, E):
    p = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
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


def process_object(obj, K, E):
    cls_name = obj.get("class")
    if cls_name not in CLASS_MAP:
        return None, "class_not_used"

    center_world = get_center_world(obj)
    extent = get_extent(obj)
    yaw = get_yaw(obj)

    if center_world is None or extent is None:
        return None, "missing_center_or_extent"

    center_cam = world_to_camera(center_world, E)

    if center_cam[1] <= 0.5:
        return None, "behind_camera"

    corners_world = get_3d_corners(center_world, extent, yaw)

    projected = []
    for corner in corners_world:
        point_cam = world_to_camera(corner, E)
        point_img = camera_to_image(point_cam, K)

        if point_img is not None:
            projected.append(point_img)

    bbox = make_bbox_from_projected(projected)

    if bbox is None:
        return None, "bbox_failed"

    return {
        "class_name": cls_name,
        "class_id": CLASS_MAP[cls_name],
        "bbox": bbox,
        "center_cam": center_cam.tolist(),
        "projected_corners": len(projected),
    }, "ok"


def draw_and_label(image, data, K, E):
    drawn = image.copy()
    debug_rows = []
    yolo_rows = []

    total = 0
    kept = 0

    for group in ["vehicles", "walkers"]:
        if group not in data:
            continue

        for obj_id, obj in data[group].items():
            total += 1

            result, status = process_object(obj, K, E)

            debug = {
                "object_id": obj_id,
                "class_name": obj.get("class"),
                "status": status,
            }

            if result is None:
                debug_rows.append(debug)
                continue

            x1, y1, x2, y2 = result["bbox"]
            cls_name = result["class_name"]
            cls_id = result["class_id"]

            debug.update(result)
            debug_rows.append(debug)

            kept += 1

            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            cv2.rectangle(
                drawn,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            cv2.putText(
                drawn,
                cls_name,
                (int(x1), max(20, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            xc, yc, w, h = bbox_to_yolo(x1, y1, x2, y2)

            # Hans start
            # Keep smaller labels for pedestrians, bicycles and motorcycles.
            # These classes are often small in traffic scenes.
            if cls_name in ["walker", "bicycle", "motorcycle"]:
                if w >= 0.001 and h >= 0.001:
                    yolo_rows.append((cls_id, xc, yc, w, h))
            else:
                if w >= 0.002 and h >= 0.002:
                    yolo_rows.append((cls_id, xc, yc, w, h))
            # Hans stopp

    return drawn, debug_rows, total, kept, yolo_rows


def main():
    yaml_path = BASE_PATH / f"{FRAME_ID}.yaml"
    img_path = BASE_PATH / f"{FRAME_ID}_{CAMERA_NAME}.png"

    data = load_yaml(yaml_path)
    image = cv2.imread(str(img_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    K, E = get_camera_matrices(data, CAMERA_NAME)

    drawn, debug_rows, total, kept, yolo_rows = draw_and_label(image, data, K, E)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_img = BASE_PATH / f"debug_bboxes_{FRAME_ID}_{CAMERA_NAME}_{timestamp}.png"
    out_debug = BASE_PATH / f"debug_bboxes_{FRAME_ID}_{CAMERA_NAME}_{timestamp}.txt"
    out_labels = BASE_PATH / f"debug_bboxes_{FRAME_ID}_{CAMERA_NAME}_{timestamp}_labels.txt"

    ok = cv2.imwrite(str(out_img), drawn)
    if not ok:
        raise RuntimeError(f"Could not save image: {out_img}")

    with open(out_debug, "w", encoding="utf-8") as f:
        f.write(f"Total candidate objects: {total}\n")
        f.write(f"Projected bbox count: {kept}\n")
        f.write(f"YOLO label count: {len(yolo_rows)}\n\n")
        for row in debug_rows:
            f.write(f"{row}\n")

    with open(out_labels, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in yolo_rows:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Saved image: {out_img}")
    print(f"Saved debug: {out_debug}")
    print(f"Saved labels: {out_labels}")
    print(f"Total candidate objects: {total}")
    print(f"Projected bbox count: {kept}")
    print(f"YOLO label count: {len(yolo_rows)}")


if __name__ == "__main__":
    main()