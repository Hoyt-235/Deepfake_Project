import os
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, List

import cv2
import dlib
import numpy as np
from skimage import transform as trans

# ===== CONFIG =====
PREDICTOR_PATH = "/home/jpcha/TFM/src/preprocessing/extraction/dlib_tools/shape_predictor_81_face_landmarks.dat"
NUM_FRAMES_PER_VIDEO = -1       # keep all candidate frames if negative
STRIDE = 5                     # 0 means process every frame
PADDING = 64                   # pad bbox by this many pixels on each side
OUTPUT_SIZE = (256, 256)       # final crop size
MAX_WORKERS = max(1, os.cpu_count() - 1)

# 5‑pt reference for alignment (if landmarks enabled)
REF_POINTS_112 = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)

# Globals to be initialized per worker
detector = None
predictor = None

def get_five_points(shape68: np.ndarray) -> np.ndarray:
    idx = [36, 45, 30, 48, 54]
    return shape68[idx]

def align_face(img: np.ndarray, pts_src: np.ndarray, out_size: Tuple[int,int], scale: float = 1.3) -> np.ndarray:
    h, w = out_size
    dst = REF_POINTS_112.copy()
    dst[:,0] = dst[:,0] * w/112 + (scale-1)*w/2
    dst[:,1] = dst[:,1] * h/112 + (scale-1)*h/2
    dst[:,0] *= w/(w + (scale-1)*w)
    dst[:,1] *= h/(h + (scale-1)*h)
    tform = trans.SimilarityTransform()
    tform.estimate(pts_src.astype(np.float32), dst)
    M = tform.params[0:2,:]
    return cv2.warpAffine(img, M, (w,h))

def worker_init(predictor_path: str, enable_landmarks: bool, enable_masks: bool):
    """
    Initialize globals in each worker process.
    """
    global detector, predictor
    cv2.setNumThreads(1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path) if enable_landmarks else None
    # Mask initialization placeholder (if enable_masks)

def extract_and_save_faces(
    video_path: str,
    out_dir: str,
    enable_landmarks: bool,
    enable_masks: bool
) -> Tuple[int, float]:
    """
    Process one video, detect & save up to NUM_FRAMES_PER_VIDEO face crops.
    Returns (faces_saved, size_in_mb).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    idxs = list(range(0, total_frames, STRIDE)) if STRIDE > 0 else list(range(total_frames))
    chosen = idxs if NUM_FRAMES_PER_VIDEO < 0 else random.sample(idxs, min(NUM_FRAMES_PER_VIDEO, len(idxs)))

    saved_count, saved_bytes = 0, 0.0

    for fi in sorted(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2))
        dets = detector(small, 1)
        if not dets:
            continue

        best = max(dets, key=lambda r: r.width()*r.height())
        x1, y1 = best.left()*2, best.top()*2
        x2, y2 = best.right()*2, best.bottom()*2

        x1, y1 = max(x1 - PADDING, 0), max(y1 - PADDING, 0)
        x2 = min(x2 + PADDING, frame.shape[1]); y2 = min(y2 + PADDING, frame.shape[0])
        crop = frame[y1:y2, x1:x2]

        if enable_landmarks:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shp = predictor(rgb, dlib.rectangle(x1, y1, x2, y2))
            pts68 = np.array([[p.x, p.y] for p in shp.parts()])
            pts5 = get_five_points(pts68)
            face_crop = align_face(frame, pts5, OUTPUT_SIZE)
        else:
            face_crop = cv2.resize(crop, OUTPUT_SIZE)

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_filename = f"{base}_{fi:06d}.jpg"
        out_path = os.path.join(out_dir, out_filename)
        cv2.imwrite(out_path, face_crop)

        saved_bytes += os.path.getsize(out_path)
        saved_count += 1

    cap.release()
    return saved_count, saved_bytes / (1024*1024)

def extract_faces_from_json(
    json_path: str,
    out_root: str,
    enable_landmarks: bool = False,
    enable_masks: bool = False
):
    """
    Extract faces for train/val/test splits defined in a JSON file.

    JSON format:
      {
        "train": {"real": [...], "fake": [...]},
        "val":   {"real": [...], "fake": [...]},
        "test":  {"real": [...], "fake": [...]}  
      }
    """
    with open(json_path, 'r') as f:
        splits = json.load(f)

    for split_name, class_dict in splits.items():
        # Flatten (class, video) pairs
        all_videos = [(cls, vid) for cls in ['real', 'fake'] for vid in class_dict.get(cls, [])]
        out_split_dir = os.path.join(out_root, split_name)
        print(f"\n🗂 Extracting split '{split_name}' ({len(all_videos)} videos) to: {out_split_dir}")

        counts = {'faces': 0, 'size': 0.0}
        pbar = tqdm(total=len(all_videos), desc=f"{split_name}")

        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=worker_init,
            initargs=(PREDICTOR_PATH, enable_landmarks, enable_masks)
        ) as exe:
            futures = {}
            for cls, vid in all_videos:
                out_dir = os.path.join(out_split_dir, cls)
                os.makedirs(out_dir, exist_ok=True)
                fut = exe.submit(extract_and_save_faces, vid, out_dir, enable_landmarks, enable_masks)
                futures[fut] = (cls, vid)

            for fut in as_completed(futures):
                cls, vid = futures[fut]
                try:
                    fc, mb = fut.result()
                    counts['faces'] += fc
                    counts['size'] += mb
                except Exception as e:
                    tqdm.write(f"❌ {cls}/{vid}: {e}")
                pbar.set_postfix(faces=counts['faces'], size=f"{counts['size']:.1f} MB")
                pbar.update(1)
        pbar.close()

        # Rename within each class folder
        for cls in ['real', 'fake']:
            class_dir = os.path.join(out_split_dir, cls)
            jpegs = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.jpg'))
            for idx, old in enumerate(jpegs, start=1):
                os.rename(
                    os.path.join(class_dir, old),
                    os.path.join(class_dir, f"face_{idx:06d}.jpg")
                )

        print(f"✅ Split '{split_name}' done: {counts['faces']} faces, ~{counts['size']:.1f} MB")

# Example CLI usage
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from video splits JSON")
    parser.add_argument('--splits', type=str, required=True, help='Path to splits JSON file')
    parser.add_argument('--out', type=str, required=True, help='Root output directory')
    parser.add_argument('--landmarks', action='store_true', default=False, help='Enable landmark-based alignment')
    parser.add_argument('--masks', action='store_true', default=False, help='Enable mask processing (optional)')

    args = parser.parse_args()
    extract_faces_from_json(
        json_path=args.splits,
        out_root=args.out,
        enable_landmarks=args.landmarks,
        enable_masks=args.masks
    )
