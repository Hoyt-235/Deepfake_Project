import os
import cv2
import dlib
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract frames + binary face‐masks from a directory of videos")
    p.add_argument("--video_dir",    required=True,
                   help="Path to folder containing your videos")
    p.add_argument("--output_dir",   required=True,
                   help="Where to write `frames/` and `masks/` subfolders")
    p.add_argument("--predictor",    required=True,
                   help="Path to dlib's shape_predictor_68_face_landmarks.dat")
    p.add_argument("--num_videos",   type=int, default=None,
                   help="Process only the first N videos (sorted by name)")
    return p.parse_args()

def face_mask_from_landmarks(shape, img_shape):
    pts  = np.array([[pt.x, pt.y] for pt in shape.parts()])
    hull = cv2.convexHull(pts)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def main():
    args = parse_args()
    frames_dir = os.path.join(args.output_dir, "frames")
    masks_dir  = os.path.join(args.output_dir, "masks")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    vids = sorted([
        os.path.join(args.video_dir, fn)
        for fn in os.listdir(args.video_dir)
        if fn.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])
    if args.num_videos:
        vids = vids[:args.num_videos]
    if not vids:
        print("No videos found in", args.video_dir)
        return

    total_frames = sum(
        int(cv2.VideoCapture(vp).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for vp in vids
    )

    pbar = tqdm(total=total_frames, desc="Samples extracted", unit="sample")
    for vpath in vids:
        basename = os.path.splitext(os.path.basename(vpath))[0]
        cap = cv2.VideoCapture(vpath)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # save frame
            frame_name = f"{basename}_frame{idx:06d}.jpg"
            cv2.imwrite(os.path.join(frames_dir, frame_name), frame)

            # detect faces & build mask
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 1)
            mask = np.zeros_like(gray)
            for d in dets:
                shape = predictor(gray, d)
                m = face_mask_from_landmarks(shape, frame.shape)
                cv2.bitwise_or(mask, m, mask)

            # save mask
            mask_name = f"{basename}_frame{idx:06d}.png"
            cv2.imwrite(os.path.join(masks_dir, mask_name), mask)

            idx += 1
            pbar.update(1)
            pbar.set_postfix_str(f"samples={pbar.n}")

        cap.release()
    pbar.close()

if __name__ == "__main__":
    main()
