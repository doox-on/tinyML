#!/usr/bin/env python3
"""
Raspberry Pi Camera Module 3 person detection

"""

import os
import time
import threading
import numpy as np
import cv2

# ---------------- Defaults ---------------------------------------------------
MODEL_PATH  = "model/coco_ssd_mobilenet_v1/detect.tflite"
LABELS_PATH = "model/coco_ssd_mobilenet_v1/labelmap.txt"
THRESHOLD   = 0.5
PREVIEW_W, PREVIEW_H = 640, 360   # lower preview size -> higher FPS
TARGET_FPS  = 30                  # best-effort
INFER_EVERY = 2                   # run inference every N frames
NUM_THREADS = max(1, os.cpu_count() // 2)
SMOOTHING_ALPHA = 0.6             # 0..1; higher = follow current box more

# ---------------- TFLite/LiteRT import --------------------------------------
Interpreter = None
load_delegate = None
try:
    from litert import Interpreter as _Interpreter  # type: ignore
    Interpreter = _Interpreter
    try:
        from tflite_runtime.interpreter import load_delegate as _load_delegate  # type: ignore
        load_delegate = _load_delegate
    except Exception:
        load_delegate = None
except Exception:
    from tflite_runtime.interpreter import Interpreter as _Interpreter, load_delegate as _load_delegate  # type: ignore
    Interpreter = _Interpreter
    load_delegate = _load_delegate

# ---------------- Camera import ---------------------------------------------
try:
    from picamera2 import Picamera2
except Exception as e:
    raise SystemExit("picamera2 not available. Install: sudo apt install -y python3-picamera2\n" + str(e))

# ---------------- Utils ------------------------------------------------------

def load_labels(path):
    with open(path, "r") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    if labels and labels[0] == '???':
        labels = labels[1:]
    return labels


def make_interpreter(model_path):
    # Try XNNPACK; set threads for CPU parallelism
    kwargs = {"model_path": model_path, "num_threads": NUM_THREADS}
    if load_delegate is not None:
        try:
            return Interpreter(**kwargs, experimental_delegates=[load_delegate('libtensorflowlite_delegate_xnnpack.so')])
        except Exception:
            pass
    return Interpreter(**kwargs)


def parse_outputs(interpreter):
    out = interpreter.get_output_details()
    boxes = interpreter.get_tensor(out[0]['index'])[0]
    classes = interpreter.get_tensor(out[1]['index'])[0]
    scores = interpreter.get_tensor(out[2]['index'])[0]
    count = int(interpreter.get_tensor(out[3]['index'])[0])
    return boxes, classes, scores, count


def draw_person(frame_bgr, detections, labels, thresh=0.5, last_box=None):
    h, w = frame_bgr.shape[:2]
    boxes, classes, scores, count = detections
    best = None
    best_score = 0.0
    for i in range(count):
        s = float(scores[i])
        if s < thresh:
            continue
        cid = int(classes[i])
        name = labels[cid] if 0 <= cid < len(labels) else str(cid)
        if name != 'person':
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        box = np.array([int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)], dtype=np.int32)
        if s > best_score:
            best, best_score = box, s

    if best is None:
        return None, 0.0

    if last_box is not None:
        best = (SMOOTHING_ALPHA*best + (1.0-SMOOTHING_ALPHA)*last_box).astype(np.int32)

    x1,y1,x2,y2 = best.tolist()
    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(frame_bgr, f"person:{best_score:.2f}", (x1, max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return best, best_score

# ---------------- Main -------------------------------------------------------

def main():
    # OpenCV speed knobs
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, os.cpu_count()//2))
    except Exception:
        pass

    labels = load_labels(LABELS_PATH)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    interp = make_interpreter(MODEL_PATH)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    in_h, in_w = int(in_det['shape'][1]), int(in_det['shape'][2])

    cam = Picamera2()
    config = cam.create_video_configuration(main={"size": (PREVIEW_W, PREVIEW_H), "format": "BGR888"}, buffer_count=6)
    cam.configure(config)
    try:
        cam.set_controls({"FrameRate": int(TARGET_FPS)})
    except Exception:
        pass
    cam.start()
    time.sleep(0.2)

    cv2.namedWindow('Pi Vision', cv2.WINDOW_AUTOSIZE)

    frames = 0
    last_infer_ms = 0.0
    last_det = None
    last_box = None

    t0 = time.time()
    shown = 0
    smoothed_fps = 0.0

    try:
        while True:
            frame_rgb = cam.capture_array()    # RGB888
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # single convert for display

            # Inference throttle
            if frames % INFER_EVERY == 0:
                resized = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                tensor = np.expand_dims(resized, 0).astype(np.uint8)
                interp.set_tensor(in_det['index'], tensor)
                t_infer = time.time()
                interp.invoke()
                last_infer_ms = (time.time() - t_infer) * 1000.0
                last_det = parse_outputs(interp)

            # Draw boxes using last detections (smooths UI between infers)
            if last_det is not None:
                new_box, score = draw_person(frame_bgr, last_det, labels, THRESHOLD, last_box)
                last_box = new_box if new_box is not None else None
            else:
                last_box = None

            # FPS (update ~every 0.5s)
            shown += 1
            now = time.time()
            if now - t0 >= 0.5:
                smoothed_fps = shown / (now - t0)
                t0 = now
                shown = 0

            # HUD
            cv2.putText(frame_bgr, f"{last_infer_ms:.1f} ms | every {INFER_EVERY}f | {smoothed_fps:.1f} FPS",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow('Pi Vision', frame_bgr)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

            frames += 1
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("bye")


if __name__ == '__main__':
    main()
