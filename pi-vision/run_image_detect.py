#!/usr/bin/env python3
"""
Performs object detection on a single image using a TFLite model.

This script loads a quantized COCO SSD MobileNet v1 model, processes a given
image, and detects objects. It then draws bounding boxes specifically for
the 'person' class on the output image.
"""
import sys
import time
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# --- Configuration Constants ---
DEFAULT_MODEL_PATH = "model/coco_ssd_mobilenet_v1/detect.tflite"
DEFAULT_IMAGE_PATH = "frame.jpg"
DEFAULT_LABEL_PATH = "model/coco_ssd_mobilenet_v1/labelmap.txt"
DETECTION_THRESHOLD = 0.5
TARGET_CLASS = "person"


def load_labels(path: str) -> list[str]:
    """Loads labels from a text file, one per line."""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def create_interpreter(model_path: str) -> Interpreter:
    """
    Creates a TFLite interpreter, attempting to use the XNNPACK delegate for
    CPU performance optimization.
    """
    try:
        # Attempt to load the interpreter with the XNNPACK delegate for acceleration
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[load_delegate('libtensorflowlite_delegate_xnnpack.so')]
        )
    except (ValueError, RuntimeError):
        # Fallback to the standard interpreter if the delegate is not available
        print("XNNPACK delegate not found or failed to load. Using standard CPU interpreter.")
        interpreter = Interpreter(model_path=model_path)
    
    return interpreter


def main():
    """
    Main function to load model, process image, and perform detection.
    """
    # Parse command-line arguments or use defaults
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    image_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGE_PATH
    label_path = DEFAULT_LABEL_PATH

    # Load labels and create the TFLite interpreter
    labels = load_labels(label_path)
    interpreter = create_interpreter(model_path)
    interpreter.allocate_tensors()

    # Get model input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, input_height, input_width, _ = input_details[0]['shape']

    # Load and pre-process the image
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Error: Unable to read image at {image_path}")
    
    original_height, original_width = image.shape[:2]
    
    # Convert color, resize, and add batch dimension for model input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image_rgb, (input_width, input_height))
    input_tensor = np.expand_dims(resized_image, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    
    start_time = time.time()
    interpreter.invoke()
    inference_time_ms = (time.time() - start_time) * 1000

    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    # Print top 5 detections for debugging/analysis
    top_pairs = sorted([(float(scores[i]), int(classes[i])) for i in range(num_detections)], reverse=True)[:5]
    print("TOP-5 Detections:", [(labels[c] if 0 <= c < len(labels) else f"ID:{c}", f"{s:.2f}") for s, c in top_pairs])

    # Draw bounding boxes for the target class
    person_detected = False
    for i in range(num_detections):
        score = scores[i]
        class_id = int(classes[i])
        
        # Check if the detection meets the threshold and is the target class
        if score >= DETECTION_THRESHOLD and 0 <= class_id < len(labels) and labels[class_id] == TARGET_CLASS:
            person_detected = True
            
            # Get bounding box coordinates and scale them to the original image size
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * original_width)
            y1 = int(ymin * original_height)
            x2 = int(xmax * original_width)
            y2 = int(ymax * original_height)
            
            # Draw the rectangle and label on the image
            label = f"{labels[class_id]}: {score:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the output image and print a summary
    output_path = "detection_output.jpg"
    cv2.imwrite(output_path, image)
    
    print(f"\n[SUCCESS] Inference time: {inference_time_ms:.1f} ms")
    print(f"Person detected: {person_detected}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
