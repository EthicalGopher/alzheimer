import sys
import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import argparse

# --- Helper Function ---
def softmax(x):
    """Compute softmax values for a set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ONNXImageDetector:
    def __init__(self, model_path, class_names, quiet=False):
        """
        Initialize ONNX Image Detector

        Args:
            model_path: Path to the .onnx model file
            class_names: List of class names
            quiet: If True, suppresses startup messages.
        """
        self.model_path = model_path
        self.class_names = class_names
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape

        if not quiet:
            print(f"Model loaded: {model_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Input name: {self.input_name}")

    def preprocess_image(self, image_path, target_size=(128, 128)):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)
            img_array = np.array(image, dtype=np.float32) / 255.0

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std

            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            return img_array

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}", file=sys.stderr)
            return None

    def predict_single_image(self, image_path):
        """Predict a single image and return structured data."""
        processed_img = self.preprocess_image(image_path)

        if processed_img is None:
            return None

        try:
            # Run inference and get raw logits
            outputs = self.ort_session.run(None, {self.input_name: processed_img})
            logits = outputs[0][0]

            # Convert logits to probabilities using softmax
            probabilities = softmax(logits)

            # Get predicted class and confidence
            pred_class_idx = np.argmax(probabilities)
            confidence = np.max(probabilities)
            pred_class_name = self.class_names[pred_class_idx]

            # Create a dictionary of all class probabilities
            all_probs_dict = {self.class_names[i]: probabilities[i] for i in range(len(self.class_names))}

            result = {
                'predicted_class': pred_class_name,
                'confidence': float(confidence),
                'all_probabilities': all_probs_dict
            }

            return result

        except Exception as e:
            print(f"Error during inference for {image_path}: {e}", file=sys.stderr)
            return None

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description='ONNX Image Detection System')
    parser.add_argument('--model', '-m', required=True, help='Path to ONNX model file')
    parser.add_argument('--image', '-i', required=True, help='Path to single image file')
    parser.add_argument('--json', action='store_true', help='Output result as a clean JSON object.')

    args = parser.parse_args()

    class_names = [
        'Mild Impairment',
        'Moderate Impairment',
        'No Impairment',
        'Very Mild Impairment',
    ]

    try:
        # Initialize in quiet mode if JSON output is requested
        detector = ONNXImageDetector(args.model, class_names, quiet=args.json)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    result = detector.predict_single_image(args.image)

    if result is None:
        print("Failed to get a prediction.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        # Format for JSON output
        json_result = {
            "predicted_class": result['predicted_class'],
            "confidence": f"{result['confidence'] * 100:.2f}%",
            "all_probabilities": {k: f"{v * 100:.2f}%" for k, v in result['all_probabilities'].items()}
        }
        print(json.dumps(json_result, indent=2))
    else:
        # Human-readable output
        print(f"\nImage: {os.path.basename(args.image)}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll Class Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")

if __name__ == "__main__":
    main()