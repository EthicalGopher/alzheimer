import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Class labels matching your model
classes = ['MI', 'MOD', 'NI', 'VMI']

def load_and_preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    try:
        # Load image
        img = Image.open(image_path)
        print(f"Original image size: {img.size}")

        # Convert to RGB if needed (handles grayscale or RGBA)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize to 224x224 (matching your training)
        img = img.resize((224, 224))

        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension [1, 224, 224, 3]
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_single_image(interpreter, image_path):
    """Predict class for a single image using TensorFlow Lite"""
    # Preprocess image
    img_array = load_and_preprocess_image(image_path)
    if img_array is None:
        return None, None, None

    # Get input and output tensors info
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions)
    predicted_class = classes[predicted_class_idx]
    confidence = predictions[predicted_class_idx]

    return predicted_class, confidence, predictions

def predict_multiple_images(interpreter, image_paths):
    """Predict classes for multiple images"""
    results = []

    for img_path in image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        predicted_class, confidence, predictions = predict_single_image(interpreter, img_path)

        if predicted_class is not None:
            results.append((img_path, predicted_class, confidence, predictions))
        else:
            print(f"Failed to process: {img_path}")

    return results

def main():
    # Check for TensorFlow Lite model
    model_path = 'alzheimers_model.tflite'

    if not os.path.exists(model_path):
        print(f"TensorFlow Lite model '{model_path}' not found!")
        print("Make sure you have 'alzheimers_model.tflite' in the current directory.")
        return

    # Load TensorFlow Lite model
    print(f"Loading TensorFlow Lite model from {model_path}...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("TensorFlow Lite model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if len(sys.argv) < 2:
        print("Usage: python classifier.py <image_path> [image_path2] [image_path3] ...")
        print("Example: python classifier.py test/test.jpg")
        print("Example: python classifier.py test/*.jpg")
        return

    image_paths = sys.argv[1:]

    # Single image prediction
    if len(image_paths) == 1:
        image_path = image_paths[0]
        print(f"\nAnalyzing image: {image_path}")

        if not os.path.exists(image_path):
            print(f"Image file '{image_path}' not found!")
            return

        predicted_class, confidence, all_predictions = predict_single_image(interpreter, image_path)

        if predicted_class is None:
            print("Failed to analyze image!")
            return

        print(f"\n{'='*50}")
        print(f"PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")

        print(f"\nAll class probabilities:")
        for i, class_name in enumerate(classes):
            print(f"  {class_name}: {all_predictions[i]*100:.2f}%")

    # Multiple image prediction
    else:
        print(f"\nAnalyzing {len(image_paths)} images...")

        # Filter existing files
        existing_files = [path for path in image_paths if os.path.exists(path)]
        if not existing_files:
            print("No valid image files found!")
            return

        results = predict_multiple_images(interpreter, existing_files)

        print(f"\n{'='*80}")
        print(f"BATCH PREDICTION RESULTS")
        print(f"{'='*80}")
        print(f"{'Image':<25} {'Predicted':<8} {'Confidence':<12} {'All Probabilities'}")
        print(f"{'-'*80}")

        for img_path, predicted_class, confidence, all_predictions in results:
            img_name = os.path.basename(img_path)
            probs_str = " | ".join([f"{classes[i]}:{all_predictions[i]*100:.1f}%" for i in range(4)])
            print(f"{img_name:<25} {predicted_class:<8} {confidence*100:>8.2f}%    {probs_str}")

        # Summary
        if results:
            class_counts = {class_name: 0 for class_name in classes}
            for _, predicted_class, _, _ in results:
                class_counts[predicted_class] += 1

            print(f"\nSUMMARY:")
            print(f"Total images processed: {len(results)}")
            for class_name, count in class_counts.items():
                percentage = (count / len(results)) * 100
                print(f"{class_name}: {count} images ({percentage:.1f}%)")

def predict_directory(directory_path):
    """Helper function to predict all images in a directory"""
    model_path = 'alzheimers_model.tflite'

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get all image files
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    for file in os.listdir(directory_path):
        if file.lower().endswith(supported_formats):
            image_files.append(os.path.join(directory_path, file))

    if not image_files:
        print(f"No supported image files found in {directory_path}")
        return

    print(f"Found {len(image_files)} images in {directory_path}")
    results = predict_multiple_images(interpreter, image_files)

    # Display results
    for img_path, predicted_class, confidence, all_predictions in results:
        print(f"{os.path.basename(img_path)}: {predicted_class} ({confidence*100:.1f}%)")

if __name__ == "__main__":
    main()

# Example usage:
# python classifier.py test/test.jpg
# python classifier.py test/image1.jpg test/image2.jpg
#
# To predict all images in a directory, you can modify main() or use:
# predict_directory("test/")