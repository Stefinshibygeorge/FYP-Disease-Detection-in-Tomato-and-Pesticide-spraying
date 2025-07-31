import numpy as np
import tensorflow as tf  # tf version == 2.13.0
import cv2
import RPi.GPIO as GPIO
import time
from settings import (
    model_path, target_size, green_threshold, n_rows, n_cols,
    speed, set_cause, disease_labels,
    IR_PIN, IN1, IN2, IN3, IN4, EN1, EN2, S1, S2
)


# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # Avoid warnings on reruns

# Pin Definitions
IR_PIN = 14

# Motor Control Pins
IN1 = 5   # Motor A forward
IN2 = 6   # Motor A backward
IN3 = 19  # Motor B forward
IN4 = 26  # Motor B backward

# Enable pins for speed control (PWM)
EN1 = 18  # Enable pin for Motor A (Speed)
EN2 = 13  # Enable pin for Motor B (Speed)

# Sprayer Pins
S1 = 17
S2 = 27

# GPIO Configuration
GPIO.setup(IR_PIN, GPIO.IN)

GPIO.setup(S1, GPIO.OUT)
GPIO.setup(S2, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(EN1, GPIO.OUT)
GPIO.setup(EN2, GPIO.OUT)

# PWM setup (for speed control)
pwm1 = GPIO.PWM(EN1, 1000)  # Frequency: 1kHz
pwm2 = GPIO.PWM(EN2, 1000)
pwm1.start(0)  # Initially off
pwm2.start(0)

# Load Model
model = tf.keras.models.load_model(model_path)

def spray_pump():
    """Activates the sprayer for 1 second."""
    print("\nSpraying pesticide...\n")
    GPIO.output(S1, GPIO.HIGH)
    GPIO.output(S2, GPIO.LOW)
    time.sleep(1)
    GPIO.output(S1, GPIO.LOW)
    GPIO.output(S2, GPIO.LOW)
    print("Spraying completed.")

def forward(speed=speed):
    """Moves the car forward with adjustable speed (0-100%)."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm1.ChangeDutyCycle(speed)
    pwm2.ChangeDutyCycle(speed)

def stop():
    """Stops the motors."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)

def capture_image():
    """Captures an image from the camera and resizes it to model input size."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    time.sleep(2)  # Camera warm-up time
    ret, frame = cap.read()
    cap.release()  # Release the camera

    if ret:
        cv2.destroyAllWindows()  # Ensure previous windows are closed
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(10)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.image.resize(frame, target_size) / 255.0  # Resize & Normalize
        return frame
    else:
        print("Error: Failed to capture video frame.")
        return None
        
        
def classify_image(image, green_threshold=green_threshold):
    """Splits the image into regions, checks green content in each, and classifies only valid regions."""
    
    # Convert TensorFlow tensor to NumPy array if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Dimensions of each sub-frame
    fragment_height = target_size[0] // n_rows
    fragment_width = target_size[1] // n_cols
    count = np.zeros(10, dtype=int)
    
    for i in range(n_rows):
        for j in range(n_cols):
            start_row, end_row = i * fragment_height, (i + 1) * fragment_height
            start_col, end_col = j * fragment_width, (j + 1) * fragment_width

            fragment = image[start_row:end_row, start_col:end_col, :]

            # Extract RGB channels
            red, green, blue = fragment[:, :, 2], fragment[:, :, 1], fragment[:, :, 0]
            print(green)

            # Create a mask for "green" pixels (where green is significantly stronger than red and blue)
            green_mask = (green > red ) & (green > blue)
            green_pixels = np.count_nonzero(green_mask)
            
            # Compute the green pixel ratio
            green_ratio = green_pixels / (fragment.shape[0] * fragment.shape[1])

            # If green ratio is below the threshold, ignore this fragment
            if green_ratio < green_threshold:
                print(f"Sub-frame ({i}, {j}) skipped due to low green content (Ratio: {green_ratio:.2f})")
                continue  # Skip this fragment
            
            # Resize & Normalize
            fragment = tf.image.resize(fragment, (224, 224)) / 255.0
            fragment = tf.expand_dims(fragment, axis=0)

            # Predict the disease in this sub-frame
            preds = np.argmax(model.predict(fragment), axis=1)[0]
            count[preds] += 1  # Count occurrences of each disease class
            
            time.sleep(2)
            
    # If no valid sub-frames were classified, return None
    if np.sum(count) == 0:
        print("No valid sub-frames detected for classification.")
        return None

    return np.argmax(count)  # Return the dominant disease index


def find_cause(dominant_disease_index):
    if dominant_disease_index == 0:
        print('bacterial')
        return 'ba'
    elif (dominant_disease_index == 1) or (dominant_disease_index == 2):
        print('fungal')
        return 'f'
    elif (dominant_disease_index == 3) or (dominant_disease_index == 4):
        print('viral')
        return 'v'
    else:
        return'h'
    

# Main Loop
try:
    while True:
        if GPIO.input(IR_PIN) == 0:
            stop()
            print("\nCar Stopped\n")
            time.sleep(2)  # Allow time for IR sensor state change

            captured_image = capture_image()
            if captured_image is not None:
                dominant_disease_index = classify_image(captured_image)
                if dominant_disease_index is not None:
                    print(f"Dominant Disease: {disease_labels[dominant_disease_index]}")
                    cause = find_cause(dominant_disease_index)
                    # Spray if not healthy
                    if set_cause == cause:
                        spray_pump()
                        time.sleep(5)

        print("Car Moving...")
        forward(speed=70)
        time.sleep(1)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")
    GPIO.cleanup()  # Reset GPIO pins
    pwm1.stop()
    pwm2.stop()
    cv2.destroyAllWindows()
