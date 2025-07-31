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
