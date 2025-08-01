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
    
