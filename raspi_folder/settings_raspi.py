model_path = "/home/Ssg_K/fyp/tomato_model.h5"

set_cause = 'v'


# Disease labels
disease_labels = [
    "Bacterial Spot",
    "Late Blight",
    "Septoria Leaf Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]




# Constants
target_size = (224, 224)  # Model expects 224x224 images
green_threshold = 0.05
n_rows = 5
n_cols = 5

# car controls
speed = 100


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
