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
