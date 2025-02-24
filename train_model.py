import requests
import json
import random

# Function to generate random shots
def generate_random_shot():
    shot_distance = round(random.uniform(3.0, 30.0), 1)  # Distance between 3 and 30 meters
    angle = round(random.uniform(10.0, 90.0), 1)  # Angle between 10 and 90 degrees
    body_part = random.choice(["foot", "head", "other"])  # Random body part
    # Probability of goal decreases with distance and angle
    goal_probability = 0.8 if shot_distance < 10 and angle > 60 else \
                       0.5 if shot_distance < 20 and angle > 30 else \
                       0.2
    goal = 1 if random.random() < goal_probability else 0
    return {"shot_distance": shot_distance, "angle": angle, "body_part": body_part, "goal": goal}

# Generate the expanded dataset
expanded_shots = [
    {"shot_distance": 5.2, "angle": 67.8, "body_part": "foot", "goal": 1},
    {"shot_distance": 18.7, "angle": 23.4, "body_part": "foot", "goal": 0},
    {"shot_distance": 8.1, "angle": 51.3, "body_part": "head", "goal": 1},
    {"shot_distance": 5.2, "angle": 67.8, "body_part": "foot", "goal": 1},
        {"shot_distance": 18.7, "angle": 23.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 8.1, "angle": 51.3, "body_part": "head", "goal": 1},
        {"shot_distance": 22.3, "angle": 19.5, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.4, "angle": 72.1, "body_part": "foot", "goal": 1},
        {"shot_distance": 17.9, "angle": 28.6, "body_part": "head", "goal": 0},
        {"shot_distance": 4.3, "angle": 81.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 24.6, "angle": 14.2, "body_part": "foot", "goal": 0},
        {"shot_distance": 11.2, "angle": 42.9, "body_part": "foot", "goal": 1},
        {"shot_distance": 19.8, "angle": 31.5, "body_part": "head", "goal": 0},
        {"shot_distance": 7.5, "angle": 63.4, "body_part": "foot", "goal": 1},
        {"shot_distance": 26.3, "angle": 12.7, "body_part": "foot", "goal": 0},
        {"shot_distance": 9.4, "angle": 52.8, "body_part": "head", "goal": 0},
        {"shot_distance": 21.7, "angle": 22.3, "body_part": "foot", "goal": 0},
        {"shot_distance": 3.8, "angle": 84.9, "body_part": "foot", "goal": 1},
        {"shot_distance": 15.6, "angle": 37.2, "body_part": "foot", "goal": 0},
        {"shot_distance": 8.9, "angle": 58.5, "body_part": "head", "goal": 1},
        {"shot_distance": 23.1, "angle": 17.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 5.7, "angle": 76.3, "body_part": "foot", "goal": 1},
        {"shot_distance": 19.4, "angle": 25.9, "body_part": "head", "goal": 0},
        {"shot_distance": 10.2, "angle": 48.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 27.8, "angle": 11.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.8, "angle": 69.5, "body_part": "foot", "goal": 1},
        {"shot_distance": 20.5, "angle": 20.6, "body_part": "head", "goal": 0},
        {"shot_distance": 4.9, "angle": 79.2, "body_part": "foot", "goal": 1},
        {"shot_distance": 16.3, "angle": 34.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 9.7, "angle": 55.1, "body_part": "head", "goal": 1},
        {"shot_distance": 24.2, "angle": 15.9, "body_part": "foot", "goal": 0},
        {"shot_distance": 7.1, "angle": 66.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 22.9, "angle": 18.3, "body_part": "other", "goal": 0},
        {"shot_distance": 3.2, "angle": 87.4, "body_part": "foot", "goal": 1},
        {"shot_distance": 14.8, "angle": 39.6, "body_part": "foot", "goal": 0},
        {"shot_distance": 8.4, "angle": 61.2, "body_part": "head", "goal": 1},
        {"shot_distance": 25.7, "angle": 13.5, "body_part": "foot", "goal": 0},
        {"shot_distance": 5.3, "angle": 73.9, "body_part": "foot", "goal": 1},
        {"shot_distance": 17.2, "angle": 32.7, "body_part": "head", "goal": 0},
        {"shot_distance": 11.6, "angle": 46.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 28.4, "angle": 10.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.1, "angle": 70.6, "body_part": "foot", "goal": 1},
        {"shot_distance": 20.9, "angle": 24.1, "body_part": "head", "goal": 0},
        {"shot_distance": 9.8, "angle": 53.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 26.5, "angle": 12.3, "body_part": "foot", "goal": 0},
        {"shot_distance": 4.5, "angle": 82.9, "body_part": "foot", "goal": 1},
        {"shot_distance": 16.7, "angle": 36.2, "body_part": "other", "goal": 0},
        {"shot_distance": 10.4, "angle": 49.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 23.6, "angle": 16.7, "body_part": "foot", "goal": 0},
        {"shot_distance": 7.3, "angle": 64.5, "body_part": "foot", "goal": 1},
        {"shot_distance": 18.2, "angle": 29.4, "body_part": "head", "goal": 0},
        {"shot_distance": 5.9, "angle": 74.8, "body_part": "foot", "goal": 1},
        {"shot_distance": 25.1, "angle": 14.7, "body_part": "foot", "goal": 0},
        {"shot_distance": 4.1, "angle": 83.6, "body_part": "foot", "goal": 1},
        {"shot_distance": 15.3, "angle": 38.7, "body_part": "head", "goal": 0},
        {"shot_distance": 8.7, "angle": 59.3, "body_part": "foot", "goal": 1},
        {"shot_distance": 27.2, "angle": 11.9, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.6, "angle": 71.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 19.1, "angle": 27.3, "body_part": "head", "goal": 0},
        {"shot_distance": 12.3, "angle": 44.5, "body_part": "foot", "goal": 1},
        {"shot_distance": 22.7, "angle": 19.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 3.6, "angle": 85.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 14.5, "angle": 40.9, "body_part": "head", "goal": 0},
        {"shot_distance": 9.2, "angle": 57.6, "body_part": "foot", "goal": 1},
        {"shot_distance": 24.8, "angle": 15.3, "body_part": "foot", "goal": 0},
        {"shot_distance": 5.5, "angle": 77.1, "body_part": "foot", "goal": 1},
        {"shot_distance": 17.6, "angle": 31.8, "body_part": "head", "goal": 0},
        {"shot_distance": 11.4, "angle": 47.2, "body_part": "foot", "goal": 1},
        {"shot_distance": 28.7, "angle": 10.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.3, "angle": 68.9, "body_part": "foot", "goal": 1},
        {"shot_distance": 21.2, "angle": 23.1, "body_part": "head", "goal": 0},
        {"shot_distance": 4.7, "angle": 80.3, "body_part": "foot", "goal": 1},
        {"shot_distance": 16.1, "angle": 35.4, "body_part": "foot", "goal": 0},
        {"shot_distance": 8.5, "angle": 60.8, "body_part": "head", "goal": 1},
        {"shot_distance": 25.4, "angle": 13.9, "body_part": "foot", "goal": 0},
        {"shot_distance": 7.8, "angle": 65.7, "body_part": "foot", "goal": 1},
        {"shot_distance": 23.3, "angle": 17.2, "body_part": "other", "goal": 0},
        {"shot_distance": 13.5, "angle": 41.5, "body_part": "foot", "goal": 0},
        {"shot_distance": 18.4, "angle": 30.1, "body_part": "head", "goal": 0},
        {"shot_distance": 5.0, "angle": 78.6, "body_part": "foot", "goal": 1},
        {"shot_distance": 26.8, "angle": 12.0, "body_part": "foot", "goal": 0},
        {"shot_distance": 8.0, "angle": 62.3, "body_part": "foot", "goal": 1},
        {"shot_distance": 20.0, "angle": 25.2, "body_part": "head", "goal": 0},
        {"shot_distance": 12.0, "angle": 43.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 27.5, "angle": 11.1, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.9, "angle": 72.5, "body_part": "foot", "goal": 1},
        {"shot_distance": 19.5, "angle": 26.8, "body_part": "head", "goal": 0},
        {"shot_distance": 3.4, "angle": 86.4, "body_part": "foot", "goal": 1},
        {"shot_distance": 15.0, "angle": 39.0, "body_part": "foot", "goal": 0},
        {"shot_distance": 9.0, "angle": 54.9, "body_part": "head", "goal": 0},
        {"shot_distance": 24.0, "angle": 16.3, "body_part": "foot", "goal": 0},
        {"shot_distance": 7.6, "angle": 67.4, "body_part": "foot", "goal": 1},
        {"shot_distance": 22.0, "angle": 20.2, "body_part": "head", "goal": 0},
        {"shot_distance": 4.4, "angle": 81.1, "body_part": "foot", "goal": 1},
        {"shot_distance": 17.0, "angle": 33.3, "body_part": "foot", "goal": 0},
        {"shot_distance": 10.3, "angle": 51.7, "body_part": "head", "goal": 1},
        {"shot_distance": 26.0, "angle": 13.1, "body_part": "foot", "goal": 0},
        {"shot_distance": 6.0, "angle": 75.3, "body_part": "foot", "goal": 1},
        {"shot_distance": 21.0, "angle": 22.7, "body_part": "head", "goal": 0},
        {"shot_distance": 11.0, "angle": 45.8, "body_part": "foot", "goal": 0},
        {"shot_distance": 25.0, "angle": 15.0, "body_part": "foot", "goal": 0},
        {"shot_distance": 12.9, "angle": 43.1, "body_part": "foot", "goal": 0},
        {"shot_distance": 23.5, "angle": 19.0, "body_part": "foot", "goal": 0},
        {"shot_distance": 17.5, "angle": 33.0, "body_part": "foot", "goal": 1}
]

# Expand the dataset to 800-900 shots
while len(expanded_shots) < 800:
    expanded_shots.append(generate_random_shot())

# Create the training data dictionary
training_data = {"shots": expanded_shots}

# Send training data to the API
try:
    response = requests.post(
        'http://127.0.0.1:5000/train',
        json=training_data
    )
    print("Response status:", response.status_code)
    print("Response content:", response.json())
except Exception as e:
    print("Error occurred:", str(e))