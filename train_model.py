import requests
import json
import random

def generate_random_shot():
    shot_distance = round(random.uniform(3.0, 35.0), 1)
    angle = round(random.uniform(0.0, 90.0), 1)
    shot_type = random.choice(["open_play", "header", "free_kick", "penalty"])
    defensive_pressure = random.choice(["low", "medium", "high"])
    assist_type = random.choice(["none", "cross", "through_ball"])
    
    # Deterministic goal probability with minimal randomness
    base_goal_probability = 0.0
    if shot_distance < 5 and angle > 70:
        base_goal_probability = 0.98
    elif shot_distance < 10 and angle > 50:
        base_goal_probability = 0.85
    elif shot_distance < 20 and angle > 30:
        base_goal_probability = 0.40
    else:
        base_goal_probability = 0.05
    
    if shot_type == "penalty":
        base_goal_probability = 0.98
    elif shot_type == "free_kick" and shot_distance < 20:
        base_goal_probability = max(base_goal_probability, 0.60)
    elif shot_type == "header" and shot_distance > 15:
        base_goal_probability *= 0.6
    
    if defensive_pressure == "high":
        base_goal_probability *= 0.6
    elif defensive_pressure == "medium":
        base_goal_probability *= 0.85
    if assist_type == "through_ball" and shot_distance < 15:
        base_goal_probability = min(base_goal_probability + 0.25, 0.98)
    elif assist_type == "cross" and shot_type == "header":
        base_goal_probability = min(base_goal_probability + 0.20, 0.98)
    
    goal_probability = max(0.05, min(0.98, base_goal_probability))
    goal = 1 if random.random() < goal_probability else 0
    return {
        "shot_distance": shot_distance,
        "angle": angle,
        "shot_type": shot_type,
        "defensive_pressure": defensive_pressure,
        "assist_type": assist_type,
        "goal": goal
    }

expanded_shots = [
    {"shot_distance": 5.2, "angle": 67.8, "shot_type": "open_play", "defensive_pressure": "low", "assist_type": "through_ball", "goal": 1},
    {"shot_distance": 18.7, "angle": 23.4, "shot_type": "open_play", "defensive_pressure": "high", "assist_type": "none", "goal": 0},
    {"shot_distance": 8.1, "angle": 51.3, "shot_type": "header", "defensive_pressure": "low", "assist_type": "cross", "goal": 1},
    {"shot_distance": 22.3, "angle": 19.5, "shot_type": "free_kick", "defensive_pressure": "medium", "assist_type": "none", "goal": 0},
    {"shot_distance": 6.4, "angle": 72.1, "shot_type": "open_play", "defensive_pressure": "low", "assist_type": "through_ball", "goal": 1},
    {"shot_distance": 17.9, "angle": 28.6, "shot_type": "header", "defensive_pressure": "high", "assist_type": "cross", "goal": 0},
    {"shot_distance": 4.3, "angle": 81.7, "shot_type": "penalty", "defensive_pressure": "low", "assist_type": "none", "goal": 1},
    {"shot_distance": 24.6, "angle": 14.2, "shot_type": "open_play", "defensive_pressure": "high", "assist_type": "none", "goal": 0},
    {"shot_distance": 11.2, "angle": 42.9, "shot_type": "open_play", "defensive_pressure": "medium", "assist_type": "cross", "goal": 1},
    {"shot_distance": 19.8, "angle": 31.5, "shot_type": "header", "defensive_pressure": "high", "assist_type": "none", "goal": 0},
    {"shot_distance": 7.5, "angle": 63.4, "shot_type": "open_play", "defensive_pressure": "low", "assist_type": "through_ball", "goal": 1},
    {"shot_distance": 26.3, "angle": 12.7, "shot_type": "free_kick", "defensive_pressure": "high", "assist_type": "none", "goal": 0},
    {"shot_distance": 9.4, "angle": 52.8, "shot_type": "header", "defensive_pressure": "medium", "assist_type": "cross", "goal": 1},
    {"shot_distance": 21.7, "angle": 22.3, "shot_type": "open_play", "defensive_pressure": "high", "assist_type": "none", "goal": 0},
    {"shot_distance": 3.8, "angle": 84.9, "shot_type": "penalty", "defensive_pressure": "low", "assist_type": "none", "goal": 1}
]

while len(expanded_shots) < 2000:
    expanded_shots.append(generate_random_shot())

training_data = {"shots": expanded_shots}

try:
    response = requests.post(
        'http://127.0.0.1:5000/train',
        json=training_data
    )
    print("Response status:", response.status_code)
    print("Response content:", response.json())
except Exception as e:
    print("Error occurred:", str(e))