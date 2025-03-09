from statsbombpy import sb
import pandas as pd
import numpy as np
import requests

# Function to calculate distance and angle from location
def calculate_distance_angle(x, y):
    goal_x, goal_y = 120, 40  # StatsBomb pitch: 120m x 80m, goal at (120, 40)
    distance = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
    angle = np.degrees(np.arctan2(abs(goal_y - y), goal_x - x))
    return distance, angle

# Define competitions to use
competitions_to_use = [
    (43, 3),   # 2018 FIFA World Cup
    (11, 90),  # La Liga 2021/2022
]

# Collect shots from all specified competitions
all_shots = []
for comp_id, season_id in competitions_to_use:
    print(f"Processing Competition ID: {comp_id}, Season ID: {season_id}")
    matches = sb.matches(competition_id=comp_id, season_id=season_id)
    print(f"Found {len(matches)} matches")
    
    for match_id in matches['match_id']:
        try:
            events = sb.events(match_id=match_id)
            shots = events[events['type'] == 'Shot']
            if not shots.empty:
                shots['shot_distance'] = shots['location'].apply(lambda loc: calculate_distance_angle(loc[0], loc[1])[0])
                shots['angle'] = shots['location'].apply(lambda loc: calculate_distance_angle(loc[0], loc[1])[1])
                shots['shot_type'] = shots['shot_type'].fillna('open_play')
                shots['defensive_pressure'] = shots['under_pressure'].apply(lambda x: 'high' if x else 'low')
                shots['goal'] = shots['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)
                shots['assist_type'] = shots['shot_key_pass_id'].apply(lambda x: 'through_ball' if pd.notna(x) else 'none')
                all_shots.append(shots[['shot_distance', 'angle', 'shot_type', 'defensive_pressure', 'assist_type', 'goal']])
        except Exception as e:
            print(f"Error processing match {match_id}: {str(e)}")
            continue

# Combine all shots into a single DataFrame
shots_df = pd.concat(all_shots).dropna()
print(f"Total shots collected: {len(shots_df)}")

# Optional: Subsample to ~5,000-6,000 if desired
if len(shots_df) > 6000:
    shots_df = shots_df.sample(n=6000, random_state=42)
    print(f"Subsampled to: {len(shots_df)} shots")

# Save to CSV for reference (optional)
shots_df.to_csv('statsbomb_shots_5000.csv', index=False)

# Prepare training data for your Flask app
training_data = {"shots": shots_df.to_dict(orient='records')}

# Send to Flask app
try:
    response = requests.post('http://127.0.0.1:5000/train', json=training_data)
    print("Response status:", response.status_code)
    print("Response content:", response.json())
except Exception as e:
    print("Error occurred:", str(e))