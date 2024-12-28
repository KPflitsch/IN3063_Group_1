import os
import pandas as pd

current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, 'task_2_dataset/london_weather_clean.csv'))

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:

    weather_data = pd.read_csv(file_path)
    bad_data = weather_data[weather_data['max_temp'] < weather_data['min_temp']]
    print(f"Number of problematic rows: {len(bad_data)}")
    print(bad_data)
