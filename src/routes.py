import math
import pandas as pd

def geo_to_cat(lat1, lon1, lat2 = 0, lon2 = 0):
    # 1-> origem ; 2-> ponto móvel
    dx = (lon1-lon2)*40000*math.cos((lat1+lat2)*math.pi/360)/360
    dy = (lat1-lat2)*40000/360
    return(dx,dy)

# Load data from CSV
def load_dist_csv(file_path):
    df = pd.read_csv(file_path)
    df['duracao_minutos'] = df['duracao'].apply(parse_duration)
    return df

# Function to parse time from string to total minutes
def parse_duration(duration_str):
    time_parts = duration_str.split()
    hours = 0  # Initialize hours to 0
    if 'hours' in duration_str: # Check if hours are present
        hours = int(time_parts[0].replace('hours', '').strip())
        minutes = int(time_parts[2].replace('mins', '').strip())
    elif 'hour' in duration_str: # Check if hours are present
        hours = int(time_parts[0].replace('hour', '').strip())
        minutes = int(time_parts[2].replace('mins', '').strip())
    else:
        minutes = int(time_parts[0].replace('mins', '').strip()) # Extract minutes directly if < 1h
    return hours * 60 + minutes