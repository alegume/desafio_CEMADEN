import pyvrp
import matplotlib.pyplot as plt
import folium
import requests
import folium
import math
import pandas as pd
import os

from pyvrp import Model
from pyvrp.plotting import plot_coordinates
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution
from folium import plugins

script_dir = os.path.dirname(os.path.abspath(__file__))
distancia_file = os.path.join(script_dir, "..", "input", "distancias.csv")
shap_file = os.path.join(script_dir, "..", "input", "SHAP_agregado.csv")
coordenadas_file = os.path.join(script_dir, "..", "input", "coordenadas.csv")

# Load the CSV file
df = pd.read_csv(coordenadas_file)

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

# Load data from CSV
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df['duracao_minutos'] = df['duracao'].apply(parse_duration)
    return df


def sum_shap_per_location(shap_data, locations):
  shap_score = {loc: 0 for loc in locations}

  # Calculate the total SHAP value for each location
  for _, row in shap_data.iterrows():
    location_name = '_'.join(row['Feature'].split('_')[:-1])  # Extract the full location name
    if location_name in shap_score:
      shap_score[location_name] += row['Mean_SHAP_Value']

  # Sort the locations by their SHAP score
  sorted_locations = sorted(shap_score.items(), key=lambda x: x[1], reverse=True)

  # Determine thresholds for class assignments
  num_locations = len(sorted_locations)
  class_0_threshold = num_locations // 3
  class_1_threshold = 2 * class_0_threshold

  shap_class = {}

  for i, (loc, score) in enumerate(sorted_locations):
    if i < class_0_threshold:
      shap_class[loc] = 0  # Highest SHAP scores
    elif i < class_1_threshold:
      shap_class[loc] = 1  # Medium SHAP scores
    else:
      shap_class[loc] = 2  # Lowest SHAP scores

  return shap_score, shap_class


# Function to load SHAP data
def load_shap_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def geo_to_cat(lat1, lon1, lat2 = 0, lon2 = 0):
  # 1-> origem
  # 2-> ponto móvel

  dx = (lon1-lon2)*40000*math.cos((lat1+lat2)*math.pi/360)/360
  dy = (lat1-lat2)*40000/360
  return(dx,dy)

# distancia_file = "distancias.csv"
# shap_file = "SHAP_agregado.csv"
df = pd.read_csv(coordenadas_file)

data = load_csv(distancia_file)
# Read the SHAP data
shap_data = load_shap_csv(shap_file)
(shap_score, shap_class) = sum_shap_per_location(shap_data, data['origem_nome'].unique())
locations = pd.concat([data['origem_nome'], data['destino_nome']]).unique()
depot_pos = -1
depot = locations[depot_pos]
n = len(locations)

dicionario_loc_to_int = {}
dicionario_loc_to_class = {}

i = 1
for loc in locations:
  dicionario_loc_to_int.update({loc: i})
  i = i+1

travel_time = {}
for _, row in data.iterrows():
  origin, destination = row['origem_nome'], row['destino_nome']
  travel_time[(origin, destination)] = row['duracao_minutos']

sorted_locations_shap = sorted(shap_score.items(), key=lambda x: x[1], reverse=True)

num_locations = len(sorted_locations_shap)
class_0_threshold = num_locations // 3
class_1_threshold = 2 * class_0_threshold

for i, (loc, score) in enumerate(sorted_locations_shap):
  if i < class_0_threshold:
    dicionario_loc_to_class.update( {loc: ('A', True)} )
  elif i < class_1_threshold:
    dicionario_loc_to_class.update( {loc: ('B', False)} )
  else:
    dicionario_loc_to_class.update( {loc: ('C', False)} )

coord_cart = []
flat = df.iat[0,2]
flon = df.iat[0,3]

for i in range (n):
  (dx,dy) = geo_to_cat(df.iat[i,2], df.iat[i,3], flat, flon)
  coord_cart.append((dx,dy,locations[i]))

m = Model()

depot = m.add_depot(x=coord_cart[-1][0], y=coord_cart[-1][1])
m.add_vehicle_type(
    3,
    start_depot=depot,
    end_depot=depot,
    max_duration=5*8*60,
)

clients = [
    m.add_client(x=coord_cart[idx][0], y=coord_cart[idx][1], service_duration = 240, prize=10000000*shap_score[coord_cart[idx][2]],required = False, name = coord_cart[idx][2])
    for idx in range(n-1)
]

locations = [depot] + clients

for frm in locations:
    for to in locations:
        #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
        distance = math.dist((frm.x,frm.y), (to.x,to.y))
        if frm == to :
          m.add_edge(frm, to, distance=distance,duration = 0)
        else:
          m.add_edge(frm, to, distance=distance,duration = travel_time.get((frm, to), 0))

res = m.solve(stop=MaxRuntime(3), display=False)
print(res)

_, ax = plt.subplots(figsize=(8, 8))
plot_solution(res.best, m.data(), plot_clients=True, ax=ax)

# Curvas
m = Model()

depot = m.add_depot(x=coord_cart[-1][0], y=coord_cart[-1][1])
m.add_vehicle_type(
    2,
    start_depot=depot,
    end_depot=depot,
    max_duration=5*8*60,
)

clients = [
    m.add_client(x=coord_cart[idx][0], y=coord_cart[idx][1], service_duration = 240, prize=10000000*shap_score[coord_cart[idx][2]],required = dicionario_loc_to_class[coord_cart[idx][2]][1], name = coord_cart[idx][2])
    for idx in range(n-1)
]

locations = [depot] + clients

for frm in locations:
    for to in locations:
        #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
        distance = math.dist((frm.x,frm.y), (to.x,to.y))
        if frm == to :
          m.add_edge(frm, to, distance=distance,duration = 0)
        else:
          m.add_edge(frm, to, distance=distance,duration = travel_time.get((frm, to), 0))

res = m.solve(stop=MaxRuntime(2), display=True)
print(res)

_, ax = plt.subplots(figsize=(8, 8))

dict_color = {'A': 'r', 'B': 'b', 'C': 'y'}
plot_solution(res.best, m.data(), ax=ax)

for i in range(n):
  ax.scatter(coord_cart[i][0], coord_cart[i][1], color = dict_color[dicionario_loc_to_class[coord_cart[i][2]][0]])

fig = os.path.join(script_dir, "..", "output", "solution_TOP.png")
plt.savefig(fig, dpi=300)
plt.show()
