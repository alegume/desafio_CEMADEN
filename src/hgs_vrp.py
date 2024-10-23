import pyvrp
import matplotlib.pyplot as plt
import folium
import requests
import folium
import pandas as pd
import os
from pyvrp import Model
from pyvrp.plotting import plot_coordinates
from pyvrp.stop import MaxRuntime
from pyvrp.plotting import plot_solution
from folium import plugins
from routes_functions import *
from shap_functions import *

script_dir = os.path.dirname(os.path.abspath(__file__))
distancia_file = os.path.join(script_dir, "..", "input", "distancias.csv")
shap_file = os.path.join(script_dir, "..", "input", "SHAP_agregado.csv")
coordenadas_file = os.path.join(script_dir, "..", "input", "coordenadas.csv")

# Load the CSV files
coord = pd.read_csv(coordenadas_file)
data = load_dist_csv(distancia_file)
shap_data = load_shap_csv(shap_file)

# TODO: trabalhar com location e não com sensor  !!!
(shap_score, shap_class) = sum_shap_per_location(shap_data, data['origem_nome'].unique())
locations = pd.concat([data['origem_nome'], data['destino_nome']]).unique()

# TODO: receive depot as input
depot_pos = -1
depot_name = locations[depot_pos]
n = len(locations)

dicionario_loc_to_class = {}
travel_time = {}
travel_dist = {}
for _, row in data.iterrows():
  origin, destination = row['origem_nome'], row['destino_nome']
  travel_time[(origin, destination)] = row['duracao_minutos']
  travel_dist[(origin, destination)] = float(row['distancia'])

# print(travel_time)
# print(travel_dist)
# exit()

# Convert to cartesian coordinates
flat = coord.iat[depot_pos,2]
flon = coord.iat[depot_pos,3]

# Format =  {'location': (lat, long)}
coord_cart = {}
for loc in coord.itertuples():
  (dx, dy) = geo_to_cat(loc.lat, loc.lon, flat, flon)
  coord_cart[loc.name] = (dx, dy)

m = Model()
depot = m.add_depot(
  x = coord_cart[depot_name][0],
  y = coord_cart[depot_name][1]
)

# TODO: pass num_available(=k), days and hours as input
m.add_vehicle_type(
  num_available = 2, 
  start_depot = depot,
  end_depot = depot,
  max_duration = 5*8*60,
)

clients = []
alpha = 10000000
for loc in coord.itertuples():
  clients.append(
    m.add_client(
      x = coord_cart[loc.name][0], 
      y = coord_cart[loc.name][1], 
      service_duration = 240,
      prize = alpha * shap_score[loc.name],
      # True if it is Class A (ClassRank == 0)
      required = (shap_class[loc.name] == 0), 
      # required = False, 
      name = loc.name
    )
  )
  # print(loc.name, " - ", shap_class[loc.name], " - ", shap_class[loc.name] == 0)

locations = [depot] + clients
for frm in locations:
  for to in locations:
    distance = travel_dist.get((frm.name, to.name), 0)
    duration = travel_time.get((frm.name, to.name), 0)
    # print(frm, ' - > ', to, ' = ', distance, " |duration: ", duration)
    if frm == to :
      m.add_edge(frm, to, distance = distance,duration = 0)
    else:
      m.add_edge(frm, to, distance = distance, duration = duration)

res = m.solve(stop=MaxRuntime(2), display=False)
print(res.best)

# Extract the visited locations from the solution
visited_locations = set()
for route in res.best.routes():
  for loc in route:
    visited_locations.add(locations[loc].name)

print(visited_locations)
# Create figure and axis
_, ax = plt.subplots(figsize=(10, 10))

# Plot the solution without plotting clients, as you want to control this manually
plot_solution(res.best, m.data(), plot_clients=False, ax=ax)

dict_color = {0: 'r', 1: 'b', 2: 'y'}

# Loop through all locations and plot them
for loc in coord.itertuples():
  color = 'r' if (loc.name == depot_name) else dict_color[shap_class[loc.name]]
  
  # Check if the location is the depot
  if loc.name == depot_name:
    ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color='r', label='Depot', s=100)
  elif loc.name in visited_locations:
    print("V: ", loc.name)
    # Plot visited locations as dots
    ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color=color, label=f"Classe {shap_class[loc.name]}", s=50)
  else:
    print("NV: ", loc.name)
    # Plot unvisited locations with 'X' marker
    ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color='gray', marker='x', label='Não visitado', s=50)

# Add a legend to show which color represents which class
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicate labels
ax.legend(by_label.values(), by_label.keys())

# Save the plot as an image
fig = os.path.join(script_dir, "..", "output", "solution_TOP2.png")
plt.savefig(fig, dpi=300)

# Optionally display the plot
plt.show()
