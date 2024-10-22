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

(shap_score, shap_class) = sum_shap_per_location(shap_data, data['origem_nome'].unique())
locations = pd.concat([data['origem_nome'], data['destino_nome']]).unique()

# TODO: receive depot as input
depot_pos = -1
depot_name = locations[depot_pos]
n = len(locations)

dicionario_loc_to_class = {}
travel_time = {}
for _, row in data.iterrows():
  origin, destination = row['origem_nome'], row['destino_nome']
  travel_time[(origin, destination)] = row['duracao_minutos']

sorted_locations_shap = sorted(shap_score.items(), key=lambda x: x[1], reverse=True)

for i, (loc, score) in enumerate(sorted_locations_shap):
  class_rank = shap_class[loc]
  if shap_class[loc] == 0:
    dicionario_loc_to_class.update( {loc: ('A', True)} )
  elif shap_class[loc] == 1:
    dicionario_loc_to_class.update( {loc: ('B', False)} )
  else:
    dicionario_loc_to_class.update( {loc: ('C', False)} )

# Convert to cartesian coordinates
# Cuñapirú is the "center"
# Format (lat, long, 'location')
flat = coord.iat[0,2]
flon = coord.iat[0,3]
# print(flat)
# print(flon)

coord_cart = {}
for loc in coord.itertuples():
  print(loc.lat, loc.lon, loc.name)
  (dx, dy) = geo_to_cat(loc.lat, loc.lon, flat, flon)
  coord_cart[loc.name] = (dx, dy)

# print(coord_cart)
# depot_df = coord[coord["name"] == depot_name]
# print(coord_cart[depot_name][0], coord_cart[depot_name][1])
# exit()

m = Model()
depot = m.add_depot(
  x = coord_cart[depot_name][0],
  y = coord_cart[depot_name][1]
)

# TODO: pass days and hours as input
# TODO: verify inputs !!
m.add_vehicle_type(
  num_available = 2, 
  start_depot = depot,
  end_depot = depot,
  max_duration = 5*8*60,
)

# clients = [
#   m.add_client(
#     x=coord_cart[idx][0], 
#     y=coord_cart[idx][1], 
#     service_duration = 240, prize=10000000*shap_score[coord_cart[idx][2]],required = False, name = coord_cart[idx][2]
#   )
#   for idx in range(n-1)
# ]
# print(clients)
# print("------")

clients = []
alpha = 10000000
for loc in coord.itertuples():
  clients.append(
    m.add_client(
      x = coord_cart[loc.name][0], 
      y = coord_cart[loc.name][1], 
      service_duration = 240,
      prize = alpha * shap_score[loc.name],
      # TODO: review this !!
      # True if it is Class A (ClassRank == 0)
      # required = (shap_class[loc.name] == 0), 
      required = False, 
      name = loc.name
    )
  )
#   print(loc.name, " - ", shap_class[loc.name], " - ", shap_class[loc.name] == 0)
# exit()

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
plot_solution(res.best, m.data(), plot_clients=True, ax=ax)

fig = os.path.join(script_dir, "..", "output", "solution_TOP2.png")
plt.savefig(fig, dpi=300)
plt.show()

## Curvas
# m = Model()

# depot = m.add_depot(x=coord_cart[-1][0], y=coord_cart[-1][1])

# m.add_vehicle_type(
#   2,
#   start_depot=depot,
#   end_depot=depot,
#   max_duration=5*8*60,
# )

# clients = [
#   m.add_client(x=coord_cart[idx][0], y=coord_cart[idx][1], service_duration = 240, prize=10000000*shap_score[coord_cart[idx][2]],required = dicionario_loc_to_class[coord_cart[idx][2]][1], name = coord_cart[idx][2])
#   for idx in range(n-1)
# ]

# locations = [depot] + clients

# for frm in locations:
#   for to in locations:
#     #distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
#     distance = math.dist((frm.x,frm.y), (to.x,to.y))
#     if frm == to :
#       m.add_edge(frm, to, distance=distance,duration = 0)
#     else:
#       m.add_edge(frm, to, distance=distance,duration = travel_time.get((frm, to), 0))

# res = m.solve(stop=MaxRuntime(2), display=True)
# print(res)

# _, ax = plt.subplots(figsize=(8, 8))

# dict_color = {'A': 'r', 'B': 'b', 'C': 'y'}
# plot_solution(res.best, m.data(), ax=ax)

# for i in range(n):
#   ax.scatter(coord_cart[i][0], coord_cart[i][1], color = dict_color[dicionario_loc_to_class[coord_cart[i][2]][0]])

# fig = os.path.join(script_dir, "..", "output", "solution_TOP.png")
# plt.savefig(fig, dpi=300)
# plt.show()
