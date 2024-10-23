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
from datetime import timedelta
from routes_functions import *
from shap_functions import *


def hgs_top(dist_file, shap_file, coord_file, depot_pos = -1,
            n_vehicle = 2, days = 5, hour_per_day = 8, man_time = 240):
  # Load the CSV files
  coord = pd.read_csv(coord_file)
  data = load_dist_csv(dist_file)
  shap_data = load_shap_csv(shap_file)

  # TODO: trabalhar com location e não com sensor  !!!
  (shap_score, shap_class) = sum_shap_per_location(shap_data, data['origem_nome'].unique())
  locations = pd.concat([data['origem_nome'], data['destino_nome']]).unique()

  depot_name = locations[depot_pos]
  travel_time = {}
  travel_dist = {}
  for _, row in data.iterrows():
    origin, destination = row['origem_nome'], row['destino_nome']
    travel_time[(origin, destination)] = row['duracao_minutos']
    travel_dist[(origin, destination)] = float(row['distancia'])

  # Convert to cartesian coordinates
  # Format =  {'location': (lat, long)}
  coord_cart = {}
  flat = coord.iat[depot_pos, 2]
  flon = coord.iat[depot_pos, 3]
  for loc in coord.itertuples():
    (dx, dy) = geo_to_cat(loc.lat, loc.lon, flat, flon)
    coord_cart[loc.name] = (dx, dy)

  # PyVRP model
  m = Model()
  depot = m.add_depot(
    x = coord_cart[depot_name][0],
    y = coord_cart[depot_name][1]
  )

  m.add_vehicle_type(
    num_available = n_vehicle, 
    start_depot = depot,
    end_depot = depot,
    max_duration = days * hour_per_day * 60,
  )

  clients = []
  alpha = 10000000
  for loc in coord.itertuples():
    clients.append(
      m.add_client(
        x = coord_cart[loc.name][0], 
        y = coord_cart[loc.name][1], 
        service_duration = man_time,
        prize = alpha * shap_score[loc.name],
        # True if it is Class A (ClassRank == 0)
        required = (shap_class[loc.name] == 0), 
        # required = False, 
        name = loc.name
      )
    )

  locations = [depot] + clients
  for frm in locations:
    for to in locations:
      distance = travel_dist.get((frm.name, to.name), 0)
      duration = travel_time.get((frm.name, to.name), 0)
      if frm == to :
        m.add_edge(frm, to, distance = distance,duration = 0)
      else:
        m.add_edge(frm, to, distance = distance, duration = duration)

  res = m.solve(stop=MaxRuntime(2), display=False)

  # Extract the visited locations and metrics from the solution
  print("\t Rotas encontradas: \n")
  visited_locations = set()
  i = 0
  for route in res.best.routes():
    total_time = travel_time.get((depot_name, locations[route[0]].name), 0)
    total_dist = travel_dist.get((depot_name, locations[route[0]].name), 0)
    i += 1

    print(f"Rota {i}: {depot_name}", end=", ")

    for j in range(len(route)):
      start = locations[route[j]].name
      end = locations[route[j + 1]].name if (j + 1 < len(route)) else depot_name
      visited_locations.add(start)
      print(f"{start}", end=", ")
      total_time += travel_time.get((start, end), 0)
      total_dist += travel_dist.get((start, end), 0)

    time_delta = timedelta(minutes=total_time)
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes = remainder // 60
    print(f"{depot_name}.")
    print(f"Distância total: {total_dist} km")
    print(f"Tempo de total de viagem: {hours} horas e {minutes} minutos \n")


  # Create figure and axis
  _, ax = plt.subplots(figsize=(10, 10))

  # Plot the solution without plotting clients, as you want to control this manually
  plot_solution(res.best, m.data(), plot_clients=False, ax=ax)

  dict_color = {0: 'r', 1: 'b', 2: 'y'}
  dict_class = {0: 'A', 1: 'B', 2: 'C'}
  # Loop through all locations and plot them
  for loc in coord.itertuples():
    color = 'r' if (loc.name == depot_name) else dict_color[shap_class[loc.name]]
    
    # Check if the location is the depot
    if loc.name == depot_name:
      pass
      # ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color='r', label='Depot', s=100)
    elif loc.name in visited_locations:
      # Plot visited locations as dots
      ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color=color, label=f"Classe {dict_class[shap_class[loc.name]]}", s=50)
    else:
      # Plot unvisited locations with 'X' marker
      ax.scatter(coord_cart[loc.name][0], coord_cart[loc.name][1], color=color, marker='x', label='Não visitado', s=50)

  # Add a legend to show which color represents which class
  handles, labels = ax.get_legend_handles_labels()
  by_label = dict(zip(labels, handles))  # Remove duplicate labels
  ax.legend(by_label.values(), by_label.keys())

  # Save the plot as an image
  fig = os.path.join(script_dir, "..", "output", "solution_TOP2.png")
  plt.savefig(fig, dpi=300)

  # Optionally display the plot
  plt.show()


if __name__ == "__main__":
  script_dir = os.path.dirname(os.path.abspath(__file__))
  dist_file = os.path.join(script_dir, "..", "input", "distancias.csv")
  shap_file = os.path.join(script_dir, "..", "input", "SHAP_agregado.csv")
  coord_file = os.path.join(script_dir, "..", "input", "coordenadas.csv")

  hgs_top(
    dist_file    = dist_file,
    shap_file    = shap_file,
    coord_file   = coord_file,
    depot_pos    = -1,
    n_vehicle    = 2, 
    days         = 5, 
    hour_per_day = 8,
    man_time     = 240
  )