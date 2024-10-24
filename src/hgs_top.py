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
from src.routes import *
from src.shap_misc import *
from src.misc_functions import *

def hgs_top(dist_file, shap_file, coord_file, depot_pos = -1,
            n_vehicle = 2, days = 5, hour_per_day = 8, 
            man_time = 240, max_run_time = 5, show_plot = False):

    # Load the CSV files
    coord = pd.read_csv(coord_file)
    data = load_dist_csv(dist_file)
    shap_data = load_shap_csv(shap_file)

    shap_score = shap_data.set_index('Location')['Total_SHAP_Value'].to_dict()
    shap_class = shap_data.set_index('Location')['Class_Rank'].to_dict()
    locations = pd.concat([data['origem_nome'], data['destino_nome']]).unique()

    depot_name = locations[depot_pos]
    travel_time = {}
    travel_dist = {}
    for _, row in data.iterrows():
        origin, destination = row['origem_nome'], row['destino_nome']
        travel_time[(origin, destination)] = row['duracao_minutos']
        travel_dist[(origin, destination)] = float(row['distancia'])

    # Convert to cartesian coordinates
    # Format =    {'location': (lat, long)}
    coord_cart = {}
    flat = coord.iat[depot_pos, 2]
    flon = coord.iat[depot_pos, 3]
    for loc in coord.itertuples():
        (dx, dy) = geo_to_cat(loc.lat, loc.lon, flat, flon)
        coord_cart[loc.name] = (dx, dy)

    ## PyVRP model, depot, vehicle and clients
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
    alpha = 10_000_000
    for loc in coord.itertuples():
        if loc.name == depot_name:
            continue
        clients.append(
            m.add_client(
                x = int(coord_cart[loc.name][0]),
                y = int(coord_cart[loc.name][1]),
                service_duration = int(man_time),
                prize = int(alpha * shap_score[loc.name]),
                # True if it is Class A (ClassRank == 0)
                required = (shap_class[loc.name] == 0),
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

    res = m.solve(stop=MaxRuntime(max_run_time), display=False)

    # Extract the visited locations and metrics from the solution
    print("\t Rotas encontradas: \n")
    visited_locations = set()
    i = 0
    total_time = total_dist = 0
    for route in res.best.routes():
        route_time = travel_time.get((depot_name, locations[route[0]].name), 0)
        route_dist = travel_dist.get((depot_name, locations[route[0]].name), 0)
        i += 1

        print(f"Rota {i}: {depot_name}", end=", ")
        for j in range(len(route)):
            start = locations[route[j]].name
            end = locations[route[j + 1]].name if (j + 1 < len(route)) else depot_name
            visited_locations.add(start)
            print(f"{start}", end=", ")
            route_time += travel_time.get((start, end), 0)
            route_dist += travel_dist.get((start, end), 0)

        formatted_time = format_time(route_time)
        total_time += route_time
        total_dist += route_dist
        print(f"{depot_name}.")
        print(f"Distância: {route_dist:.2f} km")
        print(f"Tempo de viagem: {formatted_time} \n")

    shap_total = 0
    for loc in visited_locations:
        shap_total += shap_score[loc]
    
    formatted_time = format_time(total_time)
    print("\t Sumário:")
    print(f"SHAP agregado total: {shap_total:.4f}")
    print(f"Distância total: {total_dist:.2f} km")
    print(f"Tempo de viagem total: {formatted_time}")


    ## Ploting
    _, ax = plt.subplots(figsize=(10, 10))
    plot_solution(res.best, m.data(), plot_clients=False, ax=ax)

    dict_color = {0: 'r', 1: 'b', 2: 'y'}
    dict_class = {0: 'A', 1: 'B', 2: 'C'}
    # Loop through all locations
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
    by_label = dict(zip(labels, handles))    # Remove duplicate labels
    ax.legend(by_label.values(), by_label.keys())

    # Save the plot as an image
    fig = os.path.join("output", "solution_TOP.png")
    plt.savefig(fig, dpi=300)

    # Optionally display the plot
    if (show_plot):
        plt.show()
