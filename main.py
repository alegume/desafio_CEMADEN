import os
import pandas as pd

import src.hgs_top as hgs
import src.shap_classes as sh
import src.routes as routes


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dist_file = os.path.join(script_dir, "input", "distancias.csv")
    # shap_file = os.path.join(script_dir, "input", "SHAP_agregado.csv")
    shap_file = os.path.join(script_dir, "output", "shap_agregado_classes_locations.csv")
    coord_file = os.path.join(script_dir, "input", "coordenadas.csv")
    model_file = os.path.join(script_dir, "input", "model.keras")
    data_test_file = os.path.join(script_dir, "input", "data_test.csv")
    depot_pos = -1

    ## Evaluate sensores (SHAP, Classes) and generate plots
    dist = routes.load_dist_csv(dist_file)
    locations = pd.concat([dist['origem_nome'], dist['destino_nome']]).unique()
    locations = locations[locations != 'Villa_Soriano']
    
    test = pd.read_csv(data_test_file)
    sh.save_shap_with_location(model_file, test.values, locations)


    ## Solve TOP using HGS 
    # hgs.hgs_top(
    #     dist_file = dist_file,
    #     shap_file = shap_file,
    #     coord_file = coord_file,
    #     depot_pos = depot_pos,
    #     n_vehicle = 2, 
    #     days = 5, 
    #     hour_per_day = 8,
    #     man_time = 240,
    #     max_run_time = 2,
    #     show_plot = False
    # )


if __name__ == "__main__":
    main()
