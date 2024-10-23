import os

import src.hgs_top as hgs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dist_file = os.path.join(script_dir, "input", "distancias.csv")
    shap_file = os.path.join(script_dir, "input", "SHAP_agregado.csv")
    coord_file = os.path.join(script_dir, "input", "coordenadas.csv")

    hgs.hgs_top(
        dist_file = dist_file,
        shap_file = shap_file,
        coord_file = coord_file,
        depot_pos = -1,
        n_vehicle = 2, 
        days = 5, 
        hour_per_day = 8,
        man_time = 240,
        max_run_time = 2,
        show_plot = False
    )


if __name__ == "__main__":
    main()
