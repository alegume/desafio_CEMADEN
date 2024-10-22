import pandas as pd

# Function to load SHAP data
def load_shap_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def sum_shap_per_location(shap_data, locations):
  shap_score = {loc: 0 for loc in locations}
  shap_class = {loc: float('inf') for loc in locations}  # Start with high values for Class_Rank

  # Calculate the total SHAP value for each location and track the minimum Class_Rank
  for _, row in shap_data.iterrows():
    location_name = '_'.join(row['Feature'].split('_')[:-1])  # Extract the full location name

    if location_name in shap_score:
      shap_score[location_name] += row['Mean_SHAP_Value']

      # Update the shap_class with the minimum Class_Rank for the location
      shap_class[location_name] = min(shap_class[location_name], row['Class_Rank'])

  # Sort the locations by their SHAP score
  sorted_locations = sorted(shap_score.items(), key=lambda x: x[1], reverse=True)

  # Create the final class assignments based on the lowest Class_Rank
  final_class_assignments = {}
  for loc, score in sorted_locations:
    final_class_assignments[loc] = shap_class[loc]  # Use the minimum Class_Rank for this location

  return shap_score, final_class_assignments

# def sum_shap_per_location(shap_data, locations):
#   shap_score = {loc: 0 for loc in locations}

#   # Calculate the total SHAP value for each location
#   for _, row in shap_data.iterrows():
#     location_name = '_'.join(row['Feature'].split('_')[:-1])  # Extract the full location name
#     if location_name in shap_score:
#       shap_score[location_name] += row['Mean_SHAP_Value']

#   # Sort the locations by their SHAP score
#   sorted_locations = sorted(shap_score.items(), key=lambda x: x[1], reverse=True)

#   # Determine thresholds for class assignments
#   num_locations = len(sorted_locations)
#   class_0_threshold = num_locations // 3
#   class_1_threshold = 2 * class_0_threshold

#   shap_class = {}

#   for i, (loc, score) in enumerate(sorted_locations):
#     if i < class_0_threshold:
#       shap_class[loc] = 0  # Highest SHAP scores
#     elif i < class_1_threshold:
#       shap_class[loc] = 1  # Medium SHAP scores
#     else:
#       shap_class[loc] = 2  # Lowest SHAP scores

#   return shap_score, shap_class