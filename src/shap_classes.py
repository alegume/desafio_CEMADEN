import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import shap
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.stats import norm, probplot
from scipy.stats import shapiro, kstest, norm
# from sklearn.metrics import mean_squared_error
from src.misc_functions import *
from src.shap_misc import *

# Must be used the order of traing of the model
features = [
    'Manuel_Díaz_precipitation',
    'Manuel_Díaz_level',
    'Cuñapirú_precipitation',
    'Cuñapirú_level',
    'Coelho_precipitation',
    'Coelho_level',
    'Paso_de_las_Toscas_precipitation',
    'Paso_de_las_Toscas_level',
    'Mazagano_precipitation',
    'Mazagano_level',
    'Aguiar_precipitation',
    'Aguiar_level',
    'Pereira_precipitation',
    'Pereira_level',
    'Laguna_I_precipitation',
    'Laguna_I_level',
    'Laguna_II_precipitation',
    'Laguna_II_level',
    'San_Gregorio_precipitation',
    'San_Gregorio_level',
    'Bonete_precipitation',
    'Bonete_level',
    'Paso_de_los_toros_precipitation',
    'Paso_de_los_toros_level',
    'Salsipuedes_precipitation',
    'Salsipuedes_level',
    'Sarandi_del_Yi_precipitation',
    'Sarandi_del_Yi_level',
    'Polanco_precipitation',
    'Polanco_level',
    'Durazno_precipitation',
    'Durazno_level',
    'Paso_de_Lugo_precipitation',
    'Paso_de_Lugo_level',
    'Mercedes_precipitation',
    'Mercedes_level'
]

# features = sorted(features)
# all_columns = df.columns.tolist()
# columns_to_exclude = ['id', 'dt', 'dt.1', 'Villa_Soriano_level', 'Villa_Soriano_precipitation']
# features = [column for column in all_columns if column not in columns_to_exclude]
# print(features)
# exit()


def split_data_test(df):
    # configure the data index
    df['dt'] = pd.to_datetime(df['dt'], dayfirst=True)
    data = df.set_index('dt')
    data.index = pd.to_datetime(data.index)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    x = scaler.fit_transform(data[features].values)

    # Considera como salida los datos de nivel para la estacion de Villa Soriano para el tiempo t + 1
    # y = x[:,35]
    date_split = pd.to_datetime(datetime.datetime(year=2019, month=3, day=15, hour=7, minute=0))
    train_index = data.loc[data.index < date_split]
    i = 1
    ## Remove o ultimo tempo do df de x e o primeiro do df de y
    in_data = x[:-i]
    # out_data = y[i:]

    ## Split test data
    x_test = in_data[train_index.shape[0]:, ]
    # Create a Pandas DataFrame with headers
    x_test_df = pd.DataFrame(x_test, columns=features)

    return x_test_df


### Save SHAP summary and SHAP values
def save_shap(model, x_test):
    explainer = shap.DeepExplainer(model, x_test)
    shap_values = explainer.shap_values(x_test)
    # aux = explainer(x_test)
    shap.plots.initjs()
    sv = shap_values[:,:,0]
    xv = pd.DataFrame(x_test, columns=features)
    # print(xv)
    # shap.summary_plot(sv,xv, max_display=8)
    shap.summary_plot(sv, xv, show=False)
    plt.savefig('output/shap_summary_plot.png')

    ### Classificação SHAP
    # Calculate the mean absolute SHAP value for each feature
    mean_shap_values = np.mean(np.abs(sv), axis=0)
    # Create thresholds for classification in 3 classes
    high_threshold = np.percentile(mean_shap_values, 66)
    low_threshold = np.percentile(mean_shap_values, 33)

    # Classify features into A, B, C
    feature_classes = pd.DataFrame({'Feature': features, 'Mean_SHAP_Value': mean_shap_values})
    feature_classes['Classe'] = pd.cut(feature_classes['Mean_SHAP_Value'],
    bins=[-np.inf, low_threshold, high_threshold, np.inf],
    labels=['C', 'B', 'A'])

    # Sort by Class (A > B > C) and within each class by Mean SHAP Value in descending order
    feature_classes['Class_Rank'] = feature_classes['Classe'].map({'A': 0, 'B': 1, 'C': 2})
    feature_classes = feature_classes.sort_values(by=['Class_Rank', 'Mean_SHAP_Value'], ascending=[True, False])

    # Output classification as string
    classification_result = feature_classes[['Feature', 'Classe']].to_string(index=False)
    # print(classification_result)

    # Plot the feature importance with classification
    plt.figure(figsize=(10, 6))
    colors = {'A': '#FF6347', 'B': '#FFA500', 'C': '#32CD32'} 

    # Reverse the feature_classes dataframe to ensure correct ordering on y-axis
    for feature_class, group_data in feature_classes.groupby('Classe'):
        plt.barh(
            group_data['Feature'][::-1], 
            group_data['Mean_SHAP_Value'][::-1],
            color = colors[feature_class],
            label=f'Classe {feature_class}'
        )

    plt.xlabel('Valores SHAP agregado')
    plt.ylabel('Sensor')
    plt.title('Classificação baseada nos valores SHAP')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/shap_classes.png')
    # plt.show()

    # save feature_classes to csv
    feature_classes.to_csv('output/shap_agregado_classes.csv', index=False)


def save_shap_with_location(model, x_test, locations):
    explainer = shap.DeepExplainer(model, x_test)
    shap_values = explainer.shap_values(x_test)
    shap.plots.initjs()
    sv = shap_values[:,:,0]
    xv = pd.DataFrame(x_test, columns=features)

    # Save SHAP summary plot for individual features
    shap.summary_plot(sv, xv, show=False)
    plt.savefig('output/shap_summary_plot.png')

    # Calculate the mean absolute SHAP value for each feature
    mean_shap_values = np.mean(np.abs(sv), axis=0)

    # Create a dataframe to hold features and their mean SHAP values
    feature_classes = pd.DataFrame({'Feature': features, 'Mean_SHAP_Value': mean_shap_values})
    
    ### Aggregate SHAP values per location
    shap_score = sum_shap_per_location(feature_classes, locations)

    # Convert shap_score into a DataFrame for classification
    shap_location_df = pd.DataFrame({
        'Location': shap_score.keys(),
        'Total_SHAP_Value': shap_score.values()
    })

    ### Classify locations into A, B, C based on their aggregated SHAP values
    high_threshold = np.percentile(shap_location_df['Total_SHAP_Value'], 66)
    low_threshold = np.percentile(shap_location_df['Total_SHAP_Value'], 33)

    # Classify locations into A, B, C based on SHAP values
    shap_location_df['Classe'] = pd.cut(shap_location_df['Total_SHAP_Value'],
                                        bins=[-np.inf, low_threshold, high_threshold, np.inf],
                                        labels=['C', 'B', 'A'])

    # Sort by Class (A > B > C) and within each class by Total SHAP Value in descending order
    shap_location_df['Class_Rank'] = shap_location_df['Classe'].map({'A': 0, 'B': 1, 'C': 2})
    shap_location_df = shap_location_df.sort_values(by=['Class_Rank', 'Total_SHAP_Value'], ascending=[True, False])

    # Save SHAP aggregated 
    shap_location_df.to_csv('output/shap_agregado_classes_locations.csv', index=False)

    plt.figure(figsize=(10, 6))
    colors = {0: '#FF6347', 1: '#FFA500', 2: '#32CD32'}
    sorted_locations = shap_location_df.sort_values(by='Total_SHAP_Value', ascending=False)

    for class_rank, group_data in sorted_locations.groupby('Class_Rank'):
        plt.barh(
            # reverse to correct order
            group_data['Location'][::-1],  
            group_data['Total_SHAP_Value'][::-1],
            color=colors[class_rank],
            label=f'Classe {["A", "B", "C"][class_rank]}'
        )

    plt.xlabel('Total SHAP Value per Location')
    plt.ylabel('Location')
    plt.title('Classificação por Localização baseada nos valores SHAP')
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/shap_classes_locations.png')



##### MAIN
if __name__ == "__main__":
    # Load the saved model and dataset
    model = load_model('input/model.keras')
    df = pd.read_csv('input/df_level_rain_fill.csv')
    # Save test data
    x_test = split_data_test(df)
    x_test.to_csv('output/data_test.csv', index=False)


"""
# Kmeans Clustering
# Calculate the mean absolute SHAP value for each feature
mean_shap_values = np.mean(np.abs(sv), axis=0)

# Reshape the data for clustering
mean_shap_values_reshaped = mean_shap_values.reshape(-1, 1)

# Apply KMeans to divide into 3 clusters (A, B, C)
kmeans = KMeans(n_clusters=3, random_state=2)
kmeans.fit(mean_shap_values_reshaped)
clusters = kmeans.predict(mean_shap_values_reshaped)

# Create a DataFrame to store feature names, SHAP values, and cluster assignments
feature_classes = pd.DataFrame({'Feature': features, 'Mean_SHAP_Value': mean_shap_values, 'Cluster': clusters})

# Map clusters to classes A, B, C based on the mean SHAP value in each cluster
cluster_means = feature_classes.groupby('Cluster')['Mean_SHAP_Value'].mean()
class_mapping = cluster_means.sort_values(ascending=False).index  # Sort clusters by mean SHAP value
feature_classes['Classe'] = feature_classes['Cluster'].map({class_mapping[0]: 'A', class_mapping[1]: 'B', class_mapping[2]: 'C'})

# Sort by Class (A > B > C) and within each class by Mean SHAP Value in descending order
feature_classes['Class_Rank'] = feature_classes['Classe'].map({'A': 0, 'B': 1, 'C': 2})
feature_classes = feature_classes.sort_values(by=['Class_Rank', 'Mean_SHAP_Value'], ascending=[True, False])

# Output classification as string
classification_result = feature_classes[['Feature', 'Classe']].to_string(index=False)
print(classification_result)
# Sort the DataFrame by 'Mean_SHAP_Value' in descending order (if not already)
feature_classes = feature_classes.sort_values(by=['Mean_SHAP_Value'], ascending=False)

# Plot the feature importance with classification
plt.figure(figsize=(10, 6))
colors = {'A': '#FF6347', 'B': '#FFA500', 'C': '#32CD32'}

# Loop over each class ('A', 'B', 'C') and plot bars in decreasing order
for feature_class, group_data in feature_classes.groupby('Classe'):
    # The features are already sorted, so we reverse them for proper horizontal bar plotting
    plt.barh(group_data['Feature'], group_data['Mean_SHAP_Value'],
             color=colors[feature_class], label=f'Classe {feature_class}')

plt.xlabel('Valores SHAP')
plt.ylabel('Feature')
plt.title('Classificação baseada nos valores SHAP (K-Means)')
plt.legend()

# Reverse the y-axis so that the highest SHAP values are at the top
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
"""


"""
### Código para plotar histograma e gráfico Q-Q
# Calcular os valores médios absolutos dos SHAP
mean_shap_values = np.mean(np.abs(sv), axis=0)

# Plotar o histograma com a curva de uma distribuição normal ajustada
plt.figure(figsize=(12, 6))

# Histograma dos valores SHAP
sns.histplot(mean_shap_values, bins=20, kde=False, stat='density', color='skyblue', label='Valores SHAP')

# Ajustar e plotar a curva da distribuição normal sobre o histograma
mu, std = norm.fit(mean_shap_values)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label='Curva Normal Ajustada')

plt.title('Histograma dos Valores SHAP com Curva Normal Ajustada')
plt.xlabel('Valores SHAP')
plt.ylabel('Densidade')
plt.legend()
plt.show()

# Gerar o gráfico Q-Q para verificar a normalidade
plt.figure(figsize=(6, 6))
probplot(mean_shap_values, dist="norm", plot=plt)
plt.title('Gráfico Q-Q para Verificar Normalidade')
plt.show()

## Testes de normalidade
# Calcular os valores médios absolutos dos SHAP
mean_shap_values = np.mean(np.abs(sv), axis=0)

# Teste de Shapiro-Wilk
shapiro_test = shapiro(mean_shap_values)
print(f"Shapiro-Wilk Test: Estatística={shapiro_test.statistic}, p-valor={shapiro_test.pvalue}")

# Teste de Kolmogorov-Smirnov comparando com uma distribuição normal
# Ajuste da média e desvio padrão para a distribuição normal
mu, std = norm.fit(mean_shap_values)
ks_test = kstest(mean_shap_values, 'norm', args=(mu, std))
print(f"Kolmogorov-Smirnov Test: Estatística={ks_test.statistic}, p-valor={ks_test.pvalue}")
"""