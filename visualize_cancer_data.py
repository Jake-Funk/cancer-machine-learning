"""
This script will generate individual histograms of each feature in the cancer data set and save them as pngs
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# First the data is being loaded into a dataframe from a csv file
#   then unnecessary columns are being dropped
df = pd.read_csv("./data/wisconsin_cancer_data.csv")
df.drop('id', axis=1, inplace=True)
df.drop('Unnamed: 32', axis=1, inplace=True)

# here the diagnosis designation is being mapped to the words they represent, for clarity
df['diagnosis'] = df['diagnosis'].map({'M': 'malignant', 'B': 'benign'})

# after dropping some columns, we can create a sublist of features that each represent a mean value
data_features = list(df.columns[1:11])

# setting the plot theme
sns.set_theme(style='darkgrid')

# for each feature, generate in individual png of the data distribution
for feature in data_features:
    # creating the plot
    ax = sns.displot(data=df, x=feature, hue='diagnosis', palette = ['r', 'g'], kde = True)
    
    # editing the plot format
    xlabel_string = feature.replace('_', ' ')
    ax.set(ylabel = 'cell count', xlabel = xlabel_string, title = xlabel_string.replace('mean', "").title())
    plt.tight_layout()
    sns.move_legend(ax,loc='upper right', title="Diagnosis", frameon = True)
    
    # saving the histogram as a png
    file_name = f'{feature}_histogram.png'
    plt.savefig(f'./pngs/{file_name}')