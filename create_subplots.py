"""
This script will generate a single png with a histogram of each feature as a subplot in the graphic
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

# the subplots are being styled, and set up in a 2x5 grid 
sns.set_theme(style='darkgrid')
fig, axs = plt.subplots(2,5, figsize = (25,10))

# these variables represent x and y coordinates for subplots on the overall plot grid
first = 0
second = 0

# for each feature in the list, we will create a subplot of the distribution
for index, feature in enumerate(data_features):
    # creating the plot
    ax = sns.histplot(data = df, x=data_features[index], hue = 'diagnosis', palette = ['r', 'g'], kde = True, ax = axs[first, second])
    
    # formatting the feature string for nicer axis labels
    xlabel_string = data_features[index].replace('_', ' ')
    ax.set(ylabel = 'cell count', xlabel = xlabel_string, title = xlabel_string.replace('mean', "").title())

    # incrementing the subplot coordinates
    if first == 0:
        first = 1
    elif first == 1:
        first = 0
        second += 1

# formatting and saving the final plot that includes subplots for each of the mean features
plt.tight_layout()
file_name = 'subplots.png'
plt.savefig(f'./pngs/{file_name}')