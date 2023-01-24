import pandas as pd

# assign dataset names
list_of_names = ["Breast_GSE70947", "Colorectal_GSE44076", "Liver_GSE14520_U133A", "Lung_GSE19804", "Prostate_GSE6919_U95B", "Renal_GSE53757", "Throat_GSE42743"]
 
# create empty list
dataframes_list = []
index = []
 
# append datasets into the list
for i in range(len(list_of_names)):
    temp_df = pd.read_csv("/home/dominico/Desktop/weighted_tSNE-master/RESULTS/"+list_of_names[i]+"/selectors_silhouette_tsne2d.csv")
    index.append(temp_df.loc[temp_df['Weighted silhouette'].idxmax()])
    dataframes_list.append(temp_df)

for i in range(len(list_of_names)):
    #print(dataframes_list[i])
    print(index[i])

#df = pd.read_csv("Throat_GSE42743/selectors_silhouette_tsne2d.csv")
#print(df)
#index = df.loc[df['Embedding silhouette'].idxmax()]
#print(index)

print(index[i][1])

