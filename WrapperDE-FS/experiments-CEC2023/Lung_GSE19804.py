dataset_file   = "ARTIGO1/Lung_GSE19804/Lung_GSE19804.csv" # path to data file (must be .csv, features as columns, first row and first column are labels, first column after labels should contain the classes or target values)
task           = "classification" # "classification" or "regression"
class_label    = "type"              # label of the column with the classes or target values
dataset_sep    = ","              # use ',' to separate columns in the dataset
output_folder  = 'RESULTS/ARTIGO1'        # name of directory in which the results will be saved
row_index      = 0                # The column that has the row index, None if no index

cv_splits      = None

standardized   = False # True if data should be normalized with the z-norm (M=0.0, std=1.0)
rescaled       = False # True if data should be scaled between 0 and 1
k 			   = 1
#debug		   = 1
runs		   = 16
