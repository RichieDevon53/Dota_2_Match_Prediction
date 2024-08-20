# Library untuk data manipulasi
import pandas as pd
import numpy as np

# Library varname untuk extract nama dari variabel
from varname import varname

# Library export dan import model, scaling dan encoding
import pickle
import json

# Library pengolahan statistik
from scipy import stats
from scipy.stats import kruskal
from scipy.sparse import csr_matrix

# Library pengolahan outlier
from feature_engine.outliers import Winsorizer

# Library pengolahan menggunakan machine learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, silhouette_score,silhouette_samples
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Library Visualisasi
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.gridspec import GridSpec
from IPython.display import display, HTML
from PIL import Image

# def plot_scatter(dataframe, kolom_1, kolom_2):

def plot_kluster(dataframe, name_column_cluster):
    kluster = sorted(dataframe[name_column_cluster].unique())
    for col in dataframe.drop(name_column_cluster, axis=1).columns:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(18, 4)

        ax1.grid(axis='y', linestyle='--', alpha=0.6)
        for index, unique in enumerate(kluster):
            ax1.bar(index + 1, dataframe[dataframe[name_column_cluster] == unique][col].mean(), edgecolor='black')
            
        ax1.set_xlabel('Kluster')
        ax1.set_ylabel('Rata-rata')
        ax1.set_xticks([int(x+1) for x in range(0, len(kluster))], [int(x) for x in range(0, len(kluster))])
        ax1.set_title(f'Grafik Rata-rata kluster')
        
        ax2.grid(axis='y', linestyle='--', alpha=0.6)
        for index, unique in enumerate(kluster):
            ax2.bar(index + 1, dataframe[dataframe[name_column_cluster] == unique][col].median(), edgecolor='black')
            
        ax2.set_xlabel('Kluster')
        ax2.set_ylabel('Median')
        ax2.set_xticks([int(x+1) for x in range(0, len(kluster))], [int(x) for x in range(0, len(kluster))])
        ax2.set_title(f'Garfik Median kluster')
        
        box_data = [dataframe[dataframe[name_column_cluster] == kluster][col] for kluster in kluster]
        ax3.boxplot(box_data, positions=range(1, len(kluster) + 1))
        ax3.set_xlabel('Kluster')
        ax3.set_ylabel('Value')
        ax3.set_xticks([int(x+1) for x in range(0, len(kluster))], [int(x) for x in range(0, len(kluster))])
        ax3.set_title(f'Boxplot kluster')
        
        fig.suptitle(f'Analisis Kluster pada Kolom {col}', fontsize=20, y=1.02)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

def plot_silhouette(range_n_clusters, X, random_state):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 4)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters = n_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = random_state)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')


# Function pembuatan dataframe analisis persentase outlier
def analisis_outlier(data):
    # Pembuatan dataframe beisi nama kolom, persentase outlier, jenis distribusi dan skew
    result = pd.DataFrame()
    result['Nama_kolom'] = data.columns
    result['Persentase_outlier'] = None
    result['Jumlah_outlier'] = None
    result['Jenis_distribusi'] = None
    result['Skew'] = None
    
    # Looping setiap kolom pada dataframe
    for col in data.columns:
        try:
            # Penentuan berdasarkan jenis ekstreme skew
            if (data[col].skew() > 1) or (data[col].skew() < -1):
                
                # Penentuan lower boundaries dan upper boundaries
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                lower = q1 - (3*(q3 - q1))
                upper = q3 + (3*(q3 - q1))
                
                # Pemisahan data outlier
                outlier = data[col][(data[col] < lower) | (data[col] > upper)]
                
                # Perhitungan persentasse outlier
                total_count = len(data[col])
                percentage = (len(outlier) / total_count) * 100
                
                # input data ke dataframe utama
                result.loc[result['Nama_kolom'] == col, 'Persentase_outlier'] = percentage
                result.loc[result['Nama_kolom'] == col, 'Jumlah_outlier'] = len(outlier)
                result.loc[result['Nama_kolom'] == col, 'Jenis_distribusi'] = 'Extreme Skew'
                result.loc[result['Nama_kolom'] == col, 'Skew'] = data[col].skew()
                
            # Penentuan berdasarkan jenis skew
            elif (data[col].skew() > 0.5) or (data[col].skew() < -0.5):
                
                # Penentuan lower boundaries dan upper boundaries
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                lower = q1 - (1.5*(q3 - q1))
                upper = q3 + (1.5*(q3 - q1))
                
                # Pemisahan data outlier
                outlier = data[col][(data[col] < lower) | (data[col] > upper)]
                
                # Perhitungan persentasse outlier
                total_count = len(data[col])
                percentage = (len(outlier) / total_count) * 100
                
                # input data ke dataframe utama
                result.loc[result['Nama_kolom'] == col, 'Persentase_outlier'] = percentage
                result.loc[result['Nama_kolom'] == col, 'Jumlah_outlier'] = len(outlier)
                result.loc[result['Nama_kolom'] == col, 'Jenis_distribusi'] = 'Skew'
                result.loc[result['Nama_kolom'] == col, 'Skew'] = data[col].skew()
                
            # Penentuan berdasarkan jenis normal distribution
            else:
                
                # Penentuan lower boundaries dan upper boundaries
                avg = data[col].mean()
                std = data[col].std()
                lower = avg - 3*std
                upper = avg + 3*std
                
                # Pemisahan data outlier
                outlier = data[col][(data[col] < lower) | (data[col] > upper)]
                
                # Perhitungan persentasse outlier
                total_count = len(data[col])
                percentage = (len(outlier) / total_count) * 100
                
                # input data ke dataframe utama
                result.loc[result['Nama_kolom'] == col, 'Persentase_outlier'] = percentage
                result.loc[result['Nama_kolom'] == col, 'Jumlah_outlier'] = len(outlier)
                result.loc[result['Nama_kolom'] == col, 'Jenis_distribusi'] = 'Normal'
                result.loc[result['Nama_kolom'] == col, 'Skew'] = data[col].skew()
        
          # input data ke dataframe utama ketika isi kolom bertipe string / object
        except TypeError:
            result.loc[result['Nama_kolom'] == col, 'Persentase_outlier'] = None
            result.loc[result['Nama_kolom'] == col, 'Jenis_distribusi'] = 'Categorikal'
            result.loc[result['Nama_kolom'] == col, 'Skew'] = None    
        
    # Melakukan sortir data berdasarkan persentase outlier 
    return result.sort_values(by='Persentase_outlier', ascending=False).reset_index(drop=True)

# Function pengecekan cardinality 
def analisis_cardinality(data):
    # Pembuatan dataframe kosong dengan informasi nama kolom, jumlah unique value, informasi masing-masing unique value dan list dari unique value
    result = pd.DataFrame()
    result['Nama_kolom'] = data.columns
    result['Num_Unique_Value'] = None
    result['Unique_Value : Count'] = None
    result['Unique_Value_list'] = None
    # Pengulangan pada setiap kolom pada dataframe
    for col in data.columns:
        
        # Pengisian kolom num unique value dengan jumlah unique value
        result.loc[result['Nama_kolom'] == col, 'Num_Unique_Value'] = data[col].nunique()
        
        # Pengisian kolom unique value list dengan seluruh data unique value
        result.loc[result['Nama_kolom'] == col, 'Unique_Value_list'] = str(data[col].unique())
        sub_result = []
        
        # Pengulangan untuk setiap unique value pada kolom
        for unique_value in data[col].unique():
            
            # input data pada perhitungan setiap unique value ke dalam dictionary
            dictionary = {str(unique_value) : int(data.loc[data[col] == unique_value][col].count()) }
            sub_result.append( dictionary ) 
            
        # input dictionary ke dalam kolom dataframe
        result.loc[result['Nama_kolom'] == col, 'Unique_Value : Count'] = str(sub_result)
        
    # Mengurutkan informasi berdasarkan jumlah unique value
    return result.sort_values(by='Num_Unique_Value', ascending=False).reset_index(drop=True)
        

# Function pembuat histogram dan boxplot
def diagnostic_plots(df, variable):
    # Define figure size
    plt.figure(figsize=(16, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')

    # Menampilkan keseluruhan plot
    plt.show()
    
# Pembuatan fungsi pembuatan dataframe korelasi
def correlation_df(method, dataframe_input, dataframe_target, kolom=None):
    dataframe_target = pd.DataFrame(dataframe_target)
        
    # Pemisahan penggunaan metode pearson
    if method == 'Pearson':
        
        # Pembuatan dataframe kosong
        result = pd.DataFrame()
        result['Nama Kolom'] = dataframe_input[kolom].columns
        result['Corr value'] = None
        result['P-Value'] = None
        result['Correlation Status'] =  'Not Correlated'
        
        # Pengulangan setiap kolom pada dataframe input
        for col in dataframe_input[kolom].columns:
            
            # Perhitungan statistik dari dataframe input dan target menggunakan methode pearson
            corr_value, pval_p = stats.pearsonr(dataframe_input[col], dataframe_target)
            
            # Input data dari hasil perhitungan statisik ke dalam dataframe
            result.loc[result['Nama Kolom'] == col, 'Corr value'] = corr_value
            result.loc[result['Nama Kolom'] == col, 'P-Value'] = pval_p
            result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'

    # Pemisahan penggunaan metode Spearman
    elif method == 'Spearman':
        
        # Pembuatan dataframe kosong
        result = pd.DataFrame()
        result['Nama Kolom'] = dataframe_input[kolom].columns
        result['Corr value'] = None
        result['P-Value'] = None
        result['Correlation Status'] =  'Not Correlated'
        
        # Pengulangan setiap kolom pada dataframe input
        for col in dataframe_input[kolom].columns:
            
            # Perhitungan statistik dari dataframe input dan target menggunakan methode Spearman
            corr_value, pval_p = stats.spearmanr(dataframe_input[col], dataframe_target)
            
            # Input data dari hasil perhitungan statisik ke dalam dataframe
            result.loc[result['Nama Kolom'] == col, 'Corr value'] = corr_value
            result.loc[result['Nama Kolom'] == col, 'P-Value'] = pval_p
            result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'

    # Pemisahan penggunaan metode kendall
    elif method == 'Kendall':
        
        # Pembuatan dataframe kosong
        result = pd.DataFrame()
        result['Nama Kolom'] = dataframe_input[kolom].columns
        result['Corr value'] = None
        result['P-Value'] = None
        result['Correlation Status'] =  'Not Correlated'
        
        # Pengulangan setiap kolom pada dataframe input
        for col in dataframe_input[kolom].columns:
            
            # Perhitungan statistik dari dataframe input dan target menggunakan methode kendall
            corr_value, pval_p = stats.kendalltau(dataframe_input[col], dataframe_target)
            
            # Input data dari hasil perhitungan statisik ke dalam dataframe
            result.loc[result['Nama Kolom'] == col, 'Corr value'] = corr_value
            result.loc[result['Nama Kolom'] == col, 'P-Value'] = pval_p
            result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'
            
    # Pemisahan penggunaan metode kendall
    elif method == 'Chisquare':
        
        # Pembuatan dataframe kosong
        result = pd.DataFrame()
        result['Nama Kolom'] = dataframe_input[kolom].columns
        result['Corr value'] = None
        result['P-Value'] = None
        result['Correlation Status'] = 'Not Correlated'
        
        # Pengulangan setiap kolom pada dataframe input
        for col in kolom:
            
            col_data = dataframe_input[col].squeeze()
            target_data = dataframe_target.values.ravel()
            
            # Pembuatan cross table
            contingency_table = pd.crosstab(col_data, target_data)
            
            # Perhitungan statistik dari dataframe input dan target menggunakan methode kendall
            res = stats.chi2_contingency(contingency_table)
            
            # Input data dari hasil perhitungan statisik ke dalam dataframe
            result.loc[result['Nama Kolom'] == col, 'Corr value'] = res.statistic
            result.loc[result['Nama Kolom'] == col, 'P-Value'] = res.pvalue
            result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'
            
    # Pemisahan penggunaan metode Anova
    elif method == 'Anova/Kruskal':
        
            
        # Pembuatan dataframe result
        result = pd.DataFrame()
        result['Nama Kolom'] = dataframe_input[kolom].columns
        result['Corr value'] = None
        result['P-Value'] = None
        result['Correlation Status'] = 'Not Correlated'
        
        # Penggqabungan dataframe input dan target
        Train_Concat = pd.concat([dataframe_input, dataframe_target], axis = 1)
        target_col = dataframe_target.columns[0]
        
        # Pengecekan dataframe target apabila terdiri dari binary classification
        if dataframe_target.iloc[:,0].nunique() == 2:
            
            # Pengulangan pada setiap kolom pada dataframe input
            for col in dataframe_input[kolom].columns:
                
                # Filtering apakah kolom tersebut merupakan kolom terdistribusi normal atau tidak
                if ((dataframe_input[col].skew() < 0.5) and (dataframe_input[col] > 0.5)).all():			
    
                    # Pembuatan list untuk data target pada setiap unique value
                    data_harga_per_unique = []
                    
                    # Pengulangan setiap unique value pada dataframe target
                    for unique_value in dataframe_target.iloc[:,0].unique():
                        data_harga_per_unique.append(Train_Concat[Train_Concat[target_col] == unique_value][col])
                    
                    # Perhitungan statistik 
                    res = stats.f_oneway(*data_harga_per_unique)
                
                # Filtering kolom terdistribusi tidak normal
                else: 
                    
                     # Pembuatan list untuk data target pada setiap unique value
                    data_harga_per_unique = []
                    
                    # Pengulangan setiap unique value pada dataframe input
                    for unique_value in dataframe_target.iloc[:,0].unique():
                        data_harga_per_unique.append(Train_Concat[Train_Concat[target_col] == unique_value][col])
                    
                    # Perhitungan statistik 
                    res = stats.kruskal(*data_harga_per_unique)

                # Input data ke dalam dataframe
                result.loc[result['Nama Kolom'] == col, 'Corr value'] = res.statistic
                result.loc[result['Nama Kolom'] == col, 'P-Value'] = res.pvalue
                result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'
        
        # Kondisi di mana data target merupakan data numerikal
        else:
            
            # Pengulangan setiap kolom pada dataframe input
            for col in dataframe_input[kolom].columns:
                
                # Filtering apakah kolom tersebut merupakan kolom terdistribusi normal atau tidak
                if ((dataframe_target[0].skew() < 0.5) and (dataframe_target[0] > 0.5)).all():			
    
                    # Pembuatan list untuk data target pada setiap unique value
                    data_harga_per_unique = []
                    
                    # Pengulangan setiap unique value pada dataframe concat
                    for unique_value in Train_Concat[col].unique():
                        data_harga_per_unique.append(Train_Concat[Train_Concat[col] == unique_value][target_col])
                    
                    # Perhitungan statistik
                    res = stats.f_oneway(*data_harga_per_unique)
                
                else: 
                    
                     # Pembuatan list untuk data target pada setiap unique value
                    data_harga_per_unique = []
                    
                    # Pengulangan setiap unique value pada dataframe concat
                    for unique_value in Train_Concat[col].unique():
                        data_harga_per_unique.append(Train_Concat[Train_Concat[col] == unique_value][target_col])
                    
                    # Perhitungan statistik
                    res = stats.kruskal(*data_harga_per_unique)

                # Input data ke dalam dataframe
                result.loc[result['Nama Kolom'] == col, 'Corr value'] = res.statistic
                result.loc[result['Nama Kolom'] == col, 'P-Value'] = res.pvalue
                result.loc[result['P-Value'] < 0.05 , 'Correlation Status'] = 'Correlated'
                
    return result.sort_values(by='P-Value').reset_index(drop=True)    

# Function referensi evaluasi numerik
def model_evaluation(model, X_train, X_test, y_train, y_test):
    
    # Pembuatan dataframe untuk diisi evaluasi data modeling
    df_evaluasi = pd.DataFrame(columns=['Name', 'MAE - Train', 'MAE - Test', 'MSE - Train', 'MSE - Test', 'RMAE - Train', 'RMAE - Test', 'R2 - Train', 'R2 - Test'])
    index_max = [len(model)]
    
    # Pengulangan setiap model pada list parameter model  
    for md in model:
        
        # Proses prediksi menggunakan model
        y_pred_train = md.predict(X_train)
        y_pred_test = md.predict(X_test)
        
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        
        rmse_train = root_mean_squared_error(y_train, y_pred_train)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        
        r2score_train = r2_score(y_train, y_pred_train)
        r2score_test = r2_score(y_test, y_pred_test)
        
        # Pembuatan data baru dalam bentuk dataframe
        data = {'Name': model, 'MAE - Train': mae_train, 'MAE - Test': mae_test, 'MSE - Train': mse_train, 'MSE - Test': mse_test, 'RMAE - Train': rmse_train, 'RMAE - Test': rmse_test, 'R2 - Train': r2score_train, 'R2 - Test':r2score_test}
        df_data = pd.DataFrame(data, index=index_max)
        
        # Penggabungan data baru ke dalam dataframe
        df_evaluasi = pd.concat([df_evaluasi, df_data], ignore_index=True)
    
    return df_evaluasi

# Function referensi evaluasi kategorikal
def model_evaluation_class(model, X_train, X_test, y_train, y_test):
    
    # Pembuatan list berisi nama kolom
    kolom_list = ['Name',
                  'Train Accuracy',
                  'Test Accuracy',
                  'Train Precision', 
                  'Test Precision', 
                  'Train Recall', 
                  'Test Recall', 
                  'Train F1', 
                  'Test F1' ]
    
    # Pembuatan dataframe berdasarkan list kolom
    df_evaluasi = pd.DataFrame(columns=kolom_list)
    index_max = [len(model)]
    
    # Pengulangan setiap model pada model list 
    for md in model:
        
        # Melakukan prediksi berdasarkan model
        y_pred_train = md.predict(X_train)
        y_pred_test = md.predict(X_test)
        
        train_we_accuracy = accuracy_score(y_train, y_pred_train)
        test_we_accuracy = accuracy_score(y_test, y_pred_test)
        train_we_precision = precision_score(y_train, y_pred_train, average='weighted')
        test_we_precision = precision_score(y_test, y_pred_test, average='weighted')
        train_we_recall = recall_score(y_train, y_pred_train, average='weighted')
        test_we_recall = recall_score(y_test, y_pred_test, average='weighted')
        train_we_f1 = f1_score(y_train, y_pred_train, average='weighted')
        test_we_f1 = f1_score(y_test, y_pred_test, average='weighted')
        
        try :
            data = {'Name': f'{type(md.estimator).__name__} {md.best_params_}', 
                'Train Accuracy': train_we_accuracy,
                'Test Accuracy': test_we_accuracy,
                'Train Precision': train_we_precision, 
                'Test Precision': test_we_precision,
                'Train Recall': train_we_recall, 
                'Test Recall': test_we_recall,
                'Train F1': train_we_f1, 
                'Test F1': test_we_f1,
                }
        
        except AttributeError :
            data = {'Name': md, 
                'Train Accuracy': train_we_accuracy,
                'Test Accuracy': test_we_accuracy,
                'Train Precision': train_we_precision, 
                'Test Precision': test_we_precision,
                'Train Recall': train_we_recall, 
                'Test Recall': test_we_recall,
                'Train F1': train_we_f1, 
                'Test F1': test_we_f1,
                }
        
        df_data = pd.DataFrame(data, index=index_max)
        
        df_evaluasi = pd.concat([df_evaluasi, df_data], ignore_index=True)
        
    return df_evaluasi

# Pembuatan fungsi untuk handling outlier otomatis
def auto_outlier_handling(method, dataframe_train, dataframe_test, dataframe_persentase_outlier):
    
    # Pembuatan variabel data outlier berdasarkan dataframe kolom yang memiliki outlier 
    data_outlier = dataframe_persentase_outlier[dataframe_persentase_outlier['Persentase_outlier'] > 0]
    
    # Filtering ketika metode yang dipilih adalah capping
    if method == 'Capping':
        
        # Pengulangan kolom pada setiap kolom yang tergolong memiliki outlier
        for col in data_outlier['Nama_kolom']:
            
            # Pengecekan distribusi dari masing-masing kolom
            distribution = data_outlier.loc[data_outlier['Nama_kolom'] == col, 'Jenis_distribusi'].values[0]
            
            # Filtering distribusi skew ekstrem
            if distribution == 'Extreme Skew':
                
                try : 
                    capping_extreme_skew = Winsorizer(
                        capping_method='iqr', 
                        tail='both', 
                        fold=3, 
                        missing_values='ignore',
                        variables=[col])
                    
                    # Proses fit dan transform
                    dataframe_train = capping_extreme_skew.fit_transform(dataframe_train)
                    dataframe_test = capping_extreme_skew.transform(dataframe_test)

                except ValueError:
                    continue	
     
            # Filtering distribusi skew
            elif distribution == 'Skew':
                
                # Proses winsorizer
                capping_skew = Winsorizer(
                    capping_method='iqr', 
                    tail='both', 
                    fold=1.5, 
                    missing_values='ignore',
                    variables= [col])
                
                # Proses fit dan transform 
                dataframe_train = capping_skew.fit_transform(dataframe_train)
                dataframe_test = capping_skew.transform(dataframe_test)

            # Filtering distribusi normal
            elif distribution == 'Normal':
                
                # Proses winsorizer
                capping_normal = Winsorizer(
                    capping_method='gaussian', 
                    tail='both', 
                    fold=3, 
                    missing_values='ignore',
                    variables= [col])
                
                # Proses fit dan transform 
                dataframe_train = capping_normal.fit_transform(dataframe_train)
                dataframe_test = capping_skew.transform(dataframe_test)
                
       # Filtering ketika metode yang dipilih adalah drop
    elif method == 'Drop':
        
        # Pengulangan kolom pada setiap kolom yang tergolong memiliki outlier
        for col in data_outlier['Nama_kolom']:
            
            # Pengecekan distribusi dari masing-masing kolom
            distribution = data_outlier.loc[data_outlier['Nama_kolom'] == col, 'Jenis_distribusi'].values[0]
            
            # Filtering distribusi skew ekstrem
            if distribution == 'Extreme Skew':
                Q1 = dataframe_train[col].quantile(0.25)
                Q3 = dataframe_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

            # Filtering distribusi skew
            elif distribution == 'Skew':
                Q1 = dataframe_train[col].quantile(0.25)
                Q3 = dataframe_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

            # Filtering distribusi normal
            elif distribution == 'Normal':
                mean = dataframe_train[col].mean()
                std = dataframe_train[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            
            # Filtering dataframe sesuai boundaries
            dataframe_train = dataframe_train[(dataframe_train[col] >= lower_bound) & (dataframe_train[col] <= upper_bound)]

    return dataframe_train, dataframe_test

# Pembuatan fungsi auto gridsearch untuk proses multiple modelling otomatis
def Auto_GridsearchCV(X_train, y_train, model, fold, score=None):
    
    # Melakukan proses pencarian kombinasi hyperparameter terbaik
    gridsearch = GridSearchCV(
        estimator = model['model_name'],
        param_grid = model['Hyperparameter'],
        cv=fold,
        scoring=score
    )
    
    # Pelatihan model terbaik dengan data train
    gridsearch.fit(X_train, y_train)
    
    return gridsearch, gridsearch.best_params_
