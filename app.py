import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
pv = 0
with st.sidebar :
    selected = option_menu ('Prediksi dan Klastering',
    ['Prediksi Masa Tunggu',
    'Klastering Provinsi'],
    default_index=0) 

##prediksi masa tunggu
if(selected=='Prediksi Masa Tunggu'):
    st.title("Aplikasi Prediksi Masa Tunggu Keberangkatan Jemaah Haji Indonesia")
    th = st.slider(
        min_value=2023, max_value=2120, value=2023, label="Masukan Tahun : ",
    )


    option = st.selectbox(
    'Pilih Provinsi',
    ('ACEH','BALI','BANGKA BELITUNG','BENGKULU','D.I. YOGYAKARTA','GORONTALO','JAMBI','KALIMANTAN BARAT','KALIMANTAN SELATAN','KALIMANTAN TENGAH','KALIMANTAN TIMUR','KEPULAUAN RIAU','MALUKU','MALUKU UTARA','NUSA TENGGARA BARAT','NUSA TENGGARA TIMUR','PAPUA','PAPUA BARAT','RIAU','SULAWESI BARAT','SULAWESI TENGAH','SULAWESI TENGGARA','SULAWESI UTARA','SUMATERA BARAT','JAWA BARAT','JAWA TENGAH','JAWA TIMUR','BANTEN','DKI JAKARTA','LAMPUNG','SULAWESI SELATAN','SUMATERA SELATAN','SUMATERA UTARA', 'SEMUA PROVINSI')
)
    
    if (option == "ACEH"):
        pv= 1
    elif option == "BALI":
        pv= 2
    elif option == "BANTEN":
        pv= 3
    elif option == "BANGKA BELITUNG":
        pv= 4
    elif option == "BENGKULU":
        pv=5
    elif option == "D.I. YOGYAKARTA":
        pv= 6
    elif option == "DKI JAKARTA":
        pv= 7
    elif option == "GORONTALO":
        pv= 8
    elif option == "JAMBI":
        pv= 9
    elif option == "JAWA BARAT":
        pv= 10
    elif option == "JAWA TENGAH":
        pv= 11
    elif option == "JAWA TIMUR":
        pv= 12
    elif option == "KALIMANTAN BARAT":
        pv= 13
    elif option == "KALIMANTAN SELATAN":
        pv= 14
    elif option == "KALIMANTAN TENGAH":
        pv= 15
    elif option == "KALIMANTAN TIMUR":
        pv= 16
    elif option == "KEPULAUAN RIAU":
        pv= 17
    elif option == "LAMPUNG":
        pv= 18
    elif option == "MALUKU":
        pv=19
    elif option == "MALUKU UTARA":
        pv= 20
    elif option == "NUSA TENGGARA BARAT":
        pv= 21
    elif option == "NUSA TENGGARA TIMUR":
        pv= 22
    elif option == "PAPUA":
        pv= 23
    elif option == "PAPUA BARAT":
        pv= 24
    elif option == "RIAU":
        pv= 25
    elif option == "SULAWESI BARAT":
        pv= 26
    elif option == "SULAWESI SELATAN":
        pv= 27
    elif option == "SULAWESI TENGAH":
        pv= 28
    elif option == "SULAWESI TENGGARA":
        pv= 29
    elif option == "SULAWESI UTARA":
        pv= 30
    elif option == "SUMATERA BARAT":
        pv= 31
    elif option == "SUMATERA SELATAN":
        pv= 32
    elif option == "SUMATERA UTARA":
        pv= 33
    elif option == "SEMUA PROVINSI" : 
        pv = 99


    hasil=st.button("Hasil Prediksi")
    
    model_file = open('Regresi_Linear.pkl','rb')
    model = pickle.load(model_file, encoding='bytes')
    
    if (hasil and pv != 99):
        x=np.array([[th,pv]])
        predictions_regresi_linear= model.predict(x)
        st.write(f"Hasil Prediksi Regresi Linear = {predictions_regresi_linear}")
    elif (hasil and pv == 99): 
        x=np.array([[th,1]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Aceh= {predictions_regresi_linear}")
        
        x=np.array([[th,2]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Bali= {predictions_regresi_linear}")
        
        x=np.array([[th,3]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Banten= {predictions_regresi_linear}")
        
        x=np.array([[th,4]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Bangka Belitung= {predictions_regresi_linear}")
        
        x=np.array([[th,5]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Bengkulu= {predictions_regresi_linear}")
        
        x=np.array([[th,6]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Yogyakarta= {predictions_regresi_linear}")
        
        x=np.array([[th,7]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi DKI Jakarta= {predictions_regresi_linear}")
        
        x=np.array([[th,8]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Gorontalo= {predictions_regresi_linear}")
        
        x=np.array([[th,9]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Jambi= {predictions_regresi_linear}")
        
        x=np.array([[th,10]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Jawa Barat= {predictions_regresi_linear}")
        
        x=np.array([[th,11]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Jawa Tengah= {predictions_regresi_linear}")
        
        x=np.array([[th,12]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Jawa Timur= {predictions_regresi_linear}")
        
        x=np.array([[th,13]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Kalimantan Barat= {predictions_regresi_linear}")
        
        x=np.array([[th,14]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Kalimantan Selatan= {predictions_regresi_linear}")
        
        x=np.array([[th,15]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Kalimantan Tengah= {predictions_regresi_linear}")
        
        x=np.array([[th,16]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Kalimantan Timur= {predictions_regresi_linear}")
        
        x=np.array([[th,17]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Kepulauan Riau= {predictions_regresi_linear}")
        
        x=np.array([[th,18]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Lampung= {predictions_regresi_linear}")
        
        x=np.array([[th,19]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Maluku= {predictions_regresi_linear}")
        
        x=np.array([[th,20]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Maluku Utara= {predictions_regresi_linear}")
        
        x=np.array([[th,21]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Nusa Tenggara Barat= {predictions_regresi_linear}")
        x=np.array([[th,22]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Nusa Tenggara Timur= {predictions_regresi_linear}")
        
        x=np.array([[th,23]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Papua= {predictions_regresi_linear}")
        
        x=np.array([[th,24]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Papua Barat= {predictions_regresi_linear}")
        
        x=np.array([[th,25]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Riau= {predictions_regresi_linear}")
        
        x=np.array([[th,26]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sulawesi Barat= {predictions_regresi_linear}")
        
        x=np.array([[th,27]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sulawesi Selatan= {predictions_regresi_linear}")
        
        x=np.array([[th,28]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sulawesi Tengah= {predictions_regresi_linear}")
        
        x=np.array([[th,29]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sulawesi Tenggara= {predictions_regresi_linear}")
        
        x=np.array([[th,30]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sulawesi Utara= {predictions_regresi_linear}")
        
        x=np.array([[th,31]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sumatera Barat= {predictions_regresi_linear}")
        
        x=np.array([[th,32]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sumatera Selatan= {predictions_regresi_linear}")
        
        x=np.array([[th,33]])
        predictions_regresi_linear= model.predict(x)
        st.success(f"Hasil Prediksi Regresi Linear Provinsi Sumatera Utara= {predictions_regresi_linear}")
    

if(selected=="Klastering Provinsi") :
    st.title(""" PROTOTYPE CLUSTERING DENGAN ALGORITMA KMEANS \n""")
    def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
        
        Parameters
        ----------
        x, y : array-like, shape (n, )
         Input data.
         
        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.
        
        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.
        
        **kwargs
            Forwarded to `~matplotlib.patches.Ellipse`
        
        Returns
        -------
        matplotlib.patches.Ellipse
        """
        
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
            
        cov = np.cov(x, y)
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                            facecolor=facecolor, **kwargs)
                            
        # Calculating the standard deviation of x from
        # # the squareroot of the variance and multiplying
        # # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        
        # calculating the standard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
            
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    @st.cache
    def data():
        X = np.array([3951368, 2094, 48, 68.07,3509090, 2098, 45, 70.39, 9814334, 2095, 46, 68.15, 1520430, 2095, 51, 67.54, 3175789, 2103, 44, 73.27, 8266356, 2100, 46, 71.21, 905361, 2092, 47, 66.27, 2740177, 2097, 48, 69.33, 38094970, 2101, 47, 71.57, 27253914, 2102, 46, 72.61, 31885703, 2098, 46, 69.51, 3850848, 2097, 47, 68.87, 3204519, 2094, 42, 66.97, 2053879, 2096, 43, 67.89, 3354572, 2103, 44, 72.79, 1121078, 2096, 46, 68.86, 1767979, 2097, 47, 68.24, 6481189, 2098, 49, 68.86, 1308543, 2091, 51, 64.23, 922436, 2094, 51, 66.55, 3882460, 2092, 50, 64.81, 3955577, 2093, 48, 65.28, 2494142, 2091, 44, 64.15, 730251, 2092, 45, 64.33, 5066687, 2100, 46, 69.82, 1008651, 2090, 44, 63.39, 6817600, 2099, 43, 68.77, 2304910, 2096, 46, 66.91, 1970728, 2100, 47, 69.35, 1950758, 2101, 48, 69.89, 4077628, 2098, 49, 67.7, 6396607, 2098, 49, 68.11, 10869765, 2097, 48, 67.35]).reshape(-1,4)
        return X
    X = data()
    
    klaster_slider = st.slider(
        min_value=3, max_value=5, value=3, label="Jumlah Klaster : "
    )
    
    kmeans = KMeans(n_clusters=klaster_slider, random_state=2022).fit(X)
    labels = kmeans.labels_
    
    warna = ["red", "seagreen", "orange", "blue", "yellow", "purple"]
    
    jumlah_label = len(set(labels))
    
    
    individu = False
    # st.selectbox("Subplot Individu?", [False, True])
    if individu:
        fig, ax = plt.subplots(n_cols=jumlah_label)
    else:
        fig, ax = plt.subplots()
        
    for i, yi in enumerate(set(labels)):
        if not individu:
            a = ax
        else:
            a = ax[i]
            
        xi = X[labels == yi]
        x_pts = xi[:, 0]
        y_pts = xi[:, 1]
        a.scatter(x_pts, y_pts, c=warna[yi])
    plt.tight_layout()
    st.write(fig)
    
    




    
    
