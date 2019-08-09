#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# Standard library imports
import io
#from __future__ import unicode_literals


#Django default imports
from django.shortcuts import render ,redirect
from django.views.generic import View
from django.http import HttpRequest , HttpResponse ,request
from django.template.loader import get_template
from django.template.response import TemplateResponse
from django.template.loader import render_to_string

# Third party imports
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns 
import base64
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras import backend as K
from keras import losses
from keras.models import Sequential
from keras.layers import Dense
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.iolib.smpickle import load_pickle
from geopandas import GeoDataFrame
from shapely.geometry import Point
import matplotlib.pyplot as plt

def main():

    def run_models():
        #-------------------------------------Creating and storing MLP model-----------------------------------------------------
        # Importing the dataset and separating dependent/independent variables

        dataset = pd.read_csv("assets/predicts.csv")
        
        
        

        print(dataset.dtypes)
        
            


        dataset['Main purpose of visit'].value_counts()
        dataset['Accessibility status'].value_counts()
        dataset['Accomodation status'].value_counts()
        dataset['health services status'].value_counts()

        cleanup_nums = {"Accessibility status":{"Poor": 1, "Fair": 2,"Good":3,"Better":4},
                        "Accomodation status": {"Poor": 1, "Fair": 2,"Good":3,"Better":4},
                        "health services status":{"Poor": 1, "Fair": 2,"Good":3,"Better":4},
                        }
        dataset.replace(cleanup_nums, inplace=True)
        dataset.head(5)



        print(dataset.head(5))
        X = dataset.iloc[:,1:8].values
        print(X[:,3])

        y = dataset.iloc[:,10].values
        print(y)
        # Encoding categorical data
        
        labelencoder_X_3 = LabelEncoder()
        X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

        list(labelencoder_X_3.inverse_transform([0, 1, 2, 3]))

        X[:, 3]
        X[:,0:4]
        print(X)


        onehotencoder = OneHotEncoder(categorical_features = [3] )
        X = onehotencoder.fit_transform(X).toarray()

        X = X[:, 1:]

        print('\n'.join([''.join(['{:9}'.format(item) for item in row]) 
            for row in X]))


        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


        a=y_test
        b=y_train

        # Feature Scaling //escaping
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Part 2 - making the the ANN model

        # Importing the Keras libraries and packages
        
        
        # Initialising the ANN for regression



        #Creating regression model
        REG = Sequential()

        # Adding the input layer and the first hidden layer with dropout if required
        REG.add(Dense(units=20,input_dim=9 ,kernel_initializer="normal", activation = 'relu'))
        #REG.add(Dropout(p=0.1))
        # Adding the second hidden layer
        REG.add(Dense(units =20,kernel_initializer="normal", activation = 'relu'))
        #REG.add(Dropout(p=0.1))
        # Adding the output layer
        REG.add(Dense(units = 1, kernel_initializer="normal"))

        # Compiling the ANN
        #def root_mean_squared_error(y_true, y_pred):
        #        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
            
        REG.compile(optimizer = 'adam', loss= 'mean_squared_error')

        # Fitting the ANN to the Training set
        REG.fit(X_train, y_train, batch_size = 10, epochs = 200)

        # Part 3 - Making the predictions and evaluating the model
        X_test



        # Predicting the Test set results
        y_pred = REG.predict(X_test)

        REG.save('assets/REG_MLP_model.h5')
        K.clear_session()
        #---------------------------------------------------------------------------------------------------------------------


        #---------------------------------------Creating and storing SARIMA model----------------------------------------------
         #data collecting...converting dataset to html....
        df = pd.read_csv('assets/Touristarrival_monthly.csv')
        df1=df.iloc[:5]
        html_table_template = df1.to_html(index=False)
        html_table=df.to_html(index=False)
        #data observation and log transformation
        df.index=pd.to_datetime(df['Month'])
        df['#Tourists'].plot()
        mpl.pyplot.ylabel("No.of Toursits Arrivals ")
        mpl.pyplot.xlabel("Year")
        
         #storing plots
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_input.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()  
      

        series=df['#Tourists']
        logtransformed=np.log(series)
        logtransformed.plot()
        mpl.pyplot.ylabel("log Scale(No.of Toursits Arrivals) ")
        mpl.pyplot.xlabel("Year")
        
         #storing plots 
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_input_logscaled.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()
        
        #Train test split
        percent_training=0.80
        split_point=round(len(series)*percent_training)
        print(split_point)
        training , testing = series[0:split_point] , series[split_point:]
        training=np.log(training)


        
        #differencing to achieve stationarity
        training_diff=training.diff(periods=1).values[1:]

        #plot of residual log differenced series
        mpl.pyplot.plot(training_diff)
        mpl.pyplot.title("Tourist arrivals data log-differenced")
        mpl.pyplot.xlabel("Years")
        mpl.pyplot.ylabel("Toursits arrivals")
        mpl.pyplot.clf()


        #ACF and PACF plots 1(with log differenced training data)
        lag_acf=acf(training_diff,nlags=40)
        lag_pacf=pacf(training_diff,nlags=40,method='ols')

        #plot ACF
        mpl.pyplot.figure(figsize=(15,5))
        mpl.pyplot.subplot(121)
        mpl.pyplot.stem(lag_acf)
        mpl.pyplot.axhline(y=0,linestyle='-',color='black')
        mpl.pyplot.axhline(y=-1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        mpl.pyplot.axhline(y=1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        mpl.pyplot.xlabel('lag')
        mpl.pyplot.ylabel("ACF")
        #storing plots in bytes
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_afc.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()
        

        #plot PACF
        mpl.pyplot.figure(figsize=(15,5))
        mpl.pyplot.subplot(122)
        mpl.pyplot.stem(lag_pacf)
        mpl.pyplot.axhline(y=0,linestyle='-',color='black')
        mpl.pyplot.axhline(y=-1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        mpl.pyplot.axhline(y=1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        mpl.pyplot.xlabel('lag')
        mpl.pyplot.ylabel("PACF")
         #storing plots in bytes
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_pafc.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #SARIMA Model specification
        model=sm.tsa.statespace.SARIMAX(training,order=(2,0,3),seasonal_order=(2,1,0,12),trend='c',enforce_invertibility=False,enforce_stationarity=False)

        # fit model
        model_fit = model.fit()

        model_fit.save("assets/REG_SARIMA_model.pickle")

        print(model_fit.summary())

        #plot residual errors
        # residuals = pd.DataFrame(model_fit.resid)
        # fig, ax = mpl.pyplot.subplots(1,2)
        # residuals.plot(title="Residuals", ax=ax[0])
        # residuals.plot(kind='kde', title='Density', ax=ax[1])
        # mpl.pyplot.show()
        # print(residuals.describe())

        # Model evaluation and forecast
        model_fitted=load_pickle("assets/REG_SARIMA_model.pickle")
        forecast=model_fitted.forecast(len(df)-250)
        print(forecast)
        forecast=np.exp(forecast)
        print(forecast)
        #plot forecast results and display RMSE
        mpl.pyplot.figure(figsize=(10,5))
        mpl.pyplot.plot(forecast,'r')
        mpl.pyplot.plot(series,'b')
        mpl.pyplot.legend(['Predicted test values','Actual data values'])

        mpl.pyplot.title('RMSE:%.2f'% np.sqrt(sum((forecast-testing)**2)/len(testing)))
        mpl.pyplot.ylabel("No.of Toursits Arrivals Monthly")
        mpl.pyplot.xlabel("Year")
        mpl.pyplot.autoscale(enable='True',axis='x',tight=True)
        mpl.pyplot.axvline(x=series.index[split_point],color='black');
         #storing plots 
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_result.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        forecaste=model_fitted.forecast(len(df)-214)
        forecast_next=forecaste[62:]
        forecast_next=np.exp(forecast_next)
        print(forecast_next)
        mpl.pyplot.figure(figsize=(10,5))
        mpl.pyplot.plot(forecast_next,'r')
        mpl.pyplot.plot(series,'b')
        mpl.pyplot.legend(['Predicted next steps values'])
        mpl.pyplot.title('Monthly tourist arrivals predictions')
        mpl.pyplot.ylabel("No.of Toursits Arrivals ")
        mpl.pyplot.xlabel("Year")
        mpl.pyplot.autoscale(enable='True',axis='x',tight=True)

        #storing plots in bytes
        mpl.pyplot.savefig('PredictionEngine/static/img/sarima_forecast.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()
        


 




        





        

        

            


        

        


    def run_visualizations():


        #earthquake2072_effect_on_tourism
        data=pd.read_csv("assets/earthquake2072_effect_on_tourism.csv",header=0)
        data.pivot(index='Subsector', columns='Disaster effect', values='value( NPR Million)').plot(kind='bar')
        mpl.pyplot.savefig('AnalysisEngine/static/img/id1.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #tourist arrivals by age group
        data=pd.read_csv("assets/tourist arrivals by age group.csv",header=1,index_col=0)      
        data.iloc[1].plot.bar()
        mpl.pyplot.savefig('AnalysisEngine/static/img/id2.png',dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #Economic_indicators_of_hotels.csv
        data=pd.read_csv("assets/Economic_indicators_of_hotels.csv",header=0)       
        df1=data[['Economic Indicators','Fiscal Year','val']]
        heatmap1_data = pd.pivot_table(df1, values='val',index=['Economic Indicators'],columns='Fiscal Year')
        sns_plot=sns.heatmap(heatmap1_data, cmap="YlGnBu")
        mpl.pyplot.savefig('AnalysisEngine/static/img/id3.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

        #tourist_arrivals_purpose_newlook
        data=pd.read_csv("assets/tourist_arrivals_purpose_newlook.csv",header=0 )
        data.pivot(index='year', columns='purposes', values='Arrivals').plot(kind='bar')
        mpl.pyplot.savefig('AnalysisEngine/static/img/id4.png')
        mpl.pyplot.clf()

        mpl.pyplot.axhline(0, color='k')
        #Data to plot for piechart1(tourist arrivals purpose of visits)
        labels = 'holiday pleasure', 'trekking and mountaineering', 'business', 'pilgrimage','official','conference','others'
        sizes = [489451,66490,24322,82830,21310,12801,55797] # of latest year 2017 in tourist arrivals by purpose
        colors = ['gold', 'green', 'lightcoral', 'lightskyblue','blue','red','purple']
        patches, texts = mpl.pyplot.pie(sizes, colors=colors, shadow=True, startangle=90)
        mpl.pyplot.legend(patches, labels, loc="best")
        mpl.pyplot.axis('equal')
        mpl.pyplot.tight_layout()
    
        mpl.pyplot.savefig('AnalysisEngine/static/img/id44.png')
        mpl.pyplot.clf()

        #No_tourist_industries_guides
        data=pd.read_csv("assets/No_tourist_industries_guides.csv")
        df1=data[['Industries/guides','year','numbers']]
        heatmap1_data = pd.pivot_table(df1, values='numbers',index=['Industries/guides'],columns='year')
        sns.heatmap(heatmap1_data, cmap="YlGnBu")
        mpl.pyplot.savefig('AnalysisEngine/static/img/id5.png', dpi=600,bbox_inches='tight')
        mpl.pyplot.clf()

         #No. of tourists destinations distribution map


        data=pd.read_csv("assets/nepal-district.csv")


        df2=data[['District','Zones','Development Regions','Tourist places']]
        
        #read shape file

        fp="assets/NepalMaps-master/baselayers/NPL_adm/NPL_adm3.shp"

        map_df = gpd.read_file(fp)
        

        


        # fig, ax = map_df.plot(figsize = (15, 12), color = "whitesmoke", edgecolor = "lightgrey", linewidth = 0.5)
        # texts = []









        #joining file

        merged = map_df.set_index('NAME_3').join(df2.set_index('District'))

        variable= 'Tourist places' #plotting data 

        vmin, vmax = 1, 15  #data min - max values


        map_df["center"] = map_df["geometry"].centroid
        za_points = map_df.copy()
        za_points.set_geometry("center", inplace = True)

        fig, ax = mpl.pyplot.subplots(1, figsize=(15, 7)) #number of figure and size axis
        for x, y, label in zip(za_points.geometry.x, za_points.geometry.y, za_points["NAME_3"]):
            texts.append(plt.text(x, y, label, fontsize = 8))

        
        #plotting map

        merged.plot(column = variable, cmap='Blues', linewidth = 0.8,ax=ax, edgecolor = '0.8')

        ax.axis('off')
        ax.set_title('Tourist Attraction Places in Nepal', fontdict={'fontsize':'25', 'fontweight':'3'})

        # Create colorbar as a legend

        sml = mpl.pyplot.cm.ScalarMappable(cmap='Blues', norm=mpl.pyplot.Normalize(vmin=vmin, vmax=vmax))

        # empty array for the data range

        sml._A = []

        # add the colorbar to the figure

        cbar = fig.colorbar(sml)

        #storing plots in bytes
        
        mpl.pyplot.savefig('AnalysisEngine/static/img/id6.png', dpi=700,bbox_inches='tight')
    
        mpl.pyplot.clf()



    run_visualizations()
    run_models()
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TourismAnalytics.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()