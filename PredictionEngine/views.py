

# Standard library imports
from __future__ import unicode_literals
import io
from math import sqrt
#Django default imports
from django.shortcuts import render , redirect
from django.http import HttpRequest , HttpResponse ,request
from django.template.loader import get_template
from django.template.response import TemplateResponse
from django.template.loader import render_to_string

# Third party imports
import matplotlib as mpl
mpl.use("Agg")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.iolib.smpickle import load_pickle
import base64
from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras import backend as K
from keras import losses
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense


# Local application imports
from .models import Predictions
from .utils import render_to_pdf



dic=[]
# Create your views here.

def predict_view(request):
    all_notifications_list=Predictions.objects.order_by('created_at')[:10]
    context = {
        'all_notifications_list':all_notifications_list
    }
    return render(request,'PredictionEngine/predict_list.html',context)


def listpicview(request,id):

    if id == 1:
       

        return render(request,'PredictionEngine/prediction_pic.html',{'id':1,})
    if id == 2:
       
        return render(request,'PredictionEngine/prediction_pic.html',{'id':2,})
    if id == 3:
        
        return render(request,'PredictionEngine/prediction_pic.html',{'id':3,})
    if id == 4:
       
        return render(request,'PredictionEngine/prediction_pic.html',{'id':4,})

    if id == 5:
        
        return render(request,'PredictionEngine/prediction_pic.html',{'id':5,})

    if id == 6:
        
        return render(request,'PredictionEngine/prediction_pic.html',{'id':6,})

def PDFF(request,id,*args, **kwargs):
    
    template = get_template('pdf_format.html')

    all_details=Predictions.objects.get(id=id)
    title=all_details.title
    response=predict_detail(request,id)
    html_table=response.context_data['html_table']
    image_base64=response.context_data['image_base64']
    
   
    context = {
    'all_details': all_details ,
    'html_table': html_table ,
    'image_base64': image_base64 ,
    
     
    } 
    html = template.render(context)
    pdf = render_to_pdf('pdf_format.html', context)
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = title+".pdf"
        content =" inline; filename=%s "%(filename)
        download = request.GET.get("download")
        if download:
                content = "attachment; filename=%s" %(filename)
        response['Content-Disposition'] = content
        return response

    return HttpResponse("Not found")

    return redirect(predict_view)


def predict_detail(request,id):
    if (id==1):
        #data collecting...converting dataset to html....
        data=pd.read_csv("assets\gross foreign exchange earning from tourism.csv",header=0,index_col=0)
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        #data plotting/visualizing........
       
        
        data.plot()

        #storing plots in bytes
        f = io.BytesIO()
        mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        f.close()
        mpl.pyplot.clf()
        # getting details of id
        all_details=Predictions.objects.get(id=id)

        # Calculate the mean value of a list of numbers
        def mean(values):
            return sum(values) / float(len(values))
        
        # Calculate the variance of a list of numbers
        def variance(values, mean):
            return sum([(x-mean)**2 for x in values])
        
        # Calculate covariance between x and y
        def covariance(x, mean_x, y, mean_y):
            covar = 0.0
            for i in range(len(x)):
                covar += (x[i] - mean_x) * (y[i] - mean_y)
            return covar
        # Calculate regression coefficients
        def coefficients(X,Y):	
            x_mean, y_mean = mean(X), mean(Y)
            b1 = covariance(X, x_mean, Y, y_mean) / variance(X, x_mean)
            b0 = y_mean - b1 * x_mean
            return [b0, b1]

        # Split a dataset into a train and test set
        def train_test_split(dataset, split):
            train_size = int(split * len(dataset))
            dtrain=dataset.iloc[0:train_size].values
            dtest=dataset.iloc[train_size:].values
            return dtrain, dtest
        
        # Calculate root mean squared error
        def rmse_metric(actual, predicted):
            sum_error = 0.0
            for i in range(len(actual)):
                prediction_error = predicted[i] - actual[i]
                sum_error += (prediction_error ** 2)
            mean_error = sum_error / float(len(actual))
            return sqrt(mean_error)
        
        # Evaluate an algorithm using a train/test split
        def evaluate_algorithm(dataset, algorithm, split, *args):
            train, test = train_test_split(dataset, split)
            test_set = list()
            for row in test:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)
            predicted = algorithm(train, test_set, *args)
            actual = [row[-1] for row in test]
            rmse = rmse_metric(actual, predicted)
            return rmse


        # Simple linear regression algorithm
        def simple_linear_regression(train, test):
            predictions = list()
            x_train=train[:, [0]]
            y_train=train[:, [1]]
            #x_test=test.iloc[:,0]
            #y_test=train.iloc[:,1]
            b0,b1 = coefficients(x_train,y_train)
            for row in test:
                yhat = b0 + b1 * row[0] 
                predictions.append(yhat)
            return predictions

        #next step prediction in full model

        def linear_reg_perdict(Dataset,test_x):
            X=dataset.iloc[:,0].values
            Y=dataset.iloc[:,1].values
            b0,b1 = coefficients(X,Y)
            y_predict= b0 + b1 * test_x
            return y_predict


        dataset = pd.read_csv("assets/gross foreign exchange earning from tourism.csv",skiprows=0)
        dataset.head(5)
        X=dataset.iloc[:,0].values
        Y=dataset.iloc[:,1].values



        x_mean, y_mean = mean(X), mean(Y)
        var_x, var_y = variance(X, x_mean), variance(Y, y_mean)
        covar = covariance(X, x_mean, Y, y_mean)
        b0, b1 = coefficients(X,Y)

        #showing regression graphically 
        x = X
        y1 = b0 + b1 * x
        y2= Y

        dataset.plot.line(x='fiscal  year start ', y='Net received foreign exchange earning(NRs in million)')
        mpl.pyplot.plot(x,y1,color='red')

        mpl.pyplot.scatter(x,y2,color='k')
        mpl.pyplot.show()
         #storing plots in bytes
        g = io.BytesIO()
        #fig.savefig(f, format="png", dpi=600,bbox_inches='tight')
        mpl.pyplot.savefig(g, format="png", dpi=800,bbox_inches='tight')
        image_base64g = base64.b64encode(g.getvalue()).decode('utf-8').replace('\n', '')
        g.close()
        mpl.pyplot.clf()
        x=2018
        nexts=linear_reg_perdict(dataset,x)







         #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'image_base64':image_base64 ,
        'image_base64g':image_base64g ,
        'next_year_value':nexts ,
       
        }
        return TemplateResponse(request,'PredictionEngine/predict_detail.html',context)


    if (id==2):
        #data collecting...converting dataset to html....
        data=pd.read_csv("assets/predicts.csv")
        df=data.iloc[:5]
        html_table_template = df.to_html(index=False)
        html_table=data.to_html(index=False)
        
        

        place=request.POST.get("place")
        purpose=request.POST.get("Major purpose of visit")
        ACCESSIBILITY=request.POST.get("ACCESSIBILITY STATUS")
        ACCOMODATION=request.POST.get("ACCOMODATION STATUS")
        MEDICAL=request.POST.get("MED STATUS")

        

        SPOTS=request.POST.get("spots")
        submitt=request.POST.get("S")

        if (ACCESSIBILITY=='p'):
            a_status=1
        elif(ACCESSIBILITY=='f'):
            a_status=2
        elif(ACCESSIBILITY=='g'):
            a_status=3
        elif(ACCESSIBILITY=='b'):
            a_status=4


        if (ACCOMODATION=='p'):
            am_status=1
        elif(ACCOMODATION=='f'):
            am_status=2
        elif(ACCOMODATION=='g'):
            am_status=3
        elif(ACCOMODATION=='b'):
            am_status=4    

        if (MEDICAL=='p'):
            m_status=1
        elif(MEDICAL=='f'):
            m_status=2
        elif(MEDICAL=='g'):
            m_status=3
        elif(MEDICAL=='b'):
            m_status=4 


        if(purpose=='Treeking'):
            z=7
        elif(purpose=='Treeking and Mountaineering'):
            z=6
        elif(purpose=='holiday and pleasure'):
            z=0
        elif(purpose=='Pilgrimage visit'):
            z=37

        
           
        




        
        new_prediction_value=0

       

        #################################################################################################################

        # -*- coding: utf-8 -*-
        """
        Created on Mon Jun  3 10:10:14 2019

        @author: sakar
        """
        if(submitt=='PREDICT THE PERCENTAGE OF TOURIST ARRIVALS '):
            K.clear_session()
            

            # Importing the dataset and separating dependent/independent variables

            dataset = pd.read_csv("assets/predicts.csv")
            
            
            activities=[int(0 if request.POST.get("C1") is None else float(request.POST.get("C1"))),int(0 if request.POST.get("C2") is None else float(request.POST.get("C2"))),int(0 if request.POST.get("C3") is None else float(request.POST.get("C3"))),int(0 if request.POST.get("C4") is None else float(request.POST.get("C4"))),int(0 if request.POST.get("C5") is None else float(request.POST.get("C5"))),int(0 if request.POST.get("C6") is None else float(request.POST.get("C6"))),int(0 if request.POST.get("C7") is None else float(request.POST.get("C7"))),int(0 if request.POST.get("C8") is None else float(request.POST.get("C8"))),int(0 if request.POST.get("C9") is None else float(request.POST.get("C9"))),int(0 if request.POST.get("C10") is None else float(request.POST.get("C10"))),int(0 if request.POST.get("C11") is None else float(request.POST.get("C11"))),int(0 if request.POST.get("C12") is None else float(request.POST.get("C12"))),int(0 if request.POST.get("C13") is None else float(request.POST.get("C13"))),int(0 if request.POST.get("C14") is None else float(request.POST.get("C14"))),int(0 if request.POST.get("C15") is None else float(request.POST.get("C15")))]
            #print(activities)
            #print(sum(activities))
            count=sum(activities)

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
            print(X_train)
            print(X_test)




            




            # Predicting a single new observation
            """Predict if the location with the following informations involves certain percentage of total tourist arrivals:
            place:    
            Year: 2018
            No.of other tourist attraction spots(within 25km radius): 2
            No. of available major tourist activities  nearby: 3
            Main purpose of visit:holiday/Pleasure 
            Accessibility status: Good
            Accomodation status: Better
            health services status:fair 
            n
            0=> holiday/pleasure
            6=>treeking&mountaineering
            7=>treeking
            37=>pilgrimage
            """
            v=X[[0,6,7,37],:][:, [0,1,2]]

            

            

                


            

            def get_hot_enc_val(n,v):
                A=X[[0,6,7,37],:][:, [0,1,2]]
                if ((n==0) or (n==6) or (n==7) or (n==37)):
                    if(n==0):
                        n=0
                    elif n==6 :
                        n=1
                    elif n==7 :
                        n=2
                    else :
                        n=3
                    v1=np.asscalar(A[[n],:][:, [0]])
                    v2=np.asscalar(A[[n],:][:, [1]])
                    v3=np.asscalar(A[[n],:][:, [2]])
                    if v==1 :
                        return v1
                    elif v==2 :
                        return v2
                    elif v==3:
                        return v3
            
            v1=get_hot_enc_val(z,1)
            v2=get_hot_enc_val(z,2)
            v3=get_hot_enc_val(z,3)
            


            
            REG_model=load_model('assets/REG_MLP_model.h5')
            new_prediction = REG_model.predict(sc.transform(np.array([[v1,v2,v3,2018,SPOTS,count,a_status,am_status,m_status]])))
            #new_prediction = REG.predict(sc.transform(np.array([[v1,v2,v3,2018,4,3,3,2,3]])))
            new_prediction_value=abs(np.asscalar(new_prediction))
            print(new_prediction_value)

            #dic[place]=new_prediction_value
            dic.append({place:new_prediction_value})

       

        




        

        

        context = {
        'new_prediction' :new_prediction_value ,  
        'place':place ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
        'predicks':dic ,
        

       
        }


        return TemplateResponse(request,'PredictionEngine/predicts.html',context)


    if (id==3):
        #data collecting...converting dataset to html....
        df = pd.read_csv('assets/Touristarrival_monthly.csv')
        df1=df.iloc[:5]
        html_table_template = df1.to_html(index=False)
        html_table=df.to_html(index=False)
        #data observation and log transformation
        # df.index=pd.to_datetime(df['Month'])
        # df['#Tourists'].plot()
        # mpl.pyplot.ylabel("No.of Toursits Arrivals ")
        # mpl.pyplot.xlabel("Year")
        # mpl.pyplot.show()
        #  #storing plots in bytes
        # f = io.BytesIO()
        # mpl.pyplot.savefig(f, format="png", dpi=600,bbox_inches='tight')
        # image_base64 = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
        # f.close()
        # mpl.pyplot.clf()

        # series=df['#Tourists']
        # logtransformed=np.log(series)
        # logtransformed.plot()
        # mpl.pyplot.ylabel("log Scale(No.of Toursits Arrivals) ")
        # mpl.pyplot.xlabel("Year")
        # mpl.pyplot.show()
        #  #storing plots in bytes
        # g = io.BytesIO()
        # mpl.pyplot.savefig(g, format="png", dpi=600,bbox_inches='tight')
        # image_base64g = base64.b64encode(g.getvalue()).decode('utf-8').replace('\n', '')
        # f.close()
        # mpl.pyplot.clf()
        # #Train test split
        # percent_training=0.80
        # split_point=round(len(series)*percent_training)
        # print(split_point)
        # training , testing = series[0:split_point] , series[split_point:]
        # training=np.log(training)


        
        # #differencing to achieve stationarity
        # training_diff=training.diff(periods=1).values[1:]

        # #plot of residual log differenced series
        # mpl.pyplot.plot(training_diff)
        # mpl.pyplot.title("Tourist arrivals data log-differenced")
        # mpl.pyplot.xlabel("Years")
        # mpl.pyplot.ylabel("Toursits arrivals")


        # #ACF and PACF plots 1(with log differenced training data)
        # lag_acf=acf(training_diff,nlags=40)
        # lag_pacf=pacf(training_diff,nlags=40,method='ols')

        # #plot ACF
        # mpl.pyplot.figure(figsize=(15,5))
        # mpl.pyplot.subplot(121)
        # mpl.pyplot.stem(lag_acf)
        # mpl.pyplot.axhline(y=0,linestyle='-',color='black')
        # mpl.pyplot.axhline(y=-1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        # mpl.pyplot.axhline(y=1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        # mpl.pyplot.xlabel('lag')
        # mpl.pyplot.ylabel("ACF")
        #  #storing plots in bytes
        # h = io.BytesIO()
        # mpl.pyplot.savefig(h, format="png", dpi=600,bbox_inches='tight')
        # image_base64h = base64.b64encode(h.getvalue()).decode('utf-8').replace('\n', '')
        # h.close()
        # mpl.pyplot.clf()

        # #plot PACF
        # mpl.pyplot.figure(figsize=(15,5))
        # mpl.pyplot.subplot(122)
        # mpl.pyplot.stem(lag_pacf)
        # mpl.pyplot.axhline(y=0,linestyle='-',color='black')
        # mpl.pyplot.axhline(y=-1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        # mpl.pyplot.axhline(y=1.96/np.sqrt(len(training)),linestyle='--',color='gray')
        # mpl.pyplot.xlabel('lag')
        # mpl.pyplot.ylabel("PACF")
        #  #storing plots in bytes
        # i = io.BytesIO()
        # mpl.pyplot.savefig(i, format="png", dpi=600,bbox_inches='tight')
        # image_base64i = base64.b64encode(i.getvalue()).decode('utf-8').replace('\n', '')
        # i.close()
        # mpl.pyplot.clf()

        # #SARIMA Model specification
        # model=sm.tsa.statespace.SARIMAX(training,order=(2,0,3),seasonal_order=(2,1,0,12),trend='c',enforce_invertibility=False,enforce_stationarity=False)

        # # fit model
        # model_fit = model.fit()

        # model_fit.save("assets/REG_SARIMA_model.pickle")

        # print(model_fit.summary())

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
        # mpl.pyplot.figure(figsize=(10,5))
        # mpl.pyplot.plot(forecast,'r')
        # mpl.pyplot.plot(series,'b')
        # mpl.pyplot.legend(['Predicted test values','Actual data values'])

        # mpl.pyplot.title('RMSE:%.2f'% np.sqrt(sum((forecast-testing)**2)/len(testing)))
        # mpl.pyplot.ylabel("No.of Toursits Arrivals Monthly")
        # mpl.pyplot.xlabel("Year")
        # mpl.pyplot.autoscale(enable='True',axis='x',tight=True)
        # mpl.pyplot.axvline(x=series.index[split_point],color='black');
         #storing plots in bytes
        # j = io.BytesIO()
        # mpl.pyplot.savefig(j, format="png", dpi=600,bbox_inches='tight')
        # image_base64j = base64.b64encode(j.getvalue()).decode('utf-8').replace('\n', '')
        # j.close()
        # mpl.pyplot.clf()

        forecaste=model_fitted.forecast(len(df)-238)
        forecast_next=forecaste[62:]
        forecast_next=np.exp(forecast_next)
        print(forecast_next)
        # mpl.pyplot.figure(figsize=(10,5))
        # mpl.pyplot.plot(forecast_next,'r')
        # mpl.pyplot.plot(series,'b')
        # mpl.pyplot.legend(['Predicted next steps values'])
        # mpl.pyplot.title('Monthly tourist arrivals predictions')
        # mpl.pyplot.ylabel("No.of Toursits Arrivals ")
        # mpl.pyplot.xlabel("Year")
        # mpl.pyplot.autoscale(enable='True',axis='x',tight=True)

        # #storing plots in bytes
        # k = io.BytesIO()
        # mpl.pyplot.savefig(k, format="png", dpi=600,bbox_inches='tight')
        # image_base64k = base64.b64encode(k.getvalue()).decode('utf-8').replace('\n', '')
        # k.close()
        # mpl.pyplot.clf()
        # getting details of id
        all_details=Predictions.objects.get(id=id)
        nexts=forecast_next

       






         #parsing suitable context for redering...
        context = {
        'all_details':all_details ,
        'html_table':html_table ,
        'html_table_template': html_table_template,
      
        'next_years_values':nexts ,
       
        }
        return TemplateResponse(request,'PredictionEngine/predict_detail3.html',context)



