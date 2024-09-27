
from django.shortcuts import render, redirect, get_object_or_404

import string

import re
from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,wildfire_danger_forecasting,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Wildfire_Danger_Forecasting(request):

        if request.method == "POST":


            UniqueId= request.POST.get('UniqueId')
            AdminUnit= request.POST.get('AdminUnit')
            CalFireIncident= request.POST.get('CalFireIncident')
            CanonicalUrl= request.POST.get('CanonicalUrl')
            Counties= request.POST.get('Counties')
            CrewsInvolved= request.POST.get('CrewsInvolved')
            Dozers= request.POST.get('Dozers')
            Engines= request.POST.get('Engines')
            Extinguished= request.POST.get('Extinguished')
            Fatalities= request.POST.get('Fatalities')
            Featured= request.POST.get('Featured')
            Final= request.POST.get('Final')
            Helicopters= request.POST.get('Helicopters')
            Injuries= request.POST.get('Injuries')
            Latitude= request.POST.get('Latitude')
            Location= request.POST.get('Location')
            Longitude= request.POST.get('Longitude')
            MajorIncident= request.POST.get('MajorIncident')
            Name= request.POST.get('Name')
            PercentContained= request.POST.get('PercentContained')
            PersonnelInvolved= request.POST.get('PersonnelInvolved')
            Description= request.POST.get('Description')
            Started= request.POST.get('Started')
            Status= request.POST.get('Status')
            Updated= request.POST.get('Updated')

            df = pd.read_csv('Datasets.csv')


            def apply_results(label):
                if (label == 0):
                    return 0  # Normal Fir
                elif (label == 1):
                    return 1  # Danger Fir

            df['results'] = df['Label'].apply(apply_results)

            cv = CountVectorizer(lowercase=False)

            y = df['results']
            X = df["UniqueId"].apply(str)

            print("X Values")
            print(X)
            print("Labels")
            print(y)

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            X_train.shape, X_test.shape, y_train.shape

            print("Convolutional Neural Network (CNN)")

            from sklearn.neural_network import MLPClassifier
            mlpc = MLPClassifier().fit(X_train, y_train)
            y_pred = mlpc.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, y_pred) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, y_pred))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, y_pred))
            models.append(('MLPClassifier', mlpc))
            detection_accuracy.objects.create(names="Convolutional Neural Network (CNN)",
                                              ratio=accuracy_score(y_test, y_pred) * 100)

            # SVM Model
            print("SVM")
            from sklearn import svm
            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, predict_svm))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, predict_svm))
            models.append(('svm', lin_clf))

            print("KNeighborsClassifier")
            from sklearn.neighbors import KNeighborsClassifier
            kn = KNeighborsClassifier()
            kn.fit(X_train, y_train)
            knpredict = kn.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, knpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, knpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, knpredict))
            models.append(('KNeighborsClassifier', kn))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            UniqueId1 = [UniqueId]
            vector1 = cv.transform(UniqueId1).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if prediction == 0:
                val = 'Normal Fire'
            elif prediction == 1:
                val = 'Danger Fire'

            print(val)
            print(pred1)

            wildfire_danger_forecasting.objects.create(
            UniqueId=UniqueId,
            AdminUnit=AdminUnit,
            CalFireIncident=CalFireIncident,
            CanonicalUrl=CanonicalUrl,
            Counties=Counties,
            CrewsInvolved=CrewsInvolved,
            Dozers=Dozers,
            Engines=Engines,
            Extinguished=Extinguished,
            Fatalities=Fatalities,
            Featured=Featured,
            Final=Final,
            Helicopters=Helicopters,
            Injuries=Injuries,
            Latitude=Latitude,
            Location=Location,
            Longitude=Longitude,
            MajorIncident=MajorIncident,
            Name=Name,
            PercentContained=PercentContained,
            PersonnelInvolved=PersonnelInvolved,
            Description=Description,
            Started=Started,
            Status=Status,
            Updated=Updated,
            Prediction=val)

            return render(request, 'RUser/Predict_Wildfire_Danger_Forecasting.html',{'objs':val})
        return render(request, 'RUser/Predict_Wildfire_Danger_Forecasting.html')

