from django.db.models import  Count, Avg
from django.shortcuts import render, redirect

import openpyxl
from django.http import HttpResponse,FileResponse

import string

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

import pandas as pd


# Create your views here.
from Remote_User.models import ClientRegister_Model,wildfire_danger_forecasting,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def View_Predicted_Wildfire_Danger_Forecasting_Details(request):

    obj = wildfire_danger_forecasting.objects.all()
    return render(request, 'SProvider/View_Predicted_Wildfire_Danger_Forecasting_Details.html', {'objs': obj})

def View_Predicted_Wildfire_Danger_Forecasting_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Normal Fire'
    print(kword)
    obj = wildfire_danger_forecasting.objects.all().filter(Prediction=kword)
    obj1 = wildfire_danger_forecasting.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Danger Fire'
    print(kword1)
    obj1 = wildfire_danger_forecasting.objects.all().filter(Prediction=kword1)
    obj11 = wildfire_danger_forecasting.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)


    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/View_Predicted_Wildfire_Danger_Forecasting_Ratio.html', {'objs': obj})


def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})


def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def likeschart1(request,like_chart):
    charts =detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart1.html", {'form':charts, 'like_chart':like_chart})

"""
def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = wildfire_danger_forecasting.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.UniqueId, font_style)
        ws.write(row_num, 1, my_row.AdminUnit, font_style)
        ws.write(row_num, 2, my_row.CalFireIncident, font_style)
        ws.write(row_num, 3, my_row.CanonicalUrl, font_style)
        ws.write(row_num, 4, my_row.Counties, font_style)
        ws.write(row_num, 5, my_row.CrewsInvolved, font_style)
        ws.write(row_num, 6, my_row.Dozers, font_style)
        ws.write(row_num, 7, my_row.Engines, font_style)
        ws.write(row_num, 8, my_row.Extinguished, font_style)
        ws.write(row_num, 9, my_row.Fatalities, font_style)
        ws.write(row_num, 10, my_row.Featured, font_style)
        ws.write(row_num, 11, my_row.Final, font_style)
        ws.write(row_num, 12, my_row.Helicopters, font_style)
        ws.write(row_num, 13, my_row.Injuries, font_style)
        ws.write(row_num, 14, my_row.Latitude, font_style)
        ws.write(row_num, 15, my_row.Location, font_style)
        ws.write(row_num, 16, my_row.Longitude, font_style)
        ws.write(row_num, 17, my_row.MajorIncident, font_style)
        ws.write(row_num, 18, my_row.Name, font_style)
        ws.write(row_num, 19, my_row.PercentContained, font_style)
        ws.write(row_num, 20, my_row.PersonnelInvolved, font_style)
        ws.write(row_num, 21, my_row.Description, font_style)
        ws.write(row_num, 22, my_row.Started, font_style)
        ws.write(row_num, 23, my_row.Status, font_style)
        ws.write(row_num, 24, my_row.Updated, font_style)
        ws.write(row_num, 25, my_row.Prediction, font_style)


    wb.save(response)
    return response
"""    

def Download_Trained_DataSets(request):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "sheet1"

    # Write header row
    header_row = ["UniqueId", "AdminUnit", "CalFireIncident", "CanonicalUrl","Counties","CrewsInvolved","Dozers", "Engines",
            "Extinguished", "Fatalities", "Featured", "Final", "Helicopters", "Injuries", "Latitude", "Location", "Longitude",
            "MajorIncident", "Name", "PercentContained", "PersonnelInvolved", "Description", "Started", 
            "Status", "Updated", "Prediction"]
    ws.append(header_row)

    # Write data rows
    predictions = wildfire_danger_forecasting.objects.all()
    for prediction in predictions:
        data_row = [prediction.UniqueId, prediction.AdminUnit, prediction.CalFireIncident, prediction.CanonicalUrl, 
                    prediction.Counties, prediction.CrewsInvolved, prediction.Dozers, prediction.Engines, prediction.Extinguished,
                    prediction.Fatalities, prediction.Featured, prediction.Final, prediction.Helicopters, prediction.Injuries,
                    prediction.Latitude, prediction.Location, prediction.Longitude, prediction.MajorIncident, prediction.Name,
                    prediction.PercentContained, prediction.PersonnelInvolved, prediction.Description, prediction.Started, 
                    prediction.Status, prediction.Updated, prediction.Prediction]
        ws.append(data_row)

    wb.save('Predicted_Datasets.xlsx')
    response=FileResponse(open('Predicted_Datasets.xlsx','rb'))
    #response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="Predicted_Datasets.xlsx"'
    return response

def Train_Test_DataSets(request):

    detection_accuracy.objects.all().delete()

    df = pd.read_csv('Datasets.csv')

    def apply_results(label):
        if (label == 0):
            return 0  # Normal Fir
        elif (label == 1):
            return 1  # Danger Fir

    df['results'] = df['Label'].apply(apply_results)

    cv = CountVectorizer(lowercase=False)

    y = df['results']
    X = df['UniqueId'].apply(str)

    print("X Values")
    print(X)
    print("Labels")
    print(y)

    X = cv.fit_transform(X)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train.shape, X_test.shape, y_train.shape
    print("X_test")
    print(X_test)
    print(X_train)

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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)


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
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    print("Gradient Boosting Classifier")

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
        X_train,
        y_train)
    clfpredict = clf.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, clfpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, clfpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, clfpredict))
    models.append(('GradientBoostingClassifier', clf))
    detection_accuracy.objects.create(names="Gradient Boosting Classifier",
                                      ratio=accuracy_score(y_test, clfpredict) * 100)

    obj = detection_accuracy.objects.all()

    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})














