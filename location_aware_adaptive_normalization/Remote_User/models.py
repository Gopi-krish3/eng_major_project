from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class wildfire_danger_forecasting(models.Model):

    UniqueId= models.CharField(max_length=200)
    AdminUnit= models.CharField(max_length=300)
    CalFireIncident= models.CharField(max_length=300)
    CanonicalUrl= models.CharField(max_length=300)
    Counties= models.CharField(max_length=300)
    CrewsInvolved= models.CharField(max_length=100)
    Dozers= models.CharField(max_length=100)
    Engines= models.CharField(max_length=100)
    Extinguished= models.CharField(max_length=100)
    Fatalities= models.CharField(max_length=100)
    Featured= models.CharField(max_length=300)
    Final= models.CharField(max_length=100)
    Helicopters= models.CharField(max_length=100)
    Injuries= models.CharField(max_length=300)
    Latitude= models.CharField(max_length=100)
    Location= models.CharField(max_length=300)
    Longitude= models.CharField(max_length=100)
    MajorIncident= models.CharField(max_length=300)
    Name= models.CharField(max_length=300)
    PercentContained= models.CharField(max_length=100)
    PersonnelInvolved= models.CharField(max_length=100)
    Description= models.CharField(max_length=300)
    Started= models.CharField(max_length=300)
    Status= models.CharField(max_length=100)
    Updated= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=100)


class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


