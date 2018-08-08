#!/usr/bin/python3
#Usuario Id MAx 338 Mion 0
#Servicio Max 5824 min 0
# Calificaicon 1~5

import datetime
import csv
import random

myData = [['servicio','fecha','usuario','calificacion']]
rango = range(339)
ciclo = range(20)
for valorUsuario in rango:
    print("------------")
    for numCiclo in ciclo:
        print("-",numCiclo)
        usuario = valorUsuario# de 1 a 338 se repite 20 veces
        print("usuario: ",usuario)
        fecha = datetime.datetime.now()#Justo ahora
        print("fecha: ",fecha)
        servicio = random.randint(0, 5824) #Numero aleatorio de 0 a 5824
        print("servicio: ",servicio)
        calificacion = random.randint(1, 5)#Numero de 1 a 5 
        print("cal: ",calificacion)
        myData.append([servicio,fecha,usuario,calificacion])




with open('cal.csv', 'w',newline='') as myFile:  
   writer = csv.writer(myFile)
   writer.writerows(myData)