from django.db import models
import numpy as np

class TipoUsuario(models.Model):
	tipousuario = models.CharField(max_length=250,blank = True)
	descripciontipousuario = models.CharField(max_length=250,blank = True)

class Usuario(models.Model):
	#id 0,
	#•	Nombre de usuario
	#usuario = models.OneToOneField(User,on_delete=models.CASCADE)
	#nombreusuario = models.CharField(max_length=100,blank=True)
	#ipaddress 12.108.127.138,
	direccionip = models.GenericIPAddressField(default='0.0.0.0')
	#direccionip = models.CharField(max_length=50, default='0.0.0.0')
	#•	Tipo de usuario
	tipousuario = models.ForeignKey(TipoUsuario, on_delete = models.CASCADE)


	
class Proveedor(models.Model):
	#id
	#•	Nombre del proveedor
	nombreproveedor = models.CharField(max_length=250,blank = True)
	#•	Organización
	nombreorganizacion = models.CharField(max_length=250,blank = True)
	#•	Rango AS
	rangoas = models.CharField(max_length=250,blank = True)
	#•	Grado AS
	gradoas = models.CharField(max_length=250,blank = True)

class Dominio(models.Model):
	#id
	#•	Nombre 
	nombredominio = models.CharField(max_length=250,blank = True)
	
class Servicio(models.Model):
	#id 0
	##•	Dirección WSDL
	direccionwsdl = models.CharField(max_length=250,blank = True)
	#ipaddress 8.23.224.110,
	direccionip = models.GenericIPAddressField(default='0.0.0.0' )
	#Proveedor
#	proveedor = models.ForeignKey(Proveedor,on_delete = models.CASCADE)
	proveedor = models.CharField(max_length=250,blank = True)
	#Dominio
	dominio = models.ForeignKey(Dominio, on_delete = models.CASCADE)

	def calificacion_promedio(self):
		califacionesTodas = list(map(lambda x: x.calificacion, self.calificacion_set.all()))
		return np.mean(califacionesTodas)


class Calificacion(models.Model):
    OPCIONES_CALIFICACION = (
        (1, '1: Servicio Muy Mal recomendado'),
        (2, '2: Servicio Mal recomendado'),
        (3, '3: Servicio Regular'),
        (4, '4: Servicio recomendado'),
        (5, '5: Servicio Muy recomendado'),
    )
    servicio = models.ForeignKey(Servicio, on_delete = models.CASCADE)
    fecha_calificacion= models.DateTimeField('fecha de calificación')
    usuario = models.ForeignKey(Usuario,on_delete=models.CASCADE)
    calificacion = models.IntegerField(choices=OPCIONES_CALIFICACION)
