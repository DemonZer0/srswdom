from recomendador.models import Calificacion, Servicio, Usuario, Dominio

servicio=Servicio() 
dominio=Dominio()
    #id
servicio.id = 4701
    ##• Dirección WSDL
servicio.direccionwsdl = 'http://www.birdwellmusic.com/BirdwellMusicServices.asmx?wsdl'
    #ipaddress 8.23.224.110,
  
servicio.direccionip = '0.0.0.0'
    #Proveedor
 
servicio.proveedor = ''

    #Dominio
dominio.id =5
servicio.dominio = dominio     
servicio.save()