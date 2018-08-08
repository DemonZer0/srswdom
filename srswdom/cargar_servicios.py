import sys, os
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE","srswdom.settings") 

import django
django.setup()

from recomendador.models import Servicio,  Dominio

def guardar_servicio_de_fila(servicio_fila):
	"""usuario=Usuario()
	tipousuario = TipoUsuario()

	usuario.id = usuario_fila[0]
	#usuario.nombreusuario = usuario_fila[1]

	usuario.direccionip = usuario_fila[1]
	tipousuario.id = usuario_fila[2]
	usuario.tipousuario = tipousuario
	#usuario.tipousuario = usuario_fila[3]
	print("id: "+str(usuario.id )+" ip:"+usuario.direccionip+" tipo: "+str(usuario.tipousuario) )
	
	usuario.save()"""


	servicio=Servicio()
	dominio = Dominio()
	print("id: "+str( servicio_fila[0] )+" WSDL:"+ servicio_fila[1] +" ip: "+str( servicio_fila[2])+" Proveedor: "+str( servicio_fila[3])+"Dominiio: "+str( servicio_fila[4]))
	#id
	servicio.id = servicio_fila[0]
	##•	Dirección WSDL
	servicio.direccionwsdl = servicio_fila[1]
	#ipaddress 8.23.224.110,
	
	if pd.isna(servicio_fila[2]):
		servicio_fila[2] ='0.0.0.0'
	servicio.direccionip = servicio_fila[2]
	#Proveedor
	#proveedor.nombreproveedor = 
	if pd.isna(servicio_fila[3]):
		servicio_fila[3] =''
	servicio.proveedor = servicio_fila[3]

	#Dominio
	dominio.id =servicio_fila[4]
	servicio.dominio = dominio


	print("id: "+str(servicio.id )+" WSDL:"+servicio.direccionwsdl +" ip: "+str(servicio.direccionip)+" Proveedor: "+str(servicio.proveedor)+"Dominiio: "+str(servicio.dominio) )
	
	
	servicio.save()

# comando: python cargar_servicios.py data/listaservicio.csv
if __name__ == "__main__":
	if len(sys.argv) == 2:
		print("Leyendo del archivo "+str(sys.argv[1]))
		servicios_df = pd.read_csv(sys.argv[1])
		print(servicios_df)
		servicios_df.where(servicios_df.notnull(), None)
		servicios_df.apply(
			guardar_servicio_de_fila,# Function to apply to each column/row

			axis=1 #axis : {0 or ‘index’, 1 or ‘columns’}, default 0
					#     0 or ‘index’: apply function to each column
        			#1 or ‘columns’: apply function to each row
		)

		print("Existen {} servicios".format(Servicio.objects.count()))
	else:
		print("Por favor, introduce la direccion del archivo servicios")	

"""
for i, row in df.iterrows():
    print(pd.notnull(row))
    if pd.notnull(row):
        df.loc[i, 'denyData'] = base64.b64decode(parse.unquote(row['denyData']))
    else:
        df.loc[i, 'denyData'] = np.nan
  """