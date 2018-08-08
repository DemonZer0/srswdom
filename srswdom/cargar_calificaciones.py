import sys, os
import pandas as pd
import datetime

os.environ.setdefault("DJANGO_SETTINGS_MODULE","srswdom.settings") 

import django
django.setup()

from recomendador.models import Calificacion, Servicio, Usuario

def guardar_Calificacion_de_fila(Calificacion_fila):
	
	print("aqui5!")

	calificacion=Calificacion()
	usuario = Usuario()
	servicio = Servicio()
	print("aqui6!")
	
	

	print("aqui7!")
	#id
	calificacion.id = Calificacion_fila[0]


	##servicio
	servicio.id=Calificacion_fila[1]
	calificacion.servicio = servicio
	#fecha_calificacionaddress 8.23.224.110,
	
	calificacion.fecha_calificacion = datetime.datetime.now()
	#usuario
	#usuario.nombreusuario = 
	usuario.id =  Calificacion_fila[3]
	calificacion.usuario =usuario

	#calificacion
	#calificacion.id =Calificacion_fila[4]
	calificacion.calificacion = Calificacion_fila[4]

	calificacion.save()
"""

	print("id: "+calificacion.id
		+" servicio:"+calificacion.servicio 
		+" fecha_calificacion: "+calificacion.fecha_calificacion
		+" usuario: "+calificacion.usuario
		+"calificacion: "+calificacion.calificacion) """
	
	
	

# comando: python cargar_calificaciones.py data/cal.csv
if __name__ == "__main__":
	if len(sys.argv) == 2:
		print("Leyendo del archivo "+str(sys.argv[1]))
		print("aqui1!")
		Calificacions_df = pd.read_csv(sys.argv[1])
		print("aqui2!")
		print(Calificacions_df)
		print("aqui3!")
		Calificacions_df.where(Calificacions_df.notnull(), None)
		print("aqui4!")
		Calificacions_df.apply(
			guardar_Calificacion_de_fila,# Function to apply to each column/row

			axis=1 #axis : {0 or ‘index’, 1 or ‘columns’}, default 0
					#     0 or ‘index’: apply function to each column
        			#1 or ‘columns’: apply function to each row
		)

		print("Existen {} calificaciones".format(Calificacion.objects.count()))
	else:
		print("Por favor, introduce la direccion del archivo Calificacions")	

"""
for i, row in df.iterrows():
    print(pd.notnull(row))
    if pd.notnull(row):
        df.loc[i, 'denyData'] = base64.b64decode(parse.unquote(row['denyData']))
    else:
        df.loc[i, 'denyData'] = np.nan
  """