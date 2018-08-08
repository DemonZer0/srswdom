import sys, os
import pandas as pd

os.environ.setdefault("DJANGO_SETTINGS_MODULE","srswdom.settings") 

import django
django.setup()

from recomendador.models import Usuario, TipoUsuario

def guardar_usuario_de_fila(usuario_fila):
	usuario=Usuario()
	tipousuario = TipoUsuario()

	usuario.id = usuario_fila[0]
	#usuario.nombreusuario = usuario_fila[1]

	usuario.direccionip = usuario_fila[1]
	tipousuario.id = usuario_fila[2]
	usuario.tipousuario = tipousuario
	#usuario.tipousuario = usuario_fila[3]
	print("id: "+str(usuario.id )+" ip:"+usuario.direccionip+" tipo: "+str(usuario.tipousuario) )
	
	usuario.save()


# comando: python cargar_usuario.py data/listausuarios.csv
if __name__ == "__main__":
	if len(sys.argv) == 2:
		print("Leyendo del archivo "+str(sys.argv[1]))
		usuarios_df = pd.read_csv(sys.argv[1])
		print(usuarios_df)

		usuarios_df.apply(
			guardar_usuario_de_fila,# Function to apply to each column/row

			axis=1 #axis : {0 or ‘index’, 1 or ‘columns’}, default 0
					#     0 or ‘index’: apply function to each column
        			#1 or ‘columns’: apply function to each row
		)

		print("Existen {} usuarios".format(Usuario.objects.count()))
	else:
		print("Por favor, introduce la direccion del archivo Usuarios")	

