from django.contrib import admin

from .models import Usuario,Servicio,Proveedor ,Dominio,TipoUsuario, Calificacion


class TipoUsuarioAdmin(admin.ModelAdmin):
    model = TipoUsuario
    list_display = ['tipousuario', 'descripciontipousuario']

class UsuarioAdmin(admin.ModelAdmin):
    model = TipoUsuario
    list_display = ['id','direccionip', 'tipousuario']

class ServicioAdmin(admin.ModelAdmin):
    model = TipoUsuario
    list_display = ['id','direccionwsdl','direccionip','proveedor','dominio']

class DominioAdmin(admin.ModelAdmin):
    model = Dominio
    #list_display = ['id','nombredominio']
    list_display = ['nombredominio']

class CalificacionAdmin(admin.ModelAdmin):
	model = Calificacion
	list_display = ['servicio','fecha_calificacion','usuario','calificacion'] 

admin.site.register(Usuario,UsuarioAdmin)
admin.site.register(Servicio,ServicioAdmin)
admin.site.register(Proveedor)
admin.site.register(Dominio,DominioAdmin)
admin.site.register(TipoUsuario,TipoUsuarioAdmin)
admin.site.register(Calificacion,CalificacionAdmin)