from django.urls import path
from . import views

app_name = 'recomendador'

urlpatterns = [
    # ex: /
    path('', views.lista_usuarios, name='lista_usuarios'),
    # ex: /usuarios/
    path('usuario', views.lista_usuarios, name='lista_usuarios'),
    # ex: /usuario/5/
    path('usuario/<int:usuario_id>/', views.detalle_usuario, name='detalle_usuario'),
     # ex: /servicios/
    path('servicio', views.lista_servicios, name='lista_servicios'),
    # ex: /servicio/5/
    path('servicio/<int:servicio_id>/', views.detalle_servicio, name='detalle_servicio'),
    path('servicio/<int:servicio_id>/nueva_calificacion/',views.nueva_calificacion,name='nueva_calificacion'),
    # ex: /proveedores/
    path('proveedor', views.lista_proveedores, name='lista_proveedores'),
        # ex: /proveedor/A1244213/
    path('proveedor/<int:proveedor_id>/', views.detalle_proveedor, name='detalle_proveedor'),
     # ex: /dominios/
    path('dominio', views.lista_dominios, name='lista_dominios'),
    # ex: /servicio/5/
    path('dominio/<int:dominio_id>/', views.detalle_dominio, name='detalle_dominio'),
    # ex: /calificaciones/
    path('calificacion', views.calificaciones, name='calificaciones'),    
   ]