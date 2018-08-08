from django.shortcuts import get_object_or_404,render
from .models import Servicio, Usuario, Dominio, Calificacion
from django.http import HttpResponseRedirect
from django.urls import reverse
from .forms import ReviewForm
import datetime
#USUARIO
def lista_usuarios(request):
	lista_usuarios = Usuario.objects.order_by('id')
	context = {'lista_usuarios':lista_usuarios}
	return render(request,'recomendador/lista_usuarios.html',context)

def detalle_usuario(request, usuario_id):
    usuario = get_object_or_404(Usuario, pk=usuario_id)
    return render(request, 'recomendador/detalle_usuario.html', {'usuario': usuario})
#SERVICIO
def lista_servicios(request):
	lista_servicios = Servicio.objects.order_by('id')
	context = {'lista_servicios':lista_servicios}
	return render(request,'recomendador/lista_servicios.html',context)

def detalle_servicio(request, servicio_id):
    servicio = get_object_or_404(Servicio, pk=servicio_id)
    form = ReviewForm()
    return render(request, 'recomendador/detalle_servicio.html', {'servicio': servicio,'form': form})

#@login_required
def nueva_calificacion(request, servicio_id):
    servicio = get_object_or_404(Servicio, pk=servicio_id)
    #usuario = get_object_or_404(Usuario, pk=usuario_id)
    form = ReviewForm(request.POST)
    if form.is_valid():
        cal = form.cleaned_data['calificacion']
        usuario = form.cleaned_data['usuario']
         #comentario = form.cleaned_data['comment']
        calificacion = Calificacion()
        calificacion.servicio = servicio
        calificacion.usuario = usuario
        print("cal",cal)
        calificacion.calificacion = cal
        #calificacion.comentario = comentario
        calificacion.fecha_calificacion = datetime.datetime.now()
        calificacion.save()
       # update_clusters()#Cuando una nueva reseña sea agregada, se actualizan los clusters
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('recomendador:detalle_servicio', args=(servicio_id,)))
    
    return render(request, 'recomendador/detalle_servicio.html', {'servicio': servicio,'form': form})


#PROVEEDOR

def lista_proveedores(request):
	lista_proveedores = Proveedor.objects.order_by('id')
	context = {'lista_proveedores':lista_proveedores}
	return render(request,'recomendador/lista_proveedores.html',context)

def detalle_proveedor(request, proveedor_id):
    context = {'proveedor_id':proveedor_id}
    return render(request, 'recomendador/detalle_proveedor.html',context)

#DOMINIO
def lista_dominios(request):
	lista_dominios = Dominio.objects.order_by('id')
	context = {'lista_dominios':lista_dominios}
	return render(request,'recomendador/lista_dominios.html',context)

def detalle_dominio(request, dominio_id):
    dominio = get_object_or_404(Dominio, pk=dominio_id)
    return render(request, 'recomendador/detalle_dominio.html', {'dominio': dominio})

#RECOMENDACION
def calificaciones(request):
	calificaciones = Calificacion.objects.order_by('id')
	context = {'calificaciones':calificaciones}
	return render(request,'recomendador/calificaciones.html',context)