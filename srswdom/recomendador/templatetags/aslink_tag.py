from django import template

register = template.Library()

@register.simple_tag
def extraerlink(name):
    salute = 'Hello' + name

    return salute

    #<h5><p><b>Proveedor del servicio:</b><a href="{% url 'recomendador:detalle_proveedor' servicio.proveedor %}">{{ servicio.proveedor }}</a></p></h5>