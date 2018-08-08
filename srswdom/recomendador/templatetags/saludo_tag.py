from django import template

register = template.Library()
#AS22683 Koninklijke Philips Electronics N.V. (internet presence)
@register.simple_tag
def hello_world(name):
	final = name.find(" ")
	salute = name[2:final]
	
	salute=salute.replace(" ", "")
	print(name)
	print(salute)
	if salute is '':
		return "0000"
	return salute

    #<h5><p><b>Proveedor del servicio:</b><a href="{% url 'recomendador:detalle_proveedor' servicio.proveedor %}">{{ servicio.proveedor }}</a></p></h5>

    # <h2><a href="{% url 'recomendador:detalle_proveedor' salute_text %}">Proovedor: {{ servicio.proveedor}}</a></h2>
    
    