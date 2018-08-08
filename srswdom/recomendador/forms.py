from django.forms import ModelForm, Textarea
from recomendador.models import Calificacion


class ReviewForm(ModelForm):
    class Meta:
       model = Calificacion
       #fields = ['rating', 'comment']
       fields = ['calificacion','usuario']
       #widgets = {            'comment': Textarea(attrs={'cols': 40, 'rows': 15}),        }