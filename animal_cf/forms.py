from django import forms
from .models import AnimalImage

class AnimalImageForm(forms.ModelForm):
    class Meta:
        model = AnimalImage
        fields = ('animal_image',)