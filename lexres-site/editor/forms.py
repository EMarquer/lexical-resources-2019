from django import forms  
from .utils.models import get_model_list, DEFAULT_MODEL, get_current_model

class ModelsForm(forms.Form):  
    model = forms.ChoiceField(
        choices=[(model, model) for model in get_model_list()],
        label="Modèle",
        initial=DEFAULT_MODEL
        )