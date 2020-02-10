from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseRedirect
from django.views.generic import ListView

# utilities for introspection
from .utils.introspect import as_dict

# specific modules
from .nlp.services import preds as preds_module
from .nlp.services import edits as edits_module
from .nlp.services import complete as complete_module
from .nlp.services import model as model_module
from .forms import ModelsForm, DEFAULT_MODEL, get_current_model

def editor(request):
    if request.method == 'POST':
        form = ModelsForm(request.POST)
        model_module.model_change(request.POST['model'])
    else:
        form = ModelsForm()

    return render(request, 'editor/editor.html', {'form': form, 'model': get_current_model()})

def ajax_preds(request):
    return JsonResponse(as_dict(preds_module, request))

def ajax_edits(request):
    return JsonResponse(as_dict(edits_module, request, coerce=True))

def ajax_complete(request):
    return JsonResponse(as_dict(complete_module, request, coerce=True))

def handle_404(request, exception):
    return redirect("editor")
