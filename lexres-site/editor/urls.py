from django.urls import path

from . import views

urlpatterns = [
    path('', views.editor, name='editor'),
    path('preds', views.ajax_preds, name='preds'),
    path('edits', views.ajax_edits, name='edits'),
    path('complete', views.ajax_complete, name='complete'),
]
