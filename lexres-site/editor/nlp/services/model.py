from ...utils.models import set_model

def model_change(new_model):
    set_model(new_model)
    return ["new_model"]