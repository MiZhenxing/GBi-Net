import importlib

# find the model definition by name, for example gbinet (gbinet.py)
def find_model_def(file_name, model_name):
    file_name = 'models.{}'.format(file_name)
    file = importlib.import_module(file_name)
    return getattr(file, model_name)

def find_loss_def(file_name, loss_name):
    file_name = 'models.{}'.format(file_name)
    file = importlib.import_module(file_name)
    return getattr(file, loss_name)
