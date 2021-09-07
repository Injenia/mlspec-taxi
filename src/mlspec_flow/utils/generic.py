from importlib import import_module

def dynamic_import(module, content=None):
    mod = import_module(module)
    if content is not None:
        return getattr(mod, content)
    return mod