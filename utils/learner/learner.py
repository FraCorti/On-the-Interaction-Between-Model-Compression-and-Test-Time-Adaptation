learners = {}

learners_methods = []


def register(name):
    def decorator(cls):
        learners[name] = cls
        return cls

    learners_methods.append(name)
    return decorator


def make(name, model, optimizer, **kwargs):
    if name is None:
        return None
    learner = learners[name](model, optimizer, **kwargs)
    return learner
