"""Thread-local properties for derived query indexes, with lazy defaults."""
import threading


def query_index_property(name, default_factory):
    def state(instance):
        # setdefault keeps creation safe when a cold memory is first queried by
        # several threads. The persisted documents and embeddings stay shared.
        local = instance.__dict__.get('_query_index_state')
        if local is None:
            local = instance.__dict__.setdefault('_query_index_state', threading.local())
        generation = instance.__dict__.get('_query_data_generation', 0)
        if getattr(local, '_generation', None) != generation:
            local.__dict__.clear()
            local._generation = generation
        return local

    def get(instance):
        local = state(instance)
        if not hasattr(local, name):
            setattr(local, name, default_factory())
        return getattr(local, name)

    def set_value(instance, value):
        setattr(state(instance), name, value)

    def delete(instance):
        local = state(instance)
        if hasattr(local, name):
            delattr(local, name)

    return property(get, set_value, delete)
