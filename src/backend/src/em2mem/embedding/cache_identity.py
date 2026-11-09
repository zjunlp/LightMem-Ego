"""Configuration identities for vectors, without accessing lazy model properties."""
import hashlib
import os


def embedding_cache_identity(model):
    def declared_values(value):
        # Reading descriptors can initialize a model. Inspect only stored scalar
        # configuration, including inherited class constants used by adapters.
        stored = {}
        for cls in reversed(type(value).__mro__):
            stored.update(vars(cls))
        stored.update(getattr(value, "__dict__", {}))
        result = {"class": f"{type(value).__module__}.{type(value).__qualname__}"}
        for name in ("text_model_name", "model_name", "text_backend", "revision",
                     "model_revision", "dimension", "embedding_dimension", "normalize", "remote_url"):
            item = stored.get(name)
            if isinstance(item, (str, int, float, bool)):
                # Endpoints may contain credentials or signed query parameters.
                result[name] = hashlib.sha256(item.encode()).hexdigest() if name == "remote_url" and isinstance(item, str) else item
        return result, stored

    identity, stored = declared_values(model)
    nested = stored.get("_text_model")
    # A wrapper's declared identity must be stable before and after lazy load.
    if nested is not None and not (stored.get("text_model_name") or stored.get("model_name")):
        identity["loaded_text_model"] = declared_values(nested)[0]
    if stored.get("text_backend") == "remote":
        identity["remote_config"] = {
            name: hashlib.sha256(os.getenv(name, "").encode()).hexdigest()
            for name in ("EM2MEM_TEXT_EMBED_URL", "EM2MEM_TEXT_EMBED_HOST",
                         "EM2MEM_TEXT_EMBED_PORT", "EM2MEM_TEXT_EMBED_NORMALIZE")
        }
    return identity
