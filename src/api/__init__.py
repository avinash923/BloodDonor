"""
src/api/__init__.py
API package exports.
"""

__all__ = ["run_api_server", "BloodDonorApiHandler", "SystemState"]


def __getattr__(name):
    if name in __all__:
        from src.api import server
        return getattr(server, name)
    raise AttributeError(name)
