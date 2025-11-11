from .image_collector import ImageCollector

__all__ = ["RouteGenerator", "ImageCollector"]


def __getattr__(name):
    if name == "RouteGenerator":
        from .route_generator import RouteGenerator

        return RouteGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
