# model/__init__.py
from .forecaster import (
    train_all_cities,
    forecast_city,
    forecast_all_cities,
    load_all_models,
    get_forecast_with_history,
)

__all__ = [
    "train_all_cities",
    "forecast_city",
    "forecast_all_cities",
    "load_all_models",
    "get_forecast_with_history",
]