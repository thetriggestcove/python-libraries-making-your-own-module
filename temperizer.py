"""An example library for converting temperatures."""

def convert_f_to_c(temperature_f):
    """Convert Fahrenheit to Celsius."""
    temperature_c = (temperature_f - 32) * (5/9)
    return temperature_c


def convert_c_to_f(temperature_c):
    """Convert Celsius to Fahrenheit"""
    temperature_f = (temperature_c * 9/5) + 32
    return temperature_f

def convert_c_to_k(temperature_c):
    """Convert Celsius to Kelvin"""
    return temperature_c + 273.15


def convert_f_to_k(temperature_f):
    """Convert Fahrenheit to Kelvin"""
    return convert_f_to_c(temperature_f) + 273.15

def convert_k_to_c(temperature_k):
    """Convert Kelvin to Celsius."""
    return temperature_k - 273.15


def convert_k_to_f(temperature_k):
    """Convert Kelvin to Fahrenheit"""
    return convert_c_to_f(temperature_k - 273.15)