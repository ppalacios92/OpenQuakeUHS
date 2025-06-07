import folium

def plot_uhs_location_map(lat, lon, zoom=8):
    """
    Crea un mapa interactivo con un pin en la ubicación dada.

    Parámetros:
    ------------
    lat : float
        Latitud del sitio.
    lon : float
        Longitud del sitio.
    zoom : int
        Nivel de zoom inicial (opcional).
    """
    # Crear mapa centrado en la ubicación
    mapa = folium.Map(location=[lat, lon], zoom_start=zoom)

    # Agregar marcador (pin)
    folium.Marker(
        location=[lat, lon],
        popup=f"UHS Site\nLat: {lat:.3f}, Lon: {lon:.3f}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(mapa)

    # Mostrar mapa
    return mapa
"""
Map Utilities for UHS Site Visualization
Author: Ing. Patricio Palacios Msc.
Date: June 7, 2025

Description:
------------
This module provides mapping tools using `folium` to visualize the location of
Uniform Hazard Spectrum (UHS) sites based on their latitude and longitude.
"""

import folium

def plot_uhs_location_map(lat, lon, zoom=8):
    """
    Creates an interactive map with a pin showing the location of the UHS site.

    Parameters:
    -----------
    lat : float
        Latitude of the site.
    lon : float
        Longitude of the site.
    zoom : int
        Initial zoom level for the map.

    Returns:
    --------
    folium.Map
        A Folium map object with the site marker.
    """
    # Create map centered at the given coordinates
    mapa = folium.Map(location=[lat, lon], zoom_start=zoom)

    # Add marker (pin)
    folium.Marker(
        location=[lat, lon],
        popup=f"UHS Site\nLat: {lat:.3f}, Lon: {lon:.3f}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(mapa)

    return mapa
