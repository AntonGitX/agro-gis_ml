# gis_engine.py
import numpy as np
import pyproj
import requests
from shapely.geometry import shape
from shapely.ops import transform

def analyze_polygon(geom):
    """Аналізує полігон: рахує площу, рельєф та отримує клімат"""
    polygon = shape(geom)
    
    # Розрахунок площі
    project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True).transform
    polygon_utm = transform(project, polygon)
    area_2d = polygon_utm.area / 10000 
    
    # Витягуємо координати
    coords = geom['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    
    # Розрахунок схилу
    lat_str = ",".join(map(str, lats))
    lon_str = ",".join(map(str, lons))
    elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
    
    calculated_slope = 0.0
    try:
        elev_response = requests.get(elev_url).json()
        elevations = elev_response.get('elevation', [])
        
        if elevations and len(elevations) > 1:
            min_e, max_e = min(elevations), max(elevations)
            min_idx, max_idx = elevations.index(min_e), elevations.index(max_e)
            
            geod = pyproj.Geod(ellps="WGS84")
            _, _, dist = geod.inv(lons[min_idx], lats[min_idx], lons[max_idx], lats[max_idx])
            
            if dist > 0:
                calculated_slope = np.degrees(np.arctan((max_e - min_e) / dist))
    except Exception:
        pass
        
    calculated_slope = round(calculated_slope, 2)
    area_3d = area_2d / np.cos(np.radians(calculated_slope))
    
    # Погода (заглушка або реальне API)
    lon, lat = polygon.centroid.x, polygon.centroid.y
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_req = requests.get(url).json()
        temp_val = weather_req['current_weather']['temperature']
    except:
        temp_val = 20.0 
    
    return area_2d, area_3d, calculated_slope, temp_val, 70.0, 80.0