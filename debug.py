from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) * sin(d_lat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) * sin(d_lon / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def calculate_area(min_lat, min_lon, max_lat, max_lon):
    width = haversine(min_lat, min_lon, min_lat, max_lon)
    height = haversine(min_lat, min_lon, max_lat, min_lon)
    return (width/1000) * (height/1000)

# min_lat = 40.7000000
# min_lon = -74.0199000
# max_lat = 40.7886000
# max_lon = -73.9400000

min_lat = 40.4250000
min_lon = -80.0350000
max_lat = 40.4600000
max_lon = -79.9300000

area = calculate_area(min_lat, min_lon, max_lat, max_lon)
print("The area within the boundary is approximately:", round(area, 2), "square kilometers")

width = 224
zoom = 18
dx = ((20037508.34*2*(width/2)))/(width*(2**(zoom)))
print("geographic width is approximately:", dx*2, "meters")






