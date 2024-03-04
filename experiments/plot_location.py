import os
import pandas as pd
import folium

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'datasets')
    loc_list = [
        {"loc": [40.770402,-73.964298], "color": "red"},
        {"loc": [40.770318,-73.964359], "color": "blue"},
        {"loc": [40.770404,-73.964264], "color": "blue"},
        {"loc": [40.770444,-73.964268], "color": "blue"},
        {"loc": [40.770458,-73.964388], "color": "blue"}
        ]

    # Create a map centered on a location
    map_center = [40.770402,-73.964298]
    mymap = folium.Map(location=map_center, zoom_start=12)

    # Add markers for each location with different colors
    for location in loc_list:
        coordinates = location["loc"]
        color = location["color"]
        folium.Marker(location=coordinates, icon=folium.Icon(color=color)).add_to(mymap)

    # Save the map as an HTML file
    mymap.save("link.html")

