import os
import pandas as pd
import folium

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'datasets')
    csv_list = [
        {"file": "trainstreetlearnU.csv", "color": "black"},
        {"file": "cmu5kU.csv", "color": "black"},
        {"file": "wallstreet5kU.csv", "color": "red"},
        {"file": "hudsonriver5kU.csv", "color": "blue"},
        {"file": "unionsquare5kU.csv", "color": "green"}
        ]

    # Create a map centered on a location
    map_center = [40.7424436,-73.9816155]
    mymap = folium.Map(location=map_center, zoom_start=12)

    # Add markers for each location with different colors
    for csv in csv_list:
        file = csv["file"]
        color = csv["color"]
        data = pd.read_csv(os.path.join(data_path, 'csv', file), sep=',', header=None)
        info = data.values

        for i in range(len(info)):
            coordinates = [info[i][1],info[i][2]]
            city = info[i][4]
            if city == 'manhattan':
                folium.Circle(location=coordinates, radius=0.1, color=color, fill=True, fill_color=color).add_to(mymap)

    # Save the map as an HTML file
    mymap.save("manhattan.html")

