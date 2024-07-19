import os
import pandas as pd
import folium
import random
import scipy.io as sio
random.seed(42)

def generate_random_color(existing_colors):
    # Generate a random RGB color
    color = "#{:06x}".format(random.randint(0, 256**3 - 1))
    
    # Check if the color already exists in the list
    if color in existing_colors:
        # If it does, recursively generate a new color
        return generate_random_color(existing_colors)
    else:
        # If it's unique, return the color
        return color

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'datasets')
    area = 'wallstreet5k'
    data = pd.read_csv(os.path.join(data_path, 'csv', area+'U.csv'), sep=',', header=None)
    info = data.values
    mean_lat = data.iloc[:, 1].mean()
    mean_lon = data.iloc[:, 2].mean()
    map_center = [mean_lat, mean_lon]
    mymap = folium.Map(location=map_center, zoom_start=13)

    route_colors = []
    for _ in range(50):
        color = generate_random_color(route_colors)
        route_colors.append(color)

    mat_data = sio.loadmat(os.path.join(data_path, area+'_routes_500.mat'))
    routes = mat_data['test_route']
    routes = routes.tolist()
    routes = random.sample(routes, 50)
    # Iterate over each route
    for route, color in zip(routes, route_colors):
        # Create a list to store the coordinates of this route
        route_coordinates = []
        for loc_id in route:
            # Extract latitude and longitude from location data
            lat, lon = info[loc_id-1][1], info[loc_id-1][2]
            # Add coordinate pair to the route coordinates list
            route_coordinates.append((lat, lon))
            # folium.Circle(location=[lat,lon], radius=1, color=color, fill=True, fill_color=color).add_to(mymap)
       
        # Plot the route on the map with a unique color
        folium.PolyLine(locations=route_coordinates, color=color, weight=5).add_to(mymap)

    # # Save the map to an HTML file
    mymap.save(area+'50_rd.html')    

