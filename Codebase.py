import csv
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import numpy as np
import random
from tabulate import tabulate
import matplotlib.pyplot as plt
import math
import pulp
from scipy.sparse import csr_matrix, save_npz
import geopandas as gpd
from shapely.geometry import Point

# Code Sinppet for loading the data to a CSV file

data = [
    ("CK 1", "1755 SW Jefferson St, Portland, OR"),
    ("CK 2", "4520 SE Belmont St, Portland, OR"),
    ("CK 3", "3101 NE Sandy Blvd, Portland, OR"),
    ("CK 4", "4804 SE Woodstock Blvd, Portland, OR"),
    ("CK 5", "8001 SE 72nd Ave, Portland, OR"),
    ("CK 6", "5205 SE Foster Rd, Portland, OR"),
    ("CK 7", "4617 SE Milwaukie Ave, Portland, OR"),
    ("CK 8", "6011 SE 72nd Ave, Portland, OR"),
    ("CK 9", "2117 NE Oregon St, Portland, OR"),
    ("CK 10", "4550 S Macadam Ave, Portland, OR"),
    ("CK 11", "322 SE 82nd Ave, Portland, OR"),
    ("CK 12", "2005 SE 82nd Avenue, Portland, OR"),
    ("CK 13", "6200 S Virginia Ave, Portland, OR"),
    ("CK 14", "3934 NE Martin Luther King Jr Blvd, Portland, OR"),
    ("CK 15", "10568 SE Washington St, Portland, OR"),
    ("CK 16", "1125 SE Division St, Portland, OR"),
    ("CK 17", "4152 NE Sandy Blvd, Portland, OR"),
    ("CK 18", "1500 SW 1st Ave, Portland, OR"),
    ("CK 19", "5115 NE Sandy Blvd, Portland, OR"),
    ("CK 20", "16116 NE Halsey St, Portland, OR"),
    ("CK 21", "16088 NE Sandy Blvd, Portland, OR"),
    ("CK 22", "12404 NE Halsey St, Portland, OR"),
    ("CK 23", "1606 NE 6th Ave, Portland, OR"),
    ("CK 24", "1405 NW Johnson St, Portland, OR"),
    ("CK 25", "2005 NE Alberta St, Portland, OR")
]

csv_file_path = 'BusinessLocations.csv'

with open(csv_file_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Index', 'Street Address'])
    csv_writer.writerows(data)

print(f'Data has been saved to {csv_file_path}')


# Code snippet for generating and storing coordinates {as a tuple (lat,long)} and ZIP code of cloud kitchens selected.

# Loading CSV file of collected cloud kitchen location data 
df = pd.read_csv(csv_file_path)

# Defining geocode function to get lat and long
def geocode_location(location):
    try:
        location_info = geolocator.geocode(location, timeout=3)
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        return geocode_location(location)

# Defining reverse geocode function to get ZIP from lat and long
def reverse_geocode(lat, lon):
    try:
        location_info = geolocator.reverse((lat, lon), exactly_one=True)
        if location_info:
            return location_info.raw.get('address', {}).get('postcode')
        else:
            return None
    except GeocoderTimedOut:
        return reverse_geocode(lat, lon)

geolocator = Nominatim(user_agent="location_geocoder")

# Creating new columns for lat and long
df["Latitude"] = None
df["Longitude"] = None

# Geocoding each location
for index, row in df.iterrows():
    location = row["Street Address"]
    latitude, longitude = geocode_location(location)
    df.at[index, "Latitude"] = latitude
    df.at[index, "Longitude"] = longitude
    time.sleep(1)  # Sleep for 1 second to avoid overloading

# Creating a new column for coordinates (as a tuple of lat and lon)
df["Coordinates"] = df.apply(lambda row: (row["Latitude"], row["Longitude"]), axis=1)

# Creating a new column for ZIP code
df["ZIP Code"] = df.apply(lambda row: reverse_geocode(row["Latitude"], row["Longitude"]), axis=1)

# Saving the data with coordinates and ZIP code to a new CSV file
output_csv_file_path = "geocoded_locations.csv"
df.drop(["Latitude", "Longitude"], axis=1, inplace=True)
df.to_csv(output_csv_file_path, index=False)

# Code snippet for randomly generating and storing coordinates {as a tuple (lat,long)}, ZIP code and address of 50 service delivery locations.

random.seed(42)
np.random.seed(42)

# Latitude and longitude range for our geographical area
min_latitude = 45.45
max_latitude = 45.57
min_longitude = -122.48
max_longitude = -122.7

num_service_stations = 50
service_stations = []

geolocator = Nominatim(user_agent="reverse_geocoder")

for _ in range(num_service_stations):
    latitude = np.random.uniform(min_latitude, max_latitude)
    longitude = np.random.uniform(min_longitude, max_longitude)

    # Reverse geocode using lat and long to obtain address and ZIP
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    
    if location:
        address = location.address
        zip_code = location.raw.get("address", {}).get("postcode", None)
    else:
        address = None
        zip_code = None

    # Combine lat and long as a tuple
    coordinates = (latitude, longitude)

    service_stations.append([f"SDL {len(service_stations) + 1}", address, coordinates, zip_code])

# Loading data from the cloud kitchen locations CSV file
existing_data = pd.read_csv("geocoded_locations.csv")

# Converting the loaded data to a list of lists
existing_data_list = existing_data.values.tolist()

# Combine the existing data with the new service delivery location data generated randomly
combined_data = existing_data_list + service_stations

headers = ["Index", "Street Address", "Coordinates", "Zip Code"]

table = tabulate(combined_data, headers, tablefmt="simple")

print(table)

# Saving the table
with open("Locations.txt", "w") as txt_file:
    txt_file.write(table)

# Code snippet for generating and storing lat and long (not as a tuple) for cloud kitchen locations selected.

df = pd.read_csv(csv_file_path)

geolocator = Nominatim(user_agent="location_geocoder")

# Creating new columns for latitude and longitude
df["Latitude"] = None
df["Longitude"] = None

# Geocoding each location
for index, row in df.iterrows():
    location = row["Street Address"]
    latitude, longitude = geocode_location(location)
    df.at[index, "Latitude"] = latitude
    df.at[index, "Longitude"] = longitude
    time.sleep(1)  # Sleep for 1 second to avoid overloading

# Saving the data with coordinates to a new CSV file
output_csv_file_path = "geocoded_locations_lat_lon.csv"
df.drop(["Street Address"], axis=1, inplace=True)
df.to_csv(output_csv_file_path, index=False)

# Code snippet for generating and storing lat and long (not as a tuple) for service delivery locations randomly generated.

random.seed(42)
np.random.seed(42)

num_service_stations = 50
service_stations = []

for _ in range(num_service_stations):
    latitude = np.random.uniform(min_latitude, max_latitude)
    longitude = np.random.uniform(min_longitude, max_longitude)

    service_stations.append([f"SDL {len(service_stations) + 1}", latitude, longitude])

# Creating a Dataframe from the randomly generated data
df = pd.DataFrame(service_stations, columns=["Index", "Latitude", "Longitude"])

# Saving the Dataframe to a CSV file
output_csv_file = "geocoded_service_stations_lat_lon.csv"
df.to_csv(output_csv_file, index=False)

print(f"Generated service station data has been saved to '{output_csv_file}'.")

# Code snippet for plotting cloud kitchen and service delivery locations with the margins in axes given.

# Loading the data from ck locations file
file1_path = "geocoded_locations_lat_lon.csv"  # Update with the path to your first CSV file
df1 = pd.read_csv(file1_path)

# Load the data from the sdl locations file
file2_path = "geocoded_service_stations_lat_lon.csv"  # Update with the path to your second CSV file
df2 = pd.read_csv(file2_path)

# Extract lat and long columns from both files
latitudes1 = df1["Latitude"]
longitudes1 = df1["Longitude"]
latitudes2 = df2["Latitude"]
longitudes2 = df2["Longitude"]

# Calculating the boundaries for the plot with given margin
all_latitudes = pd.concat([latitudes1, latitudes2])
all_longitudes = pd.concat([longitudes1, longitudes2])

margin = 2.5 / 69  # 1 degree of latitude is approximately 69 miles
min_lat = all_latitudes.min() - margin
max_lat = all_latitudes.max() + margin
min_lon = all_longitudes.min() - margin
max_lon = all_longitudes.max() + margin

# Creating a scatter plot of the locations from both files within the plot region
plt.figure(figsize=(10, 8))
plt.scatter(longitudes1, latitudes1, marker='o', color='blue', label='Cloud Kitchen Locations')
plt.scatter(longitudes2, latitudes2, marker='*', color='red', label='Service Delivery Locations')
plt.xlim(min_lon, max_lon)
plt.ylim(min_lat, max_lat)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations on Map')

plt.legend()

# Saving the plot as a JPEG file
plot_file_path = "Locations.jpg"  # Update with the desired output path
plt.savefig(plot_file_path, format='jpeg')

plt.show()

# Code snippet for generating and storing the distance matrix between cloud kitchen and service delivery locations.

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculating the great-circle distance between two points on the Earth's surface.
    Source: https://en.wikipedia.org/wiki/Haversine_formula
    lat1: Latitude of the first point (in degrees)
    lon1: Longitude of the first point (in degrees)
    lat2: Latitude of the second point (in degrees)
    lon2: Longitude of the second point (in degrees)
    return: Distance between the two points in miles
    """
    # Converting degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius_of_earth_miles = 3959  # Mean radius of the Earth in miles
    distance = radius_of_earth_miles * c

    return distance

def read_coordinates_from_csv(file_path, lat_column, lon_column):
    coordinates = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            latitude = float(row[lat_column])
            longitude = float(row[lon_column])
            coordinates.append((latitude, longitude))
    return coordinates

def calculate_distance_matrix(cloud_kitchens, service_locations):
    num_kitchens = len(cloud_kitchens)
    num_locations = len(service_locations)
    distance_matrix = [[0] * num_locations for _ in range(num_kitchens)]

    for i in range(num_kitchens):
        for j in range(num_locations):
            lat1, lon1 = cloud_kitchens[i]
            lat2, lon2 = service_locations[j]
            distance_matrix[i][j] = haversine_distance(lat1, lon1, lat2, lon2)

    return distance_matrix

# Defining file paths and column names for cloud kitchens and service locations CSV files
cloud_kitchens_csv = "geocoded_locations_lat_lon.csv"
service_locations_csv = "geocoded_service_stations_lat_lon.csv"
lat_column_kitchens = "Latitude"  
lon_column_kitchens = "Longitude"  
lat_column_locations = "Latitude"  
lon_column_locations = "Longitude"  

# Reading coordinates from CSV files
cloud_kitchens = read_coordinates_from_csv(cloud_kitchens_csv, lat_column_kitchens, lon_column_kitchens)
service_locations = read_coordinates_from_csv(service_locations_csv, lat_column_locations, lon_column_locations)

# Calculating the distance matrix
distance_matrix = calculate_distance_matrix(cloud_kitchens, service_locations)

# Saving the distance matrix to a CSV file
output_csv = "Distances.csv"
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)

    header = [""] + [f"SDL {i+1}" for i in range(len(service_locations))]
    writer.writerow(header)

    for i, row in enumerate(distance_matrix):
        writer.writerow([f"CK {i+1}"] + row)

print(f"Distance matrix saved to {output_csv}")

# Code snippet for solving and storing the given linear programming model.

# Reading the distance matrix from a CSV file
distance_matrix = pd.read_csv('Distances.csv', index_col=0)

# Defining indices
I = list(distance_matrix.index)
J = list(distance_matrix.columns)

index_to_integer = {index: i for i, index in enumerate(I)}

# Creating the LP problem
assignment_problem = pulp.LpProblem("AssignmentProblem", pulp.LpMinimize)

z = pulp.LpVariable.dicts("z", (I, J), cat=pulp.LpBinary)

# Defining the objective function to minimize total distance traveled
assignment_problem += pulp.lpSum(distance_matrix.loc[index][j] * z[index][j] for index in I for j in J)

# Adding constraints

# Each cloud kitchen delivers to exactly two service stations
for i in I:
    assignment_problem += pulp.lpSum(z[i][j] for j in J) == 2

# Each service station is delivered by exactly one cloud kitchen
for j in J:
    assignment_problem += pulp.lpSum(z[i][j] for i in I) == 1

assignment_problem.solve()

assignment_problem.writeMPS("AP.mps")

# Checking the status of the solution
if pulp.LpStatus[assignment_problem.status] == "Optimal":
    print("Optimal solution found.")
    print("Assignments:")
    for i in I:
        for j in J:
            if pulp.value(z[i][j]) == 1:
                print(f"{i} delivers to {j}")
else:
    print("No optimal solution found.")

total_distance = pulp.value(assignment_problem.objective)
print(f"Total distance traveled: {total_distance}")

# Code snippet for generating and storing the assignments in sparse matrix format.

assignment_matrix = np.array([[z[i][j].value() for j in J] for i in I])
csr_assignment_matrix = csr_matrix(assignment_matrix)

dense_matrix = csr_assignment_matrix.toarray()

filename = "Solution"

# Saving both CSR and dense assignment matrices in the same .npz file
save_npz(filename, csr_assignment_matrix)
np.savez(filename, dense_matrix=dense_matrix)

print(csr_assignment_matrix)
print(dense_matrix)

# Code snippet for generating and storing the assignments in the table format.

# Create an empty list to store the assignments
assignments = []

# Iterate through the indices and decision variables
for i in I:
    for j in J:
        if pulp.value(z[i][j]) == 1:
            # If z[i][j] is 1, it means cloud kitchen i is assigned to service station j
            assignments.append((i, j))

od_table = []
for i, j in assignments:
    od_table.append([i, j, distance_matrix.loc[i, j]])
    
headers = ["Cloud Kitchen Index (Origin)", "Service Station Index (Destination)", "Distance (miles)"]
table_txt = tabulate(od_table, headers, tablefmt="simple")

with open('OD.txt', 'w') as txt_file:
    txt_file.write(table_txt)

print("Assignments (zij = 1):")
print(table_txt)
print("Table saved to OD.txt")

# Code snippet for generating and storing the frequency percentage chart.

distances = []
for i in I:
    for j in J:
        if pulp.value(z[i][j]) == 1:
            distance = distance_matrix.loc[i][j]
            distances.append(distance)

# Defining distance ranges
short_range = (0, 3)
medium_range = (3, 6)
long_range = (6, float('inf'))

short_count = 0
medium_count = 0
long_count = 0

# Calculating frequencies for each range
for distance in distances:
    if short_range[0] <= distance < short_range[1]:
        short_count += 1
    elif medium_range[0] <= distance < medium_range[1]:
        medium_count += 1
    else:
        long_count += 1

# Calculating total assignments
total_assignments = short_count + medium_count + long_count

# Calculating frequencies as percentages
short_freq = (short_count / total_assignments) * 100
medium_freq = (medium_count / total_assignments) * 100
long_freq = (long_count / total_assignments) * 100

# Creating the bar chart for frequency graph
distance_ranges = ['< 3 miles', '3-6 miles', '> 6 miles']
frequencies = [short_freq, medium_freq, long_freq]

for i, v in enumerate(frequencies):
    plt.text(i, v, f"{v:.2f}%", ha='center', va='bottom')

plt.bar(distance_ranges, frequencies)
plt.xlabel('Distance Ranges')
plt.ylabel('Frequency (%)')
plt.title('Frequency of Assignments')

# Saving the plot as a JPEG file
plot_file_path = "Frequency.jpg"
plt.savefig(plot_file_path, format='jpeg')
plt.show()

# Code snippet for generating and storing the solution mapping chart.

# Reading the cloud kitchen and service station data with their coordinates
cloud_kitchen_locations = pd.read_csv('geocoded_locations_lat_lon.csv')
service_station_locations = pd.read_csv('geocoded_service_stations_lat_lon.csv')

# Creating GeoDataFrames for the locations
cloud_kitchen_geometry = [Point(xy) for xy in zip(cloud_kitchen_locations['Longitude'], cloud_kitchen_locations['Latitude'])]
service_station_geometry = [Point(xy) for xy in zip(service_station_locations['Longitude'], service_station_locations['Latitude'])]

cloud_kitchen_gdf = gpd.GeoDataFrame(cloud_kitchen_locations, geometry=cloud_kitchen_geometry)
service_station_gdf = gpd.GeoDataFrame(service_station_locations, geometry=service_station_geometry)

fig, ax = plt.subplots(figsize=(10, 10))

# Plotting cloud kitchens
cloud_kitchen_gdf.plot(ax=ax, color='blue', markersize=50, label='Cloud Kitchens')

# Plotting service stations
service_station_gdf.plot(ax=ax, color='green', markersize=50, label='Service Stations')

# Plot the assignments on the map
for i in I:
    for j in J:
        if pulp.value(z[i][j]) == 1:
            start_location = cloud_kitchen_gdf.iloc[index_to_integer[i]].geometry
            end_location = service_station_gdf.iloc[J.index(j)].geometry
            plt.plot([start_location.x, end_location.x], [start_location.y, end_location.y], color='red')
            
for i, row in cloud_kitchen_gdf.iterrows():
    ax.text(row['Longitude'], row['Latitude'], row['Index'], fontsize=5, ha='right')

# Add data labels for service stations
for i, row in service_station_gdf.iterrows():
    ax.text(row['Longitude'], row['Latitude'], row['Index'], fontsize=5, ha='right')

# Add legends for markers
ax.legend(loc='upper right', fontsize=10, markerscale=1)

ax.set_title("Assignment Solution on Map", fontsize=12)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)

# Saving the chart as a JPEG file
plt.savefig('Solution.jpg', format='jpg')

# Display the plot
plt.show()

