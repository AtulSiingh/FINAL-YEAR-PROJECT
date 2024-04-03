import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import scipy.cluster.vq
import folium
import streamlit as st
from sklearn import preprocessing
import streamlit_folium as sf
from math import radians, sin, cos, sqrt, atan2
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
import random
from folium import plugins
import requests
from prettytable import PrettyTable



def get_train_schedule( from_station_code, to_station_code, date_of_journey):
    url = "https://irctc1.p.rapidapi.com/api/v3/trainBetweenStations"

    querystring = {
        "fromStationCode": from_station_code,
        "toStationCode": to_station_code,
        "dateOfJourney": date_of_journey,
    }

    headers = {
	"X-RapidAPI-Key": "5df875fbadmshecf65e9f6e6f705p1d410fjsn7fa0573fb48b",
	"X-RapidAPI-Host": "irctc1.p.rapidapi.com"
}

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching train schedule. Please try again later.")
        return None


def select_csv_by_district_linear_search(user_district):

    csv_files = [file for file in os.listdir() if file.endswith('.csv')]

    for file in csv_files:
        file_district = file.split('.')[0].replace('_', ' ')
        if file_district.lower() == user_district.lower():
            file_path = os.path.join(os.getcwd(), file)

            try:
                # Specify the encoding (e.g., ISO-8859-1) based on your CSV file
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
                return df
            except UnicodeDecodeError:
                st.error(f"Error decoding CSV file for {user_district}. Try specifying a different encoding.")
                return None

    return None

def find_optimal_clusters(df):
    features = df[['Latitude', 'Longitude']]
    selected_features = features[['Latitude', 'Longitude']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(selected_features)

    distortions = []
    max_k = 10

    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_features)
        distortions.append(kmeans.inertia_)

    # find the lowest derivative of K
    k = [i * 100 for i in np.diff(distortions, 2)].index(min([i * 100 for i in np.diff(distortions, 2)]))

    return k

def get_lat_long(locality, city):
    address = f"{locality}, {city}"
    geolocator = Nominatim(user_agent="my_geocoder")

    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            print(f"Location not found for {address}")
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def calc_all_dist(df,lat,long):

  data=df
  loc1=[lat, long]
  data["Distance"] = data.apply(lambda row: geodesic((row["Latitude"], row["Longitude"]), loc1).km, axis=1)
  df=data
  return df

def dist(lat1, lon1, lat2, lon2):
    
  lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

  # Haversine formula
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))

  # Radius of the Earth in kilometers (change to 3959.0 miles for miles)
  radius_earth = 6371.0

  # Calculate the distance
  distance = radius_earth * c

  return distance

def calculate_amen_count(final_row, new_df):
    count = 0
    for _, new_row in new_df.iterrows():
        if (dist(final_row['Latitude'], final_row['Longitude'], new_row['Latitude'], new_row['Longitude']) <= 1):
            count += 1
    return count

def k_means(df_final, k):

  model = KMeans(n_clusters = k, init = 'k-means++')
  X = df_final[['Latitude', 'Longitude']]
  dtf_X = X.copy()

  dtf_X["cluster"] = model.fit_predict(X)
  ## find real centroids
  closest, distances = scipy.cluster.vq.vq(model.cluster_centers_,
                      dtf_X.drop("cluster", axis=1).values)
  dtf_X["centroids"] = 0
  for i in closest:
      dtf_X["centroids"].iloc[i] = 1
  ## add clustering info to the original dataset
  df_final[["cluster","centroids"]] = dtf_X[["cluster","centroids"]]
  return df_final

def create_interactive_map(data,user_coordinates,area):
    # Create a Folium map
    map_center=user_coordinates
    mymap = folium.Map(location=map_center, zoom_start=14)
    plugins.Fullscreen().add_to(mymap)
    locality_centre = gpd.GeoDataFrame(geometry=[Point(data['Longitude'].iloc[0], data['Latitude'].iloc[0])],
                                      crs="EPSG:4326")

    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(mymap)
    dis = area*0.01
    buffer_distance = dis  # This is a small buffer for demonstration purposes
    locality_boundary = locality_centre.buffer(buffer_distance)

    # Add the GeoJSON boundary layer to the map
    folium.GeoJson(locality_boundary.__geo_interface__).add_to(mymap)

    # Add markers to the map
    for index, row in data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['Name']} - Count: {row['amen_count']}",
        ).add_to(marker_cluster)

    # Convert Folium map to Streamlit map
    sf.folium_static(mymap)

def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))






def radius(df,dis):
  new_df = df[df['Distance'] <= dis+1]
  return new_df

def inject_custom_css():
    with open('E:\MAJOR_PROJECT\style.css') as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)





#try and catch block



def main():
    inject_custom_css()
    station = pd.read_csv('E:\MAJOR_PROJECT\stations.csv')
    station.dropna(inplace=True)
    st.title("Place Finder")

    try:
        l = [ '','Lucknow', 'Kanpur', 'Gurgaon', 'Pune', 'Banglore', 'Chennai', 'Hyderabad']
        district = st.selectbox("Enter your city Name", l)
        df = select_csv_by_district_linear_search(district)
        str1 = district+'2'
        area_lat_long = select_csv_by_district_linear_search(str1)
        if df is not None:
            df.dropna(inplace=True)
        lowest_derivative = 0

        if df is not None:
            lowest_derivative = find_optimal_clusters(df)
        else:
            return

        lat, long = (0, 0)
        if lowest_derivative != 0:
            address = st.selectbox('Enter the Locality', area_lat_long['Area'])
            lat = area_lat_long[area_lat_long['Area'] == address]['Latitude'].values[0]
            long = area_lat_long[area_lat_long['Area'] == address]['Longitude'].values[0]

            if lat is not None and long is not None:
                pass
            else:
                st.warning("Failed to get coordinates.")
                return

        selected_category = []
        st.subheader("User Preferences")
        options1 = st.selectbox('Living or Staying', ('Hostel', 'Hotel'))
        options2 = st.selectbox('Food Services or Dining', ('Restaurant', 'Tiffin'))
        exclude_categories = ['Hostel', 'Hotel', 'Restaurant', 'Tiffin']
        filtered_categories = [category for category in df['Category'].unique() if category not in exclude_categories]
        options3 = st.multiselect("Select Other Facilities:", filtered_categories)
        selected_category.append(options1)
        selected_category.append(options2)

        for i in options3:
            selected_category.append(i)
        house = selected_category.pop(0)

        df = calc_all_dist(df, lat, long)
        options = ['4km', '3km', '2km', '1km']
        selected_option = st.selectbox("Set the range", options)
        area = int(selected_option[:-2])
        df = radius(df, area)

        if len(selected_category):
            new_df = df[(df['Category'] == house) & (df['Distance'] <= area)]
            filtered_df = df[df['Category'].isin(selected_category)]
            new_df['amen_count'] = new_df.apply(lambda row: calculate_amen_count(row, filtered_df), axis=1)

            try:
                new_df = k_means(new_df, lowest_derivative)
                folium_map = create_interactive_map(new_df, (lat, long), area)
                st.subheader("Top location where you can live")
                top_result = new_df['Area'].iloc[0]
                st.write(top_result)
                
            except Exception as e:
                st.write("Error while generating the map. Please try a different combination.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    from_station_code = st.selectbox("Enter your current city:", station['Station_Name'])
    from_station_code = station.loc[station['Station_Name'] == from_station_code, 'Code'].iloc[0]
    temp = calc_all_dist(station, lat, long)
    temp = temp.sort_values('Distance')
    temp1 = str(temp["Station_Name"].iloc[0])
    # st.write(temp1)
    to_station_code = station.loc[station['Station_Name'] == temp1, 'Code'].iloc[0]
    date_of_journey = st.date_input("Enter the Date of Journey (format: YYYY-MM-DD):")
    # st.write(from_station_code, to_station_code)

    if st.button("Get Train Schedule"):
        if not from_station_code or not to_station_code or not date_of_journey:
            st.warning("Please enter your Details care fully")
        else:
            train_schedule = get_train_schedule(from_station_code, to_station_code, date_of_journey)
        if train_schedule:
            data_list = train_schedule["data"]
            df4 = pd.DataFrame(data_list)
            table = PrettyTable()
            table.field_names = ["Train Number", "Train Name", "Run Days", "Source", "Destination", "Start Time", "Reach Time"]
            for data in data_list:
                run_days = ", ".join(data["run_days"])
                table.add_row([data["train_number"], data["train_name"], run_days, data["train_src"], data["train_dstn"],
                               data["from_std"], data["to_std"]])
            st.write(table)


if __name__ == "__main__":
    main()