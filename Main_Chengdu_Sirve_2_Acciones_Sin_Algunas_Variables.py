from __future__ import division, print_function

import copy
import csv
import heapq
import itertools
import math
import os
import random
import sys
from builtins import range
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
#os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd


#Clase con data histórica
class environment():
    def __init__(self, file_path, file_path_velocities_morning, file_path_velocities_afternoon, clients, horizon_start_time, horizon_end_time):
        '''
        Input: archivo mega_city
        Output: dataframe con los datos
        '''
        self.horizon_start_time = horizon_start_time
        self.horizon_end_time = horizon_end_time

        self.process_all_data()

        #Data completa
        self.df = pd.read_csv(file_path)

        # Concatenamos las velocidades
        self.df_morning = pd.read_csv(file_path_velocities_morning)
        self.df_afternoon = pd.read_csv(file_path_velocities_afternoon)
        self.data_velocity = pd.concat([self.df_morning, self.df_afternoon], ignore_index=True)

        # Creamos un dataframe con toda la data unida en cada link
        self.full_data = pd.merge(self.df, self.data_velocity, on='Link')

        #Transformamos las distancias a km y las horas a minutos
        self.full_data["Length"] = self.full_data["Length"]/1000

        #Agregamos travel time al dataframe
        #self.full_data['Travel_Time'] = (self.full_data['Length']) / (self.full_data['Speed'])

        self.convert_time_to_minutes()

        #Filtramos el dataframe con tal de guardar los datos de 300 o más, dado que los otros no nos sirven

        self.full_data = self.full_data[self.full_data['Minute_start'] >= 300]


        #self.process_all_data()

        self.merged = pd.merge(self.full_data, self.aggregated_data, on=['Link', 'Period'], how='left', suffixes=('', '_df2'))


        
        self.merged = self.merged.drop("Speed", axis=1)

        self.merged.rename(columns={"speed": "Speed"}, inplace=True)

        self.full_data = self.merged

        self.full_data['Travel_Time'] = (self.full_data['Length']) / (self.full_data['Speed'])
        
        self.df_filtered_minute_start = self.full_data[self.full_data['Minute_start'] >= 300]
    
            
        # Calculamos la velocidad promedio por cada link
        average_speed = self.df_filtered_minute_start.groupby('Link')['Speed'].mean().reset_index()

        # Obtenemos un solo valor de latitud y longitud de inicio y fin por cada link, incluyendo Node_Start
        coords_start = self.df_filtered_minute_start.drop_duplicates(subset='Link')[['Link', 'Node_Start', 'Latitude_Start', 'Longitude_Start']]
        coords_end = self.df_filtered_minute_start.drop_duplicates(subset='Link')[['Link', 'Node_End', 'Latitude_End', 'Longitude_End', "Length"]]

        # Fusionamos los DataFrames para crear el resultado final
        self.data_average = average_speed.merge(coords_start, on='Link')
        self.data_average = self.data_average.merge(coords_end, on='Link')
        self.data_average['Travel_Time'] = (self.data_average['Length']) / (self.data_average['Speed'])

        #Creamos un df que mantenga información de cada nodo
        self.node_start_dataframe = self.df.drop_duplicates(subset='Node_Start', keep='first').copy()

        #Creamos el grafo.
        #self.read_network()

        #Guardamos los clientes
        self.clients = clients

        #Creamos una data con df valores unicos para facilitar algunos calculos
        self.df = self.df.drop_duplicates(subset=['Node_Start'], keep='first')


        #Generamos data filtrada, de tal forma de, guardar todos los travel times
        self.filtered_data_x = self.full_data.loc[
            (self.full_data['Node_Start'] == 0) &
            (self.full_data['Node_End'] ==48)]

        #Guardamos los travel_times
        self.travel_times = self.filtered_data_x['Minute_start'].tolist()

        #Todos los nodos.
        self.node_list = self.node_start_dataframe['Node_Start'].tolist()


        #Generamos la matriz de shortest path
        #self.get_shortest_path_times_from_all_nodes_to_all_clients()
        del self.df_morning, self.df_afternoon, self.data_velocity
        del self.all_data, self.aggregated_data
        del self.merged, self.df_filtered_minute_start

        # Si no los necesitas luego:
        del self.filtered_data_x, self.travel_times, self.node_start_dataframe


    def read_network(self):
        '''
        Output: Un grafo con los pesos de los arcos respectivo a las velocidades media y distancias.
        '''
        # Se crea el grafo
        self.G = nx.DiGraph()
        for _, row in self.data_average.iterrows():
            # Convertimos la velocidad de km/h a km/min (dividiendo entre 60)
            distance_km = row['Length']
            velocity_km_mins = row['Speed']
            #Travel_time queda en km/min
            travel_time = distance_km / velocity_km_mins
            # Añadir o actualizar el arco con el tiempo de viaje
            self.G.add_edge(row['Node_Start'], row['Node_End'], length=row['Length'], weight=travel_time)

    @staticmethod
    def haversine_distance(lon1, lat1, lon2, lat2):
        '''
        Input: longitud y latitud de 2 nodos.
        Output: distancia entre los 2 nodos.
        '''
        # Radio de la Tierra en km.
        R = 6371.0

        # Convertir grados en radianes.
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Diferencias de coordenadas.
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Fórmula Haversine
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance

    def get_shortest_path(self, posicion, target):
        '''
        Input: Un grafo, la posición en que el vehículo se encuentra y una lista de nodos a los que quiero ir (target).
        Output: Una lista con shortest paths para cada nodo target.
        '''
        path = nx.dijkstra_path(self.G, source=posicion, target=target, weight='weight')
        return path

    def lowest_routes_travel_times(self, routes):
        """
        Calcula la distancia total para una lista de rutas y encuentra la ruta con la menor distancia.

        Input:
            routes (lista de lista de nodos): Cada sublista representa una ruta definida por nodos.

        Output:
            Indice de la ruta con menor travel time
        """
        route_distances = []
        for route in routes:
            total_distance = 0
            # Recorremos los nodos de la ruta sumando las distancias de los arcos
            for i in range(len(route) - 1):
                if self.G.has_edge(route[i], route[i+1]):
                    total_distance += self.G[route[i]][route[i+1]]['weight']
                else:
                    total_distance += float('inf')  # Si no existe camino, consideramos distancia infinita
            route_distances.append(total_distance)

        # Encontrar la ruta con la menor distancia
        min_time= min(route_distances)
        minimum_route_index = route_distances.index(min_time)

        return minimum_route_index


    def convert_time_to_minutes(self):
        '''
        Convierte el tiempo de la columna 'period' del dataframe full_data en minutos desde las 3:00 AM representando el minuto 0.
        '''
        def time_to_minutes(time_str):
            # Extraemos la hora y el minuto de inicio
            start_time = time_str.split('-')[0]
            start_hour, start_minute = map(int, start_time.split(':'))

            # Calculamos los minutos desde las 3:00 AM
            minutes_since_start = (start_hour - 3) * 60 + start_minute
            return minutes_since_start
        # Aplicamos la función a la columna 'period'
        self.full_data['Minute_start'] = self.full_data['Period'].apply(time_to_minutes)
        #self.aggregated_data["Minute_start"] = self.aggregated_data["Period"].apply(time_to_minutes)

    def read_network_time(self, specific_period):
        '''
        Input: Un periodo especifico de tiempo en minutos.
        Output: Un grafo con los pesos de los arcos respectivo a las velocidades del tiempo.
        '''
        #Guardamos la data de un periodo especifico
        period_data = self.full_data[self.full_data['Minute_start'] == specific_period]
        # Crear el grafo
        self.G = nx.DiGraph()
        for _, row in period_data.iterrows():
            # Añadir o actualizar el arco con el tiempo de viaje
            if self.G.has_edge(row['Node_Start'], row['Node_End']):
                self.G[row['Node_Start']][row['Node_End']]['weight'] = row["Travel_Time"]
            else:
                self.G.add_edge(row['Node_Start'], row['Node_End'], length=row['Length'], weight= row["Travel_Time"])

    def preprocess_data_average(self):
        """
        Preprocesa el DataFrame data_average para convertirlo en un diccionario que permita
        accesos rápidos a los tiempos de viaje.
        """
        self.travel_time_dict = {}
        for _, row in self.data_average.iterrows():
            self.travel_time_dict[(row['Node_Start'], row['Node_End'])] = (row['Travel_Time'], row["Length"])

    def calculate_time_for_route(self, route):
        travel_time = 0
        length = 0
        for i in range(len(route) - 1):
            Node_Start = route[i]
            Node_End = route[i + 1]
            # Usamos el diccionario preprocesado para acceder al tiempo de viaje
            current_travel_time, current_length = self.travel_time_dict[(Node_Start, Node_End)]

            travel_time += current_travel_time
            length += current_length
        return travel_time, length


    def get_shortest_path_times_from_all_nodes_to_all_clients(self):
        matrix = np.zeros((len(self.clients), len(self.clients)), dtype=int)
        #Para cada nodo
        for node in range(len(self.clients)):
            for client in range(len(self.clients)):
                travel_time = 0
                #Obtenmos la shortest path
                shortest_path = self.get_shortest_path(self.clients[node], self.clients[client])
                #Obtenemos travel time
                travel_time, _ = self.calculate_time_for_route(shortest_path)
                #Adjuntamos el total_travel_time a la matriz
                matrix[node][client] = travel_time

        self.shortest_path_df = pd.DataFrame(matrix, index=self.clients, columns=self.clients)


    def save_df_to_csv(self, seed_number):
        # Guardar el DataFrame en CSV con ; como delimitador
        filename = f"seed_{seed_number}.csv"
        self.shortest_path_df.to_csv(filename, sep=';', index=False, header=False)



    def save_matrix_to_txt(self):
        # Guarda el DataFrame en un archivo de texto
        with open('shortest_path_matrix_250.txt', 'w') as file:
            # Escribe la cabecera con un espacio inicial para desplazarla a la derecha
            header = '\t' + '\t'.join([str(i) for i in self.shortest_path_df.columns])
            file.write(header + '\n')

            # Escribe las filas, comenzando cada una con el índice de la fila seguido de los datos
            for idx, row in self.shortest_path_df.iterrows():
                line = '\t'.join(map(str, row))
                file.write(f'{idx}\t{line}\n')

        print('Archivo TXT generado correctamente con cabecera desplazada a la derecha.')


    def save_vehicle_data_txt(self):
        with open('vehicle_data_final.txt', 'w') as file:
            file.write("VEHICLE\n")
            file.write("NUMBER     CAPACITY\n")
            file.write("  100       10000\n\n")

            file.write("CUSTOMER\n")
            file.write("CUST NO.  XCOORD.    YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME\n\n")

            # Modificación para ajustar el espaciado y alineación de las columnas
            for client in self.clients:
                # Aquí se ajusta el formato para que los ceros en 'SERVICE TIME' estén correctamente alineados
                file.write(f"{client:>8} {0:>10} {0:>10} {0:>10} {0:>12} {100000:>10}{0:>5}\n")

        print("Archivo vehicle_data_final.txt generado correctamente.")

    def process_all_data(self):
        # Diccionario para almacenar cada dataframe combinado
        combined_dataframes = {}

        # Rango de los números de archivos
        for i in range(601, 631):
            # Ruta de los archivos
            path_zero = f'speed[{i}]_[0].csv'
            path_one = f'speed[{i}]_[1].csv'

            # Leer los archivos
            df_zero = pd.read_csv(path_zero)
            df_one = pd.read_csv(path_one)

            # Combinar los dataframes
            combined_df = pd.concat([df_zero, df_one], ignore_index=True)

            # Filtrar las filas que no quieres, manteniendo solo las que empiezan desde las 8:00
            #combined_df['start_time'] = combined_df['Period'].str.split('-').str[0]  # Extraer el tiempo de inicio
            #combined_df['start_time'] = pd.to_datetime(combined_df['start_time'], format='%H:%M')  # Convertir a datetime
            #combined_df = combined_df[combined_df['start_time'].dt.hour * 60 + combined_df['start_time'].dt.minute >= 480]  # Filtrar

            # Guardar el dataframe combinado en el diccionario
            combined_dataframes[i] = combined_df

        for i in range(701, 715):
            # Ruta de los archivos
            path_zero = f'speed[{i}]_[0].csv'
            path_one = f'speed[{i}]_[1].csv'

            # Leer los archivos
            df_zero = pd.read_csv(path_zero)
            df_one = pd.read_csv(path_one)

            # Combinar los dataframes
            combined_df = pd.concat([df_zero, df_one], ignore_index=True)

                # Filtrar las filas que no quieres, manteniendo solo las que empiezan desde las 8:00
            #combined_df['start_time'] = combined_df['Period'].str.split('-').str[0]  # Extraer el tiempo de inicio
            #combined_df['start_time'] = pd.to_datetime(combined_df['start_time'], format='%H:%M')  # Convertir a datetime
            #combined_df = combined_df[combined_df['start_time'].dt.hour * 60 + combined_df['start_time'].dt.minute >= 480]  # Filtrar

            # Guardar el dataframe combinado en el diccionario
            combined_dataframes[i] = combined_df

            self.all_data = pd.concat(combined_dataframes.values(), ignore_index=True)
        
        self.all_data['Speed'] = self.all_data['Speed'] / 60
        self.all_data.dropna()
        #Calculamos velocidades promedio y su desv estandar
        self.aggregated_data = self.all_data.groupby(['Period', 'Link']).agg(
            speed=('Speed', 'mean'),
            speed_std=('Speed', 'std')
            ).reset_index()
        #self.aggregated_data['speed_std'] = self.aggregated_data['speed_std'].fillna(0)#Clase con data històrica

#Clase con data històrica
class DataCalculations():
    def __init__(self, env, congestion_max_duration):
        self.env = env
        self.full_data = self.env.full_data
        self.data_unique = self.env.data_average

        self.congestion_max_duration = congestion_max_duration

        self.total_congestions = 0

        
        #Diccioario que tiene toda la data para cada minuto
        self.travel_data = {}

        #Diccionario para guardar recordar velocidades que se cambian cuando ocurre un evento inesperado
        self.unexpected_event_velocity = {}

        #Otro diccionario que almacena para cada node_start los node_end que puede tener
        self.travel_arc_information_dictionary = {}

        #Diccionario que guarda la latitud y longitude de todo nodo
        self.latitude_and_longitude = {}

        #Diccionarios para calcular las desviaciones estándar de las velocidades
        #self.speed_list = defaultdict(list)
        self.get_standard_deviation_dict = {}

        self.probability_of_event = {}

        #Calculamos las desv estándar

        #Agregamos minutos al dataframe
        self.process_data()

        self.get_standard_deviation()
        
        #Lo pasamos a diccionarios
        self.arcs_data_to_dictionary()

        #Leemos todos los dataframes
        self.read_all_data()

        #self.all_30_min_data = False
        
        #Obtenemos la media de todos los dataframes
        self.get_mean_of_all_intervals()

        self.store_probability_for_event_of_all_arcs()

        del self.env
        del self.full_data, self.data_unique


        self.congested_arcs = {}

        self.all_arc_velocity = {}

        self.event_quantity_per_episode = 0

    def get_interpolated_speed(self, row, speed_lookup):
        link = row['Link']
        minute_start = row['Minute_start']
        original_speed = row['Speed']
        
        # Intervalo 1: 420 < minute_start < 540
        if 420 < minute_start < 540:
            ratio_540 = (minute_start-420)/(540-420)
            ratio_420 = 1-ratio_540
            sp_420 = speed_lookup.get((link, 420), None)
            sp_540 = speed_lookup.get((link, 540), None)
            if sp_420 is not None and sp_540 is not None:
                speed = (sp_420*ratio_420) + (sp_540*ratio_540)
                return speed
        
        # Intervalo 2: 660 < minute_start < 840
        elif 660 < minute_start < 840:
            ratio_840 = (minute_start-660)/(840-660)
            ratio_660 = 1-ratio_840
            sp_660 = speed_lookup.get((link, 660), None)
            sp_840 = speed_lookup.get((link, 840), None)
            if sp_660 is not None and sp_840 is not None:
                speed = (sp_660*ratio_660) + (ratio_840*sp_840)
                return speed
        
        # Intervalo 3: 960 < minute_start < 1080
        elif 960 < minute_start < 1080:
            ratio_1080 = (minute_start-960)/(1080-960)
            ratio_960 = 1-ratio_1080
            sp_960 = speed_lookup.get((link, 960), None)
            sp_1080 = speed_lookup.get((link, 1080), None)
            if sp_960 is not None and sp_1080 is not None:
                speed = (sp_960*ratio_960) + (sp_1080*ratio_1080)
                return speed
         
        # Si no está en ninguno de los intervalos, no se modifica la velocidad
        else:
            return original_speed

    def get_standard_deviation(self):
        std_speed_dict = self.full_data.set_index(['Node_Start', 'Node_End', "Minute_start"])['speed_std'].to_dict()
        for idx, row in self.full_data.iterrows():
            key = (row['Node_Start'], row['Node_End'], row["Minute_start"])
            minute_start = row['Minute_start']
            
            # Intervalo 1: 420 < minute_start < 540
            if 420 < minute_start < 540:
                ratio_540 = (minute_start-420)/(540-420)
                ratio_420 = 1-ratio_540
                sp_420 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 418))
                sp_540 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 542))
                if sp_420 is not None and sp_540 is not None:
                    row['Speed_std'] =  (sp_420*ratio_420) + (sp_540*ratio_540)
                    self.get_standard_deviation_dict[key] = row['speed_std']
            
            # Intervalo 2: 660 < minute_start < 840
            elif 660 < minute_start < 840:
                ratio_840 = (minute_start-660)/(840-660)
                ratio_660 = 1-ratio_840
                sp_660 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 658))
                sp_840 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 842))
                if sp_660 is not None and sp_840 is not None:
                    self.get_standard_deviation_dict[key] = (sp_660*ratio_660) + (ratio_840*sp_840)
            
            # Intervalo 3: 960 < minute_start < 1080
            elif 960 < minute_start < 1080:
                ratio_1080 = (minute_start-960)/(1080-960)
                ratio_960 = 1-ratio_1080
                sp_960 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 958))
                sp_1080 = std_speed_dict.get((row['Node_Start'], row['Node_End'], 1082))
                if sp_960 is not None and sp_1080 is not None:
                    self.get_standard_deviation_dict[key] = (sp_960*ratio_960) + (sp_1080*ratio_1080)
            
            # Si no está en ninguno de los intervalos, no se modifica la desviación estándar
            else:
                self.get_standard_deviation_dict[key] = row['speed_std']
            

    def arcs_data_to_dictionary(self):
        for idx, row in self.full_data.iterrows():
            key = (row['Node_Start'], row['Node_End'], row['Minute_start'])
            self.travel_data[key] = (row['Length'], row['Speed'])

        for idx, row in self.data_unique.iterrows():
            node_start = row['Node_Start']
            node_end = row['Node_End']
            if node_start not in self.travel_arc_information_dictionary:
                self.travel_arc_information_dictionary[node_start] = []  # Inicializar la lista si la clave no existe
            self.travel_arc_information_dictionary[node_start].append(node_end)

        for idx, row in self.data_unique.iterrows():
            node_start = row['Node_Start']
            latitude = row["Latitude_Start"]
            longitude = row["Longitude_Start"]
            if node_start not in self.latitude_and_longitude:
                self.latitude_and_longitude[node_start] = []
            self.latitude_and_longitude[node_start] = [latitude,longitude]

    def process_data(self):
        # Ordenamos el DataFrame por Link y Minute_start para asegurar el correcto procesamiento
        df = self.full_data.sort_values(by=['Link', 'Minute_start'])

        # Lista para guardar las nuevas filas
        new_rows = []

        # Recorremos cada link
        for link in df['Link'].unique():
            # Filtramos el DataFrame por link
            link_df = df[df['Link'] == link]
            prev_row = None

            # Iteramos sobre las filas del DataFrame filtrado
            for index, row in link_df.iterrows():
                if prev_row is not None:
                    # Calculamos la diferencia en minutos
                    time_diff = row['Minute_start'] - prev_row['Minute_start']

                    # Verificamos si la diferencia es mayor a 1 minuto
                    if time_diff > 1:
                        # Calculamos los nuevos valores para Minute_start y creamos nuevas filas
                        new_minute_starts = range(prev_row['Minute_start'] + 1, row['Minute_start'], 1)

                        for minute in new_minute_starts:
                            new_row = prev_row.copy()
                            new_row['Minute_start'] = minute
                            new_rows.append(new_row)

                # Actualizamos prev_row con la fila actual
                prev_row = row

        # Convertimos la lista de nuevas filas en un DataFrame
        new_rows_df = pd.DataFrame(new_rows)
        # Concatenamos el DataFrame original con el nuevo DataFrame de filas añadidas
        self.full_data = pd.concat([df, new_rows_df]).sort_values(by=['Link', 'Minute_start'])
        
        speed_dict = self.full_data.set_index(['Link', 'Minute_start'])['Speed'].to_dict()

        self.full_data['Speed'] = self.full_data.apply(
            lambda row: self.get_interpolated_speed(row, speed_dict), axis=1
            )

        self.full_data['Travel_Time'] = (self.full_data['Length']) / (self.full_data['Speed'])



    def generate_normal_velocity(self, Node_start, Node_end, Minute_start):
        Minute_start = math.floor(Minute_start)
        if Minute_start % 2 != 0:
            Minute_start -=1

        key = ((Node_start, Node_end, Minute_start))
        #key_std = (Node_start, Node_end)
        #or probability_for_event > probability_input
        #Obtenemos la velocidad
        self.length, speed = self.travel_data[key]

        #speed = self.travel_data[key][1]
        #mean_log_speed = np.log(speed)

        std = self.get_standard_deviation_dict[key]
        #if  np.isnan(std):
        #    print(std)

        #self.random_velocity = np.random.lognormal(mean_log_speed, std)
        #self.random_velocity = np.random.normal(loc=speed, scale=std)
        self.random_velocity = random.gauss(speed,std)
        self.random_velocity = max(self.random_velocity, 0)


        #Pondremos velocidad máxima 60 km/hr para calles sin carretera (recordar que la velocidad esta en km/min)
        if speed < 1 and self.random_velocity>1:
            self.random_velocity = 1
        
        if self.random_velocity > 2:
            self.random_velocity = 2
        
        if self.random_velocity <= 0:
            self.random_velocity = 0.001
        

        self.travel_time = self.length / self.random_velocity

        #Agregamos esta velocidad al diccionario.
        self.all_arc_velocity[key] = [self.random_velocity, self.travel_time, self.length]


    def create_random_velocity(self, Node_start, Node_end, tau_episode, probability_input):
        Minute_start = math.floor(tau_episode)
        if Minute_start % 2 != 0:
            Minute_start -= 1

        key_arc = (Node_start, Node_end)
        key_minute = (Node_start, Node_end, Minute_start)

        if key_arc not in self.congested_arcs:
            if key_minute not in self.all_arc_velocity:
                self.generate_normal_velocity(Node_start, Node_end, Minute_start)
            else:
                self.random_velocity, self.travel_time, self.length = self.all_arc_velocity[key_minute]

        else:
            event_end = self.congested_arcs[key_arc][1]
            if tau_episode >= event_end:
                if key_minute not in self.all_arc_velocity:
                    self.generate_normal_velocity(Node_start, Node_end, Minute_start)
                else:
                    self.random_velocity, self.travel_time, self.length = self.all_arc_velocity[key_minute]
            else:
                self.length, speed = self.travel_data[key_minute]
                congestion_multiplier = self.congested_arcs[key_arc][0]
                velocity = max(speed * congestion_multiplier, 0.0001)
                travel_time = self.length / velocity

                self.random_velocity = velocity
                self.travel_time = travel_time

        return self.travel_time, self.random_velocity, self.length
    

    def create_random_unexpected_event(self, Minute_start, probability_input, max_depth):
        #Generamos un evento aleatorio en un nodo aleatorio
        #Number_clients = len(clients_not_visited)
        #if Number_clients == 0:
        #    Node_start = random.randint(0,1900)

        #else:
        #    Node_start = random.choice(clients_not_visited)
        #self.eliminate_velocity_penalization(Minute_start)
        #self.congested_arcs = {}

        #if 300<=Minute_start< 400:
        #    time = 300

        #else:
        #    time = ((Minute_start-300) // 120) * 120
        #    time += 300

        all_nodes = list(range(1901))
        probability_for_79_congestions = random.random()
        if probability_for_79_congestions< 0.54875:
            Congested_Nodes = random.sample(all_nodes, 8)
        
        else:
            Congested_Nodes = random.sample(all_nodes, 8)
        
        

        Minute_start = math.floor(Minute_start)
        if Minute_start % 2 != 0:
            Minute_start -=1

        for Node_start in Congested_Nodes:
            probability_for_congestion = random.random()
            if probability_for_congestion < 0.65:
                penalization = random.uniform(0.4, 0.5)
                velocity_penalization = penalization
                #state_time_elimination = random.randint(30,120)
                #state_time_elimination = random.randint(60,120)
                #state_time_elimination = 120
                #print(velocity_penalization)
                #velocity = round(random.uniform(0.1, 0.3), 4)

                #if state_time_elimination != 60:
                #    state_time_elimination = random.randint(30,120)
                
                #else:
                #    state_time_elimination = random.randint(30,60)
            elif probability_for_congestion < 0.916:
                penalization = random.uniform(0.3, 0.4)
                velocity_penalization = penalization
            
            elif probability_for_congestion < 0.9888:
                penalization = random.uniform(0.2, 0.3)
                velocity_penalization = penalization


            else:
                #velocity = round(random.uniform(0.01, 0.08), 4)
                velocity_penalization = random.uniform(0.1, 0.2)
                state_time_elimination = 60

            probability_for_congestion_time = random.random()

            if probability_for_congestion_time < 0.663:
                state_time_elimination = 30
            
            elif probability_for_congestion_time < 0.7886:
                state_time_elimination = 60
            
            else:
                state_time_elimination = 120

            if state_time_elimination + Minute_start > 1198:
                state_time_elimination = 1198 - Minute_start

            probability_for_event = random.random()
            if probability_for_event <= probability_input:
                node_starts, depth = self.get_all_node_starts(Node_start, 0, max_depth-1)
                for node_start in node_starts:
                    connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                    
                    for affected_node in connected_nodes:
                        if depth[node_start] == 0:
                            factor = 1
                        
                        elif depth[node_start] == 1:
                            factor = 0.83
                        
                        elif depth[node_start] == 2:
                            factor = 0.78
                        
                        elif depth[node_start] == 3:
                            factor = 0.73

                        #print("La penalizacion pre factor", velocity_penalization)
                        velocity_penalization_for_depth = velocity_penalization/factor
                        #print(velocity_penalization_for_depth)
                        #print("La penalizacion pos factor", velocity_penalization_for_depth)
                        #velocity_penalization = 1-velocity_penalization
    
                        self.congested_arcs[(float(node_start), float(affected_node))] = [float(velocity_penalization_for_depth), float(Minute_start+state_time_elimination)]




                        #for time_event in range(Minute_start, Minute_start + state_time_elimination, 2):

                            #key = (node_start, affected_node, time_event)
                            #if key not in self.unexpected_event_velocity:
                            #    original_length, original_speed = self.travel_data[key]
                                #Factor de penalización por profundidad
                            #    Depth_Penalization = (0.95) ** (depth[node_start])
                                #Penalización total
                            #    penalization = (original_speed - original_speed*speed_event_penalization)*Depth_Penalization
                            #    penalized_speed = original_speed - penalization
                            #    self.travel_data[key] = (original_length, penalized_speed)

                            #    self.unexpected_event_velocity[key] = (original_speed, Minute_start + state_time_elimination)

    def create_one_random_unexpected_event(self, Minute_start, probability_input, max_depth):
        #Generamos un evento aleatorio en un nodo aleatorio
        #Number_clients = len(clients_not_visited)
        #if Number_clients == 0:
        #    Node_start = random.randint(0,1900)

        #else:
        #    Node_start = random.choice(clients_not_visited)
        #self.eliminate_velocity_penalization(Minute_start)
        #self.congested_arcs = {}

        #if 300<=Minute_start< 400:
        #    time = 300

        #else:
        #    time = ((Minute_start-300) // 120) * 120
        #    time += 300

        all_nodes = list(range(1901))

        Congested_Nodes = random.sample(all_nodes, 2)

        probability_for_3_congestions = random.random()
        if probability_for_3_congestions < 0.618:
            Congested_Nodes = random.sample(all_nodes, 3)

        

        Minute_start = math.floor(Minute_start)
        if Minute_start % 2 != 0:
            Minute_start -=1

        for Node_start in Congested_Nodes:
            probability_for_congestion = random.random()
            if probability_for_congestion < 0.6502:
                penalization = random.uniform(0.4, 0.5)
                velocity_penalization = penalization

            elif probability_for_congestion < 0.9116:
                penalization = random.uniform(0.3, 0.4)
                velocity_penalization = penalization
            
            elif probability_for_congestion < 0.9893:
                penalization = random.uniform(0.2, 0.3)
                velocity_penalization = penalization


            else:
                velocity_penalization = random.uniform(0.1, 0.2)
                state_time_elimination = 60

            probability_for_congestion_time = random.random()

            if probability_for_congestion_time < 0.6634:
                state_time_elimination = 30
            
            elif probability_for_congestion_time < 0.7879:
                state_time_elimination = 60
            
            else:
                state_time_elimination = 120

            if state_time_elimination + Minute_start > 1198:
                state_time_elimination = 1198 - Minute_start

            probability_for_event = random.random()
            if probability_for_event <= probability_input:
                node_starts, depth = self.get_all_node_starts(Node_start, 0, max_depth-1)
                for node_start in node_starts:
                    connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                    
                    for affected_node in connected_nodes:
                        if depth[node_start] == 0:
                            factor = 1
                        
                        elif depth[node_start] == 1:
                            factor = 0.83
                        
                        elif depth[node_start] == 2:
                            factor = 0.78
                        
                        elif depth[node_start] == 3:
                            factor = 0.73

                        #print("La penalizacion pre factor", velocity_penalization)
                        velocity_penalization_for_depth = velocity_penalization/factor
                        #print(velocity_penalization_for_depth)
                        #print("La penalizacion pos factor", velocity_penalization_for_depth)
                        #velocity_penalization = 1-velocity_penalization
    
                        self.congested_arcs[(float(node_start), float(affected_node))] = [float(velocity_penalization_for_depth), float(Minute_start+state_time_elimination)]

    def get_all_node_starts(self, Node_start, depth, max_depth, visited=None, node_depth = None):
        #Funcion obtiene todos los node_starts
        if visited is None:
            visited = set()
            node_depth = {}


        # Añadir el nodo actual al conjunto de visitados
        visited.add(Node_start)
        node_depth[Node_start] = depth

        if depth < max_depth:
            # Obtener nodos conectados al nodo actual
            connected_nodes = self.travel_arc_information_dictionary.get(Node_start, [])

            # Recursivamente buscar en los nodos conectados que no han sido visitados
            for node in connected_nodes:
                if node not in visited:
                    self.get_all_node_starts(node, depth + 1, max_depth, visited, node_depth)

        return visited, node_depth


    def eliminate_velocity_penalization(self, current_time):
        #si el tiempo en que nos encontramos es mayor o igual a la clave del diccionario, entonces se cambia a la velocidad anterior y se elimina.
        if not self.unexpected_event_velocity:
            return

        keys_to_delete = [key for key, value in self.unexpected_event_velocity.items() if value[1] <= current_time]

        for key in keys_to_delete:
            # Restaurar la velocidad original en travel_data antes de eliminar la entrada en unexpected_event_velocity y la llave completa por eficiencia.
            original_speed, expiration_time = self.unexpected_event_velocity[key]
            length = self.travel_data[key][0]
            #if original_speed < 0.09:
               # print(original_speed)
                #print(self.travel_data[key][1])
            self.travel_data[key] = (length, original_speed)

            del self.unexpected_event_velocity[key]


    def travel_time(self, node_start, node_end, time):
        time_int = int(time)
        key = (node_start, node_end, time_int)
        length, speed = self.travel_data[key]
        travel_time = length/speed

        return travel_time

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calcula la distancia entre dos puntos en la superficie de una esfera utilizando la fórmula de Haversine.
        """
        R = 6371  # Radio de la Tierra en kilómetros

        # Convertir coordenadas de grados a radianes
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        # Fórmula de Haversine
        a = math.sin(delta_phi / 2.0)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distancia = R * c  # Distancia en kilómetros
        return distancia

    def get_nodes_in_circle(self, center_node, radius):
        """
        Retorna una lista de nodos que están dentro de un radio dado desde el nodo central.

        Parámetros:
        - center_node: El nodo que será el centro del círculo.
        - radius: El radio del círculo en kilómetros.

        Retorna:
        - Una lista de IDs de nodos dentro del radio especificado.
        """


        center_lat, center_lon = self.latitude_and_longitude[center_node]
        nodes_in_circle = []

        for node_id, (lat, lon) in self.latitude_and_longitude.items():
            distancia = self.haversine_distance(center_lat, center_lon, lat, lon)
            if distancia <= radius:
                nodes_in_circle.append(node_id)

        return nodes_in_circle

    def create_random_unexpected_event_with_radius(self, Minute_start, probability_input, radius):
        self.circle_center = []
        self.congested_arcs = {}
        
        if 300<=Minute_start< 400:
            time = 300

        else:
            time = ((Minute_start-300) // 120) * 120
            time += 300

        all_nodes = list(range(1901))
        Congested_Nodes = random.sample(all_nodes, 9)
        state_time_elimination = 60        

        if state_time_elimination + Minute_start > 1198:
            state_time_elimination = 1198 - Minute_start

        for Node_start in Congested_Nodes:
            probability_for_event = random.random()

            probability_for_congestion = random.random()
            if probability_for_congestion < 0.67:
                penalization = random.uniform(0.4, 0.5)
                velocity_penalization = penalization
                #state_time_elimination = random.randint(30,120)
                state_time_elimination = 120
                #state_time_elimination = 120
                #print(velocity_penalization)
                #velocity = round(random.uniform(0.1, 0.3), 4)

                #if state_time_elimination != 60:
                #    state_time_elimination = random.randint(30,120)
                
                #else:
                #    state_time_elimination = random.randint(30,60)
            elif probability_for_congestion < 0.916:
                penalization = random.uniform(0.3, 0.4)
                velocity_penalization = penalization
                state_time_elimination = 120
            
            elif probability_for_congestion < 0.989:
                penalization = random.uniform(0.2, 0.3)
                velocity_penalization = penalization
                state_time_elimination = 120

            else:
                #velocity = round(random.uniform(0.01, 0.08), 4)
                velocity_penalization = random.uniform(0.1, 0.2)
                state_time_elimination = 120
            #Obtenemos todos los nodos afectados en el circulo de radio 2
            node_starts = self.get_nodes_in_circle(Node_start, radius)
            self.circle_center.append(Node_start)
            for node_start in node_starts:
                connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                for affected_node in connected_nodes:
                    self.congested_arcs[(node_start, affected_node)] = [velocity_penalization, Minute_start+state_time_elimination]
      

    def calculate_nodes_caracteristics(self, nodes):
        distances = []
        if len(nodes) <= 1:
            return 0,0,0,0
        else:
            for i in range(len(nodes)):
                node_i = nodes[i]
                if node_i not in self.latitude_and_longitude:
                    continue
                lat1, lon1 = self.latitude_and_longitude[node_i]
                for j in range(i + 1, len(nodes)):
                    node_j = nodes[j]
                    if node_j not in self.latitude_and_longitude:
                        continue
                    lat2, lon2 = self.latitude_and_longitude[node_j]
                    distance = self.haversine_distance(lat1, lon1, lat2, lon2)
                    distances.append(distance)

            if len(distances) == 0:
                return 0
            desviacion_estandar = np.std(distances)
            mean = np.mean(distances)
            max_distance = max(distances)
            min_distance = min(distances)
            return desviacion_estandar, mean, max_distance, min_distance
        
    def calculate_clients_density(self, nodes):
        if not nodes or len(nodes) < 2:
            return 0  # Si no hay suficientes nodos, la densidad es 0
        
        latitudes = []
        longitudes = []
        
        for node in nodes:
            if node in self.latitude_and_longitude:
                lat, lon = self.latitude_and_longitude[node]
                latitudes.append(lat)
                longitudes.append(lon)
        
        if not latitudes or not longitudes:
            return 0  # Si no hay coordenadas válidas, la densidad es 0
        
        # Determinar los límites geográficos
        min_lat, max_lat = min(latitudes), max(latitudes)
        min_lon, max_lon = min(longitudes), max(longitudes)
        
        # Calcular área aproximada en km²
        avg_lat = (min_lat + max_lat) / 2  # Latitud promedio para corrección
        width_km = (max_lon - min_lon) * 111.321 * np.cos(np.radians(avg_lat))  # Longitud en km
        height_km = (max_lat - min_lat) * 111.0  # Latitud en km
        area_km2 = width_km * height_km
        
        if area_km2 == 0:
            return 0  # Prevenir división por cero
        
        # Calcular densidad de clientes
        num_clients = len(latitudes)
        density = num_clients / area_km2
        
        return density

    def calculate_distance_metrics_to_depot(self, nodes):
        if len(nodes) <= 1:
            return 0,0,0,0

        else:

            distances = []
            node_zero = 0

            lat0, lon0 = self.latitude_and_longitude[node_zero]

            for node in nodes:
                if node == node_zero:
                    continue  # Omitir si el nodo es el mismo que el nodo 0
                if node not in self.latitude_and_longitude:
                    continue
                lat, lon = self.latitude_and_longitude[node]
                distance = self.haversine_distance(lat0, lon0, lat, lon)
                distances.append(distance)

            if len(distances) == 0:
                print("No se pudieron calcular distancias entre el nodo 0 y los nodos proporcionados.")
                return None

            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            max_distance = max(distances)
            min_distance = min(distances)

            return mean_distance, std_distance, max_distance, min_distance
    
    def calculate_clients_dispersion(self, nodes, vehicle_positions):
        #nodes.append(0)
        lat_list = []
        long_list = []
        if len(nodes) <= 2:
            return 0, 0 ,0, 0
        else:
            for node in nodes:
                lat, lon = self.latitude_and_longitude[node]
                lat_list.append(lat)
                long_list.append(lon)
            
            lat_centroid = np.mean(lat_list)
            long_centroid = np.mean(long_list)

            vehicle_distance_to_centroid = 0
            for vehicle in vehicle_positions:
                lat, lon = self.latitude_and_longitude[vehicle]
                vehicle_distance_to_centroid += self.haversine_distance(lat_centroid, long_centroid, lat, lon)

            distance_list = []

            for i in range(len(nodes)):
                lat_client = lat_list[i]
                long_client = long_list[i]
                distance = self.haversine_distance(lat_centroid, long_centroid, lat_client, long_client)
                distance_list.append(distance)
            

            
            max_dist_to_centroid = max(distance_list)
            mean_dist_to_centroid = np.mean(distance_list)
            std_dist_to_centroid = np.std(distance_list)

        return vehicle_distance_to_centroid, max_dist_to_centroid, mean_dist_to_centroid, std_dist_to_centroid
            
    @staticmethod
    def haversine_distance(lon1, lat1, lon2, lat2):
        '''
        Input: longitud y latitud de 2 nodos.
        Output: distancia entre los 2 nodos.
        '''
        # Radio de la Tierra en km.
        R = 6371.0

        # Convertir grados en radianes.
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Diferencias de coordenadas.
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Fórmula Haversine
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return distance
    
    def read_all_data(self):
        combined_dataframes = {}

        # ----- 1) Lectura y combinación de archivos -----
        # Rango 1 (601 a 630)
        cont = 0
        for i in range(601, 631):
            path_zero = f'speed[{i}]_[0].csv'
            path_one = f'speed[{i}]_[1].csv'
            
            # Leer CSVs
            df_zero = pd.read_csv(path_zero)
            df_one = pd.read_csv(path_one)
            
            # Combinar
            combined_df = pd.concat([df_zero, df_one], ignore_index=True)
            
            # Extraer hora de inicio
            combined_df['start_time'] = combined_df['Period'].str.split('-').str[0]
            combined_df['start_time'] = pd.to_datetime(combined_df['start_time'], format='%H:%M')
            
            # Filtrar registros >= 08:00
            combined_df = combined_df[combined_df['start_time'].dt.hour >= 8]
            
            # Calcular intervalos
            combined_df['30_min_interval'] = combined_df['start_time'].dt.floor('30min')
            combined_df['60_min_interval'] = combined_df['start_time'].dt.floor('h')
            combined_df['90_min_interval'] = combined_df['start_time'].dt.floor('90min')
            combined_df['120_min_interval'] = combined_df['start_time'].dt.floor('2h')
            
            # ----- Agrupación por intervalo de 30 minutos -----
            grouped_30_min = combined_df.groupby(['Link', '30_min_interval'], as_index=False)['Speed'].mean()
            grouped_30_min['Period'] = grouped_30_min['30_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 60 minutos -----
            grouped_60_min = combined_df.groupby(['Link', '60_min_interval'], as_index=False)['Speed'].mean()
            grouped_60_min['Period'] = grouped_60_min['60_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 90 minutos -----
            grouped_90_min = combined_df.groupby(['Link', '90_min_interval'], as_index=False)['Speed'].mean()
            grouped_90_min['Period'] = grouped_90_min['90_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 120 minutos -----
            grouped_120_min = combined_df.groupby(['Link', '120_min_interval'], as_index=False)['Speed'].mean()
            grouped_120_min['Period'] = grouped_120_min['120_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # Guardamos todos los DataFrames en el diccionario
            combined_dataframes[f"{i}_30min"] = grouped_30_min
            combined_dataframes[f"{i}_60min"] = grouped_60_min
            combined_dataframes[f"{i}_90min"] = grouped_90_min
            combined_dataframes[f"{i}_120min"] = grouped_120_min
            cont+= 1

        # Rango 2 (701 a 714)
        for i in range(701, 715):
            path_zero = f'speed[{i}]_[0].csv'
            path_one = f'speed[{i}]_[1].csv'
            
            # Leer CSVs
            df_zero = pd.read_csv(path_zero)
            df_one = pd.read_csv(path_one)
            
            # Combinar
            combined_df = pd.concat([df_zero, df_one], ignore_index=True)
            
            # Extraer hora de inicio
            combined_df['start_time'] = combined_df['Period'].str.split('-').str[0]
            combined_df['start_time'] = pd.to_datetime(combined_df['start_time'], format='%H:%M')
            
            # Filtrar registros >= 08:00
            combined_df = combined_df[combined_df['start_time'].dt.hour >= 8]
            
            # Calcular intervalos
            combined_df['30_min_interval'] = combined_df['start_time'].dt.floor('30min')
            combined_df['60_min_interval'] = combined_df['start_time'].dt.floor('h')
            combined_df['90_min_interval'] = combined_df['start_time'].dt.floor('90min')
            combined_df['120_min_interval'] = combined_df['start_time'].dt.floor('2h')
            
            # ----- Agrupación por intervalo de 30 minutos -----
            grouped_30_min = combined_df.groupby(['Link', '30_min_interval'], as_index=False)['Speed'].mean()
            grouped_30_min['Period'] = grouped_30_min['30_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 60 minutos -----
            grouped_60_min = combined_df.groupby(['Link', '60_min_interval'], as_index=False)['Speed'].mean()
            grouped_60_min['Period'] = grouped_60_min['60_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 90 minutos -----
            grouped_90_min = combined_df.groupby(['Link', '90_min_interval'], as_index=False)['Speed'].mean()
            grouped_90_min['Period'] = grouped_90_min['90_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # ----- Agrupación por intervalo de 120 minutos -----
            grouped_120_min = combined_df.groupby(['Link', '120_min_interval'], as_index=False)['Speed'].mean()
            grouped_120_min['Period'] = grouped_120_min['120_min_interval'].dt.strftime('%H:%M-%H:%M')
            
            # Guardamos todos los DataFrames en el diccionario
            combined_dataframes[f"{i}_30min"] = grouped_30_min
            combined_dataframes[f"{i}_60min"] = grouped_60_min
            combined_dataframes[f"{i}_90min"] = grouped_90_min
            combined_dataframes[f"{i}_120min"] = grouped_120_min
            cont+=1

        # ----- 2) Concatenar todos los DataFrames (si lo necesitas globalmente) -----
        self.all_30_min_data = pd.concat([df for key, df in combined_dataframes.items() if '30min' in key], ignore_index=True)
        print(self.all_30_min_data)
        self.all_60_min_data = pd.concat([df for key, df in combined_dataframes.items() if '60min' in key], ignore_index=True)
        self.all_90_min_data = pd.concat([df for key, df in combined_dataframes.items() if '90min' in key], ignore_index=True)
        self.all_120_min_data = pd.concat([df for key, df in combined_dataframes.items() if '120min' in key], ignore_index=True)

    def get_mean_of_all_intervals(self):
        # Crear una copia del DataFrame original
        df = self.full_data.copy()

        # Extraer la hora desde el campo 'Period'
        df["hour"] = df["Period"].str.extract(r"(\d{2})")[0]
        df["hour"] = df["hour"].astype(int)  # Convertir a entero

        # Convertir 'Period' a formato de tiempo para facilitar las agrupaciones
        df["start_time"] = pd.to_datetime(df["Period"].str.split('-').str[0], format='%H:%M')

        # Crear intervalos de tiempo utilizando los nuevos nombres
        #df["60_min_interval"] = df["start_time"].dt.floor('h')      # Intervalo de 60 minutos
        df["30_min_interval"] = df["start_time"].dt.floor('30min')  # Intervalo de 30 minutos
        #df["90_min_interval"] = df["start_time"].dt.floor('90min')  # Intervalo de 90 minutos
        #df["120_min_interval"] = df["start_time"].dt.floor('2h')    # Intervalo de 120 minutos

        self.df_mean = df.groupby(["Link", "Node_Start", "Node_End"]).agg({"Speed": "mean"}).reset_index()
        self.df_mean.rename(columns={"Speed": "avg_speed"}, inplace=True)
        self.df_mean["avg_speed"] = self.df_mean["avg_speed"]  # Escalar velocidad si es necesario
        #print(self.df_mean)

        # Agrupación por intervalos de 60 minutos
        #self.df_60 = df.groupby(["Link", "Node_Start", "Node_End", "60_min_interval"]).agg({"Speed": "mean"}).reset_index()
        #self.df_60.rename(columns={"Speed": "avg_speed"}, inplace=True)
        #self.df_60["avg_speed"] = df_60["avg_speed"] # Escalar velocidad si es necesario

        # Agrupación por intervalos de 30 minutos
        self.df_30 = df.groupby(["Link", "Node_Start", "Node_End", "30_min_interval"]).agg({"Speed": "mean"}).reset_index()
        self.df_30.rename(columns={"Speed": "avg_speed"}, inplace=True)
        self.df_30["avg_speed"] = self.df_30["avg_speed"]  # Escalar velocidad si es necesario

        print(self.df_30)

        # Agrupación por intervalos de 120 minutos
        #self.df_120 = df.groupby(["Link", "Node_Start", "Node_End", "120_min_interval"]).agg({"Speed": "mean"}).reset_index()
        #self.df_120.rename(columns={"Speed": "avg_speed"}, inplace=True)
        #self.df_120["avg_speed"] = df_120["avg_speed"]  # Escalar velocidad si es necesario

    
    def store_probability_for_event_of_all_arcs(self):
        unit_of_time_of_congestions = self.congestion_max_duration/60
        hours = 8/unit_of_time_of_congestions
        self.merged_df_30 = pd.merge(
            self.all_30_min_data,
            self.df_mean,  # Assuming grouped_df contains the "Speed" column and "hour" column
            on=["Link"],
            how="outer"
            )
        self.merged_df_30["Speed"] = self.merged_df_30["Speed"]/60

        

        for _, row in self.merged_df_30.iterrows():
            key = (row["Node_Start"], row["Node_End"])
            if key not in self.probability_of_event:
                self.probability_of_event[key] = 0

            # Aplicamos el filtro para cada fila
            if row['Speed'] <= 0.4 * row['avg_speed'] and row['Speed'] >= 0.1 * row['avg_speed']:
                self.probability_of_event[key] += 1
        
        for key in self.probability_of_event:
            self.probability_of_event[key] = self.probability_of_event[key]*2/(44*hours*3)


        #self.node_start_sums = {}
        #self.node_start_counts = {}

        # Iterar sobre el diccionario original
        #for (node_start, node_end), value in self.probability_of_event.items():
        #    if node_start not in self.node_start_sums:
        #        self.node_start_sums[node_start] = 0
        #        self.node_start_counts[node_start] = 0
        #    self.node_start_sums[node_start] += value
        #    self.node_start_counts[node_start] += 1

        # Crear un nuevo diccionario para guardar los promedios
        #self.node_start_averages = {
        #    node_start: self.node_start_sums[node_start] / self.node_start_counts[node_start]
        #    for node_start in self.node_start_sums
        #    }
        
    def create_random_unexpected_event_with_probability(self, Minute_start, probability_input, max_depth):

        if probability_input != 0:
            cont = 0
            for key in self.probability_of_event:
            #for key in self.node_start_averages:
                probability_for_congestion = random.random()
                if probability_for_congestion < self.probability_of_event[key]:
                #if probability_for_congestion < self.node_start_averages[key]:
                    cont += 1
                    node = key[0]
                    #node = key
                    velocity_penalization = random.uniform(0.1, 0.4)
                    state_time_elimination = random.randint(30,self.congestion_max_duration)
                    node_starts, depth = self.get_all_node_starts(node, 0, max_depth-1)
                    for node_start in node_starts:
                        connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                        
                        for affected_node in connected_nodes:
                            if node_start == affected_node:
                                continue

                            if (node_start,affected_node) in self.congested_arcs:
                                if self.congested_arcs[((node_start,affected_node))][1] > Minute_start:
                                    continue
                        

                            if depth[node_start] == 0:
                                factor = 1
                            
                            elif depth[node_start] == 1:
                                factor = 0.83
                            
                            elif depth[node_start] == 2:
                                factor = 0.78
                            
                            elif depth[node_start] == 3:
                                factor = 0.73

                            #print("La penalizacion pre factor", velocity_penalization)
                            velocity_penalization_for_depth = velocity_penalization/factor
                            #print(velocity_penalization_for_depth)
                            #print("La penalizacion pos factor", velocity_penalization_for_depth)
                            #velocity_penalization = 1-velocity_penalization
        
                            self.congested_arcs[(float(node_start), float(affected_node))] = [float(velocity_penalization_for_depth), float(Minute_start+state_time_elimination)]
    
    def create_random_unexpected_event_with_probability_and_2_nodes(self, Minute_start, probability_input, max_depth, lower_congestion_bound, upper_congestion_bound):
        if probability_input != 0:
            cont = 0
            for key in self.probability_of_event:
            #for key in self.node_start_averages:
                #probability_for_congestion = random.random()
                probability_for_congestion = np.random.uniform(0, 1)
                #print("probability_for_congestion", probability_for_congestion)
                if probability_for_congestion < self.probability_of_event[key]:
                    Node_start_congestion = key[0]
                    Node_end_congestion = key[1]
                    congestion_road = [Node_start_congestion, Node_end_congestion]
                    velocity_penalization = np.random.uniform(lower_congestion_bound, upper_congestion_bound)
                    #print("velocity_penalization", velocity_penalization)
                    #velocity_penalization = 0.1    
                    #velocity_penalization = 0.1
                    #velocity_penalization = 0.2
                    #velocity_penalization = random.uniform(0.01, 0.1)
                    #state_time_elimination = random.randint(30,self.congestion_max_duration)
                    state_time_elimination = np.random.uniform(30, self.congestion_max_duration)

                    self.congested_arcs[(float(Node_start_congestion), float(Node_end_congestion))] = [float(velocity_penalization), float(Minute_start+state_time_elimination)]

                    for node in congestion_road:
                        node_starts, depth = self.get_all_node_starts(node, 0, max_depth-1)
                        for node_start in node_starts:
                            connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                            
                            for affected_node in connected_nodes:
                                if node_start == affected_node:
                                    continue
                                
                                
                                if depth[node_start] == 0:
                                    factor = 1
                                
                                elif depth[node_start] == 1:
                                    factor = 0.83
                                
                                elif depth[node_start] == 2:
                                    factor = 0.78
                                
                                elif depth[node_start] == 3:
                                    factor = 0.73

                                velocity_penalization_for_depth = velocity_penalization/factor
                                if (node_start,affected_node) in self.congested_arcs:
                                    if self.congested_arcs[((node_start,affected_node))][1] > Minute_start:
                                        if self.congested_arcs[((node_start,affected_node))][0] <= velocity_penalization_for_depth:
                                            continue
                        

                                #print("La penalizacion pre factor", velocity_penalization)
                            
                                #print(velocity_penalization_for_depth)
                                #print("La penalizacion pos factor", velocity_penalization_for_depth)
                                #velocity_penalization = 1-velocity_penalization
            
                                self.congested_arcs[(float(node_start), float(affected_node))] = [float(velocity_penalization_for_depth), float(Minute_start+state_time_elimination)]

    def create_one_congestion(self,Minute_start, probability_input, max_depth, lower_congestion_bound, upper_congestion_bound, Node_start_congestion, Node_end_congestion):
        state_time_elimination = random.randint(30,60)
        velocity_penalization = random.uniform(lower_congestion_bound, upper_congestion_bound)
        congestion_road = [Node_start_congestion, Node_end_congestion]
        self.congested_arcs[(float(Node_start_congestion), float(Node_end_congestion))] = [float(velocity_penalization), float(Minute_start+state_time_elimination)]
        for node in congestion_road:
            node_starts, depth = self.get_all_node_starts(node, 0, max_depth-1)
            for node_start in node_starts:
                connected_nodes = self.travel_arc_information_dictionary.get(node_start, [])
                            
                for affected_node in connected_nodes:
                    if node_start == affected_node:
                        continue
                                
                                
                    if depth[node_start] == 0:
                        factor = 1
                                
                    elif depth[node_start] == 1:
                        factor = 0.83
                                
                    elif depth[node_start] == 2:
                        factor = 0.78

                    velocity_penalization_for_depth = velocity_penalization/factor
                    if (node_start,affected_node) in self.congested_arcs:
                        if self.congested_arcs[((node_start,affected_node))][1] > Minute_start:
                            if self.congested_arcs[((node_start,affected_node))][0] <= velocity_penalization_for_depth:
                                continue

            
                    self.congested_arcs[(float(node_start), float(affected_node))] = [float(velocity_penalization_for_depth), float(Minute_start+state_time_elimination)]

class shortest_path_memory:
    def __init__(self, environment):
        self.env = environment
        self.nodes = self.env.node_list
        self.shortest_paths = {}  
        self.load_paths_from_csv("all_shortest_paths.csv")
        del self.env
        del self.nodes
        
        # Diccionario para guardar los caminos más cortos
        #self.shortest_paths = defaultdict(list)

    def path_to_dictionary(self):
        for node in self.nodes:
            for client in self.nodes:
            #for client in self.nodes:
                key = (node,client)
                if key not in self.shortest_paths:
                    shortest_path = self.env.get_shortest_path(node, client)
                    #Calculamos el tiempo promedio de esa ruta
                    average_time_path, length = self.env.calculate_time_for_route(shortest_path)
                    # Guardar el camino en el diccionario con la tupla (node, client) como llave, entrega una lista
                    self.shortest_paths[(node, client)] = (shortest_path, average_time_path, length)
                    #self.shortest_paths[(node, client)] = shortest_path

    def get_shortest_path(self, node, client):
        # Método para obtener el camino más corto desde el diccionario
        return self.shortest_paths.get((node, client), None)

    def save_paths_to_csv(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node', 'Client', 'ShortestPath', 'AverageTime', 'Length'])
            for (node, client), (path, avg_time, length) in self.shortest_paths.items():
                # Asumiendo que el camino más corto es una lista de nodos, lo convertimos a string para guardarlo
                path_str = '->'.join(map(str, path))
                writer.writerow([node, client, path_str, avg_time, length])

    def load_paths_from_csv(self, file_path):
        self.shortest_paths = {}
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Salta la cabecera
            for row in reader:
                node, client, path_str, avg_time, length = row
                node = int(node)  # Convertir node a int
                client = int(client)  # Convertir client a int
                path = list(map(float, path_str.split('->')))  # Convertir cada elemento del camino a float
                self.shortest_paths[(node, client)] = (path, float(avg_time), float(length))#Clientgenerator con random depot

class ClientGenerator():
    def __init__(self, random_depot):
        #self.data_cal = data_calculations
        x = 1
        self.random_depot = random_depot

    def client_generator_function(self, random_seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time):
        random.seed(random_seed)
        #diff_TW : diferencia entre cota inferior y superior de ventana de tiempo
        #Number_clients: número de clientes
        self.clients = {}
        number_clients = int(random.gauss(mean_number_clients, 30))
        if number_clients < 60:
            number_clients = 60
        
        if mean_number_clients == 150:
            ratio_vehicles_to_clients = 28
        
        elif mean_number_clients == 250:
            ratio_vehicles_to_clients = 29

        self.client_list = random.sample(range(1, 1900), number_clients)
        #self.client_list.remove(self.random_depot)
        self.clients_list_2 = copy.deepcopy(self.client_list)

        if number_clients % ratio_vehicles_to_clients == 0:
            self.number_vehicles =  number_clients/ratio_vehicles_to_clients

        else:
            self.number_vehicles = int(number_clients/ratio_vehicles_to_clients) + 1
            #self.number_vehicles = round(number_clients/ratio_vehicles_to_clients)

        self.number_vehicles = int(self.number_vehicles)
        #self.client_list = random.sample(range(1, 750), number_clients)

        for key in self.client_list:
            early_tw = random.randint(horizon_start_time, horizon_end_time - diff_TW)
            late_tw = early_tw + diff_TW
            self.clients[key] = [early_tw, late_tw]
            #self.clients[key] = [0, 10000]

        #Agregamos el depot
        #self.client_list.insert(0, self.random_depot)
        #self.clients[self.random_depot] = [300,1200]

    def write_clients_to_file(self, seed_number):
        filename = f"clients_seed_{seed_number}_service_time_5.txt"
        with open(filename, 'w') as file:
            file.write("C104\n\n")
            file.write("VEHICLE\n")
            file.write("NUMBER    CAPACITY\n")  # Ajustado para tener cuatro espacios
            file.write(f"  {self.number_vehicles}      250\n\n")
            file.write("CUSTOMER\n")
            file.write("CUST NO.   XCOORD.   YCOORD.   DEMAND   READY TIME   DUE DATE   SERVICE TIME\n\n")  # Ajustado para tener cuatro espacios

            for i, client in enumerate(self.client_list):
                demand = 1
                service_time = 5
                early_tw, late_tw = self.clients[client]
                early_tw = early_tw -300
                late_tw = late_tw - 300
                #late_tw += 100
                xcoord = ycoord = 70  # Coordenadas estáticas para simplificar el ejemplo
                # Ajustes de formato para cada columna con cuatro espacios entre cada una
                file.write(f"{i:>5}    {xcoord:>4}    {ycoord:>7}    {demand:>7}    {early_tw:>7}    {late_tw:>7}    {service_time:>7}\n")

    def assign_clients_to_hyper_routes(self, random_seed, number_clients, diff_TW, horizon_start_time, horizon_end_time, route):
        #Generamos los clientes
        self.client_generator_function(random_seed, number_clients, diff_TW, horizon_start_time, horizon_end_time)
        self.real_route = [[] for _ in range(len(route))]
        self.client_dictionary = {}
        #Agregamos el depot
        self.client_dictionary[0]  = self.random_depot
        cont = 1
        for client in self.client_list:
            self.client_dictionary[cont] = client
            cont+=1

        #Luego, recorremos la ruta y asignamos los numeros reales
        for vehicle in range(len(route)):
            for node in route[vehicle]:
                real_node = self.client_dictionary[node]
                self.real_route[vehicle].append(real_node)

        return self.real_route
    
#Estado con random depot
class state():
    '''
    El agente necesita tener la capacidad de:
    1) Actualizar su posición
    2) Saber su posición
    3) Saber donde se puede mover
    4) cuantos vehículos posee el vrp
    5) tomar en cuenta la ruta que posee cada vehículo
    6) la accion vendria siendo a que nodo me muevo
    '''
    def __init__(self, number_vehicles, clients, n_arcs, horizon_start_time, random_depot):
        #Tiempo transcurrido del episodio
        self.tau_episode = horizon_start_time

        self.horizon_start_time = horizon_start_time

        #Posición vehículo: Especificamente, Cliente en que se encuentra saliendo cada vehículo
        self.vehicle_position = [random_depot for _ in range(number_vehicles)]

        #Clientes que faltan por visitar
        self.clients_not_visited = clients

        #Velocidad observada en los últimos n arcos del vehículo
        self.observed_velocity = [[0 for _ in range(n_arcs)] for _ in range(number_vehicles)]

        self.n_arcs = n_arcs

        self.terminal = False

        self.number_vehicles = number_vehicles

        self.vehicles_direction = [random_depot for _ in range(number_vehicles)]

        self.clients_arrival = {}

        self.total_vehicle_distance_travelled = {}


        self.vehicle_next_node = [random_depot for _ in range(number_vehicles)]

        for vehicle in range(number_vehicles):
            self.total_vehicle_distance_travelled[vehicle] = 0
        
        greedy_insertion_routes = None

        self.vehicle_completing_service = [random_depot for _ in range(number_vehicles)]

#Politica con random depot
class policy():
    def __init__(self, number_vehicles, static_routes, shortest_path_memory, client_generator, data_calculations, state, number_clients, epsilon, random_depot, congestion_lower_bound, congestion_upper_bound, number_actions_train, number_actions_test, learning_rate, W):
        self.number_vehicles = number_vehicles
        self.static_routes = static_routes
        self.spm = shortest_path_memory
        self.data_calculations = data_calculations

        self.local_rng = random.Random()
        self.local_rng_2 = random.Random()
        self.cont_1 = 0
        self.cont_2 = 0
        self.x = 0

        self.random_depot = random_depot

        self.o = 0

        #Guardamos clientes con ventanas de tiempo
        self.cg = client_generator

        #Penalización por llegar demasiado temprano a un cliente
        self.alpha = 0.1
        self.epsilon = epsilon

        #Creamos el vector W para la regresión
        self.action = []
        self.W = W
        self.number_clients = number_clients

        self.delay_cost_factor = 1
        #self.delay_cost_factor = 2
        self.distance_cost_factor = 1
        self.earliness_cost_factor = 0.1
        #self.earliness_cost_factor = 1
        self.overtime_cost = 5/6
        #self.overtime_cost = 2
        self.start_of_horizon = 300
        self.end_of_horizon = 780
        self.service_time = 5
        self.event_duration = 100
        
        self.number_actions_train = number_actions_train
        self.number_actions_test = number_actions_test

        self.learning_rate = learning_rate



        self.number_of_unexpected_events = {}

        self.change_client_dictionary_1 = {}
        self.change_client_dictionary_2 = {}
        self.change_client_dictionary_3 = {}

        self.episode_velocities = {}

        

        self.overtime_dictionary = {}
        for vehicle in range(self.number_vehicles):
            self.overtime_dictionary[vehicle] = [0,0]
        self.delay_dictionary = {}

        self.state = state

        self.congestion_lower_bound = congestion_lower_bound
        self.congestion_upper_bound = congestion_upper_bound

        self.delay_tw = []
        self.all_delay_tw = []

        self.sensibility_analysis_vector = []

        #self.routes = self.cheapest_inertion_route()



        #std_dev_depot_distance, mean_depot_distance, max_depot_distance, min_depot_distance = self.data_calculations.calculate_distance_metrics_to_depot(self.state.clients_not_visited)
        #std_dev_distance, mean_distance, self.max_distance, min_distance = data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)

        #print("La media de dispersion es:", mean_distance, "La desv estandar de distancia es:", std_dev_distance, "La distancia máxima al depot es: ", max_depot_distance)
        #self.mean_depot_distance_normalized = mean_depot_distance/std_dev_depot_distance
        #self.std_dev_distance_normalized = std_dev_distance/mean_distance
    
        #self.mean_depot_distance = mean_depot_distance
        #self.std_depot_distance = std_dev_depot_distance
        #self.max_depot_distance = max_depot_distance
        #self.mean_distance = mean_distance
        #self.std_dev_distance_normalized = std_dev_distance
        #self.max_distance = max_distance

        
        # Extraer las coordenadas de los nodos en la lista
       # coordenadas = [self.data_calculations.latitude_and_longitude[node] for node in self.state.clients_not_visited]

        # Convertir las coordenadas a un array de numpy para usar en KMeans
        #data = np.array(coordenadas)

        # Definir el número de clusters que queremos
        #k = self.number_vehicles


        #kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096)

        #kmeans.fit(data)

        #labels = kmeans.labels_

        #self.cluster_centers = kmeans.cluster_centers_


        #self.cluster_dictionary = {self.state.clients_not_visited[i]: labels[i] for i in range(self.number_clients)}
        
        


        #self.client_dictionary = {}
        #possible_actions = self.select_vehicle_possible_actions(number_vehicles)
        #for client in possible_actions[0]:
        #    if client == 0:
        #        pass
        #    else:
        #        self.action.append(client)

        self.action = [random.choice(self.state.clients_not_visited) for _ in range(self.number_vehicles)]

        
        
        self.monte_carlo_policy_test(state)


        #self.X_Normalization_List = norm_list

        #Es el parámetro del valor del estado de Q learning
        self.Q_real = 0






        #Para crear la politica estática podriamos hacer un diccionario que encuentra el valor para cada nodo. Es decir, nodo:accion.
        #Función: le paso un estado y me devuelve una acción.
        #Estática: Input: rutas y estado output acción
        #Estática: Necesita los clientes que falta, en base a los clientes que faltan toma la siguiente acción para cada vehículo. Si sabe que clientes ya visitó ya sabe a que cliente ir.

    def static_policy(self, state):
        self.state = state
        #Si el estado no es terminal se elige un acción para cada vehículo
        if self.state.terminal == False:
            for vehicle in range(self.number_vehicles):
                #Si no queda clientes por visitar, entonces el vehículo viaja al depot
                if len(self.state.clients_not_visited) == 0:
                    self.action[vehicle] = self.random_depot

                #Si le quedan dos nodos a la ruta son los depots
                elif len(self.static_routes[vehicle]) == 2:
                    self.action[vehicle] = self.random_depot

                #Si se encuentra el siguiente nodo dentro de los clientes no visitados, entonces se viaja a el
                elif self.static_routes[vehicle][1] in self.state.clients_not_visited:
                    self.action[vehicle] = self.static_routes[vehicle][1]

                #Si no se encuentra, entonces lo eliminamos y viajamos al próximo
                else:
                    #Eliminamos la posición 1
                    self.static_routes[vehicle].pop(1)
                    self.action[vehicle] = self.static_routes[vehicle][1]
            #print(self.action)
            return self.action

    def dynamic_policy(self, state):
        self.state = state
        if self.state.terminal == False:
            #Si estamos al inicio, generamos la primera acción para todo vehículo
            if self.state.tau_episode == self.state.horizon_start_time:
                self.generate_first_action_for_dynamic_policy()
                #Si no quedan clientes por visitar, entonces el vehículo viaja al depot
                return self.action

            #Si no quedan clientes por visitar, entonces devolvemos todos los vehículos al depot.
            elif len(self.state.clients_not_visited) == 0:
                for vehicle in range(self.number_vehicles):
                    self.action[vehicle] = 0
                return self.action

            #Quedan clientes por visitar, por lo que, se crean acciones para llegar a ellos
            else:
                self.client_list = copy.deepcopy(self.state.clients_not_visited)
                for vehicle in range(self.number_vehicles):
                    #Si no quedan clientes por visitar entonces mandamos el resto de los vehículos al depot
                    if len(self.client_list) == 0:
                        self.action[vehicle] = 0
                    else:
                        self.generate_dynamic_vehicle_action(vehicle)

                return self.action



    def generate_dynamic_vehicle_action(self, vehicle):
        prev_vehicle_action = self.action[vehicle]
        #Revisamos que la última velocidad no perteneza a una congestión
        vehicle_position = self.state.vehicle_position[vehicle]

        #Verificamos que aún no se llega al nodo que fue asignado
        if prev_vehicle_action in self.client_list:
            next_node = self.spm.shortest_paths[(vehicle_position , prev_vehicle_action)][0][1]
            tau_episode = math.trunc(self.state.tau_episode)
            average_velocity = self.data_calculations.travel_data[vehicle_position, next_node, tau_episode][1]
            change_of_speed = 0.5*average_velocity

            #Si ocurre que el cliente aun no ha sido visitado y y change of route es igual a 1,
            #entonces se deja la misma acción
            if self.change_of_route[vehicle] == 1:
                self.client_list.remove(prev_vehicle_action)


            #Sino, revisamos la útlima velocidad para ver si cambiamos el cliente
            elif self.state.observed_velocity[vehicle][self.state.n_arcs-1] < change_of_speed and len(self.client_list) > 1:
                #cambiamos la acción al cliente segundo más cercano
                self.client_list.remove(prev_vehicle_action)
                vehicle_action = self.get_best_penalty_client(vehicle_position)
                self.action[vehicle] = vehicle_action
                #print("cambiamos de acción porque la velocidad es: ", self.state.observed_velocity[vehicle][self.state.n_arcs-1])

                #Volvemos a añadir la acción a client_list
                self.client_list.append(prev_vehicle_action)

                #Lo cambiamos a 1 para que no vuelva a cambiar la acción
                self.change_of_route[vehicle] = 1



            #Sino, volvemos a ver cual es la mejor acción y la eliminamos de client list
            else:
                #Si ya visitamos el cliente, entonces le asignamos un 0 para
                # que pueda a volver a cambiar de ruta si hay un evento

                vehicle_action = self.get_best_penalty_client(vehicle_position)
                self.action[vehicle] = vehicle_action

        #Si no sigue en la lista, entonces se llegó al cliente o otro vehículo va en camino a ese cliente,
        #por lo que, hay que escoger otra acción
        else:
            self.change_of_route[vehicle] = 0
            vehicle_action = self.get_best_penalty_client(vehicle_position)
            self.action[vehicle] = vehicle_action



    def get_best_penalty_client(self, vehicle_position):
        penalty_comparison = math.inf
        for client in self.client_list:
            travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
            earliness_time_window = self.cg.clients[client][0]

            estimated_arrival_time = self.state.tau_episode + travel_time

            if estimated_arrival_time < earliness_time_window:
                penalty = self.alpha * (earliness_time_window-estimated_arrival_time) + (1-self.alpha) * travel_time

            else:
                #En el caso que no llegar temprano, quiere decir que no tiene penalización por llegar temprano
                penalty = (1-self.alpha) * travel_time

            if penalty < penalty_comparison:
                penalty_comparison = penalty
                vehicle_action = client

        #Lo removemos de la lista de clientes para que otros vehículos no tomen la decisión.
        self.client_list.remove(vehicle_action)

        return vehicle_action


    def generate_first_action_for_dynamic_policy(self):
        self.client_list = copy.deepcopy(self.state.clients_not_visited)
        self.action = []
        self.change_of_route = [0 for _ in range(self.number_vehicles)]
        for vehicle in range (self.number_vehicles):
            vehicle_action = self.get_best_penalty_client(0)
            self.action.append(vehicle_action)

    #############################################
    #Montecarlo policy functions
    
    def monte_carlo_policy_train(self, state):
        self.state = state

            #self.number_of_actions = self.number_vehicles +1
            #self.number_of_actions = self.number_vehicles*2
            #self.number_of_actions = 10
            #self.number_of_actions = self.number_vehicles+1
            
        self.number_of_actions = self.number_actions_train
            
            



        self.select_epsilon_greedy_action_train()
        #print("Tiempo episodio", self.state.tau_episode)


        

        #for vehicle in range(self.number_vehicles):
            #if self.action[vehicle] == self.state.vehicle_position[vehicle]:
                #print("xd")
       # print("La acción es", self.action)
        return self.action


    def monte_carlo_policy_test(self, state):
        self.state = state

            #self.number_of_actions = self.number_vehicles +1
            #self.number_of_actions = self.number_vehicles*2
            #self.number_of_actions = 10
            #self.number_of_actions = self.number_vehicles+1
            
        self.number_of_actions = self.number_actions_test
            
            



        self.select_epsilon_greedy_action_test()
        #self.compare(self.state, self.action)

        

        #for vehicle in range(self.number_vehicles):
            #if self.action[vehicle] == self.state.vehicle_position[vehicle]:
                #print("xd")
       # print("La acción es", self.action)
        #print(self.action)
        #print("valor q", self.min_q)
        #print("time", self.state.tau_episode)
        #print("Velocidades", self.state.observed_velocity)
        

        #if self.state.tau_episode == 320:
        #    print("Se escoge la acción", self.action)
        #print("tiempo del episodio", self.state.tau_episode)
        #print("Acción", self.action)
        #print("features", self.X_state_action)
        #print(self.action)
        return self.action
  
  
    """
    def select_vehicle_possible_actions(self, number_of_actions):
        possible_actions = []
        remaining_clients = {}
        for client in self.state.clients_not_visited:
            remaining_clients[client] = []

        for idx, vehicle_position in enumerate(self.state.vehicle_position):
            if vehicle_position == 0 and self.state.tau_episode > 320:
                possible_actions.append([0])

            elif len(self.state.clients_not_visited) == 0:
                possible_actions.append([0])

            else:
                travel_times = []
                for key in remaining_clients:
                    if self.cluster_dictionary[key] == idx:
                        travel_times.append((self.spm.shortest_paths[(vehicle_position, key)][1], key))

                #Obtenemos los clientes que poseen el menor travel time
                if len(travel_times) != 0:
                    top_actions = [client for _, client in heapq.nsmallest(number_of_actions, travel_times)]

                else:
                    top_actions = [0]

                possible_actions.append(top_actions)

        #Guardamos los clientes mas cercanos a los vehiculos.
        vehicle_to_clients = defaultdict(list)
        for client in clients:
            assigned_vehicle = None
            min_travel_time = float('inf')
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx

            if assigned_vehicle is not None:
                vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))

        #Si la llegada estimada del vehiculo mas cercano a un cliente podria sobrepasar la ventana de tiempo entonces lo unimos como posible acción
        for vehicle_idx, client_list in vehicle_to_clients.items():
            if len(possible_actions[vehicle_idx]) >= number_of_actions + 1:
                continue

            client_list.sort()
            for travel_time, client in client_list:
                if len(possible_actions[vehicle_idx]) >= number_of_actions + 1:
                    break  #Dejamos de añadir acciónes cuando ya pasamos el limite

                delay_tw = self.cg.clients[client][1]
                if travel_time + self.state.tau_episode >= delay_tw:
                    if client not in possible_actions[vehicle_idx]:
                        possible_actions[vehicle_idx].append(client)

        return possible_actions
    """
    

    """
    def select_vehicle_possible_actions(self, number_of_actions):
        #Como manejamos el agregar el 0?
        clients = self.state.clients_not_visited
        possible_actions = []
        for vehicle_position in self.state.vehicle_position:
            if vehicle_position == 0 and self.state.tau_episode > 350:
                possible_actions.append([0])

            elif len(self.state.clients_not_visited) == 0:
                possible_actions.append([0])
            
            else:
                travel_times = [
                    (self.spm.shortest_paths[(vehicle_position, client)][1], client) for client in clients]
                #Obtenemos los clientes que poseen el menor travel time
                top_actions = [client for _, client in heapq.nsmallest(number_of_actions, travel_times)]

                

                #Se añade el depot cuando el vehiculo se encuentra cerca de sobrepasar el horizonte de tiempo
                #if self.state.tau_episode + self.spm.shortest_paths[(vehicle_position, 0)][1] >  self.end_of_horizon-280:
                    #if self.state.vehicle_position.count(0) <= self.number_vehicles-2:
                        #top_actions.append(0)
                    
                #if len(self.state.clients_not_visited)-self.number_vehicles < 0:
                    #top_actions.append(0)
                
        
                possible_actions.append(top_actions)
        
        for vehicle in range(self.number_vehicles):
            if len(possible_actions[vehicle]) == 0:
                possible_actions[vehicle].append(0)
            
        #return possible_actions

        #Guardamos los clientes mas cercanos a los vehiculos.
        vehicle_to_clients = defaultdict(list)
        for client in clients:
            assigned_vehicle = None
            min_travel_time = float('inf')
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx
                
            if assigned_vehicle is not None:
                vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))
        
        #Si la llegada estimada del vehiculo mas cercano a un cliente podria sobrepasar la ventana de tiempo entonces lo unimos como posible acción
        for vehicle_idx, client_list in vehicle_to_clients.items():
            if len(possible_actions[vehicle_idx]) >= number_of_actions + 2:
                continue

            client_list.sort()
            for travel_time, client in client_list:
                if len(possible_actions[vehicle_idx]) >= number_of_actions + 2:
                    break  #Dejamos de añadir acciónes cuando ya pasamos el limite

                delay_tw = self.cg.clients[client][1]
                if travel_time + self.state.tau_episode >= delay_tw:
                    if client not in possible_actions[vehicle_idx]:
                        possible_actions[vehicle_idx].append(client
)
        
        return possible_actions
    """

    """
    def select_vehicle_possible_actions(self, number_of_actions):
        clients = set(self.state.clients_not_visited)
        possible_actions = [[] for _ in range(len(self.state.vehicle_position))]
        assigned_clients = set()

        # Crear un diccionario para almacenar el cliente más cercano para cada vehículo
        vehicle_to_clients = defaultdict(list)

        # Calcular el cliente más cercano para cada vehículo y almacenar la asignación
        for client in clients:
            min_travel_time = float('inf')
            assigned_vehicle = None

            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx

            if assigned_vehicle is not None:
                vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))

        # Asignar clientes a los vehículos según el cálculo anterior
        for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
            # Verificar si el vehículo está en el depot y el horizonte de tiempo está casi superado
            if vehicle_position == 0 and self.state.tau_episode > 350:
                possible_actions[vehicle_idx].append(0)
                continue

            # Obtener los clientes más cercanos asignados a este vehículo
            if vehicle_to_clients[vehicle_idx]:
                top_actions = [client for _, client in heapq.nsmallest(number_of_actions, vehicle_to_clients[vehicle_idx])]
                possible_actions[vehicle_idx].extend(top_actions)
                assigned_clients.update(top_actions)

            else:
                # Si no hay clientes disponibles, la única acción posible es quedarse.
                possible_actions[vehicle_idx].append(0)

            if self.spm.shortest_paths[(vehicle_position, 0)][1] + self.state.tau_episode > self.end_of_horizon - 60:
                if 0 not in possible_actions[vehicle_idx]:
                    possible_actions[vehicle_idx].append(0)

        # Si los vehículos están en la misma posición, asignar las mismas acciones
        vehicle_positions = defaultdict(list)
        for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
            vehicle_positions[vehicle_position].append(vehicle_idx)

        for vehicle_list in vehicle_positions.values():
            if len(vehicle_list) > 1:
                common_actions = possible_actions[vehicle_list[0]]
                for idx in vehicle_list:
                    possible_actions[idx] = common_actions


        return possible_actions
    """
    """
    def select_epsilon_greedy_action(self):
        random_number = random.random()
        if random_number > self.epsilon:
            #Generamos una acción en base a nuestra regresión
            self.generate_best_approximate_action()
        else:
            #Generamos una acción aleatoria para cada vehículo
            possible_actions = self.select_vehicle_possible_actions(self.number_of_actions)
            for vehicle in range(self.number_vehicles):
                number_of_possible_actions = len(possible_actions[vehicle])
                random_action = random.randint(0, number_of_possible_actions-1)
                self.action[vehicle] = possible_actions[vehicle][random_action]
    """
    def select_epsilon_greedy_action_test(self):
        self.clasify_delayed_clients()
        self.extract_general_state_features()

        for vehicle in range(self.number_vehicles):
            self.possible_actions = list(self.select_vehicle_possible_actions(self.number_of_actions, vehicle))
            #for i in self.possible_actions:
                #if i not in self.state.clients_not_visited:
                #    print(i)
            self.generate_best_Q_pred_for_1_vehicle(vehicle)

    def select_epsilon_greedy_action_train(self):
        self.clasify_delayed_clients()
        self.extract_general_state_features()
        #Primero de que self.action es factible.
        for vehicle in range(self.number_vehicles):
            self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions, vehicle)
            if self.action[vehicle] not in self.possible_actions:
                random_choice = self.local_rng_2.choice(self.possible_actions)
                self.action[vehicle] = random_choice

        
        for vehicle in range(self.number_vehicles):
            #self.possible_actions = list(self.select_vehicle_possible_actions(self.number_of_actions, vehicle))
            self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions, vehicle)
            if self.local_rng.random() < self.epsilon:
                # Exploración: seleccionamos una acción aleatoria
                self.action[vehicle] = random.choice(self.possible_actions)
            else:
                self.generate_best_Q_pred_for_1_vehicle(vehicle)

    
    def select_epsilon_greedy_action_train_2(self):
        self.clasify_delayed_clients()
        self.extract_general_state_features()
        for vehicle in range(self.number_vehicles):
            #self.possible_actions = list(self.select_vehicle_possible_actions(self.number_of_actions, vehicle))
            self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions, vehicle)
            if self.local_rng.random() < self.epsilon:
                # Exploración: seleccionamos una acción aleatoria
                self.action[vehicle] = random.choice(self.possible_actions)
            else:
                self.generate_best_Q_pred_for_1_vehicle(vehicle)

    def select_epsilon_greedy_action_test_2(self):
        self.clasify_delayed_clients()
        self.extract_general_state_features()
        
        for vehicle in range(self.number_vehicles):
            self.possible_actions = list(self.select_vehicle_possible_actions(self.number_of_actions, vehicle))
            #self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions, vehicle)
            #if random.random() < self.epsilon:
                # Exploración: seleccionamos una acción aleatoria
            #    self.action[vehicle] = random.choice(self.possible_actions)
            #else:
            self.generate_best_Q_pred_for_1_vehicle(vehicle)
        


    def generate_best_Q_pred_for_1_vehicle(self, vehicle):
        # Explotación: seleccionamos la mejor acción según Q-pred
        min_q_value = float('inf')
        #self.min_q = 0
        best_client = 0
        for client in self.possible_actions:
            current_action = self.action.copy()  # Crear una copia de la acción actual
            current_action[vehicle] = client
            self.extract_state_action_features(current_action)
            X = list(itertools.chain(self.X_general_state, self.X_state_action))
            if self.W is None:
                self.create_W(len(X))
                        
            q_value = np.dot(X, self.W)
            if q_value < min_q_value:
                min_q_value = q_value
                #self.min_q = min_q_value
                #print("Mejor valor", q_value)
                best_client = client

        #print("Action", self.action)
        #print("Q value", q_value)

        self.action[vehicle] = best_client

    
    def fourier_features(self,x, num_freqs=2):
        features = []
        for n in range(1, num_freqs + 1):
            features.append(np.sin(2 * np.pi * n * x))
            features.append(np.cos(2 * np.pi * n * x))
        return features

    def extract_general_state_features_oficial(self):
        self.X_general_state = []
        #Agregamos una constante 1 para cumplir la funcion afin.
        #1
        self.X_general_state.append(1)
        #self.X_general_state.extend(self.fourier_features(1))
        #2
        #Agregamos el tiempo del episodio
        tau_episode = self.state.tau_episode
        #Normalización
        time_left = (1150-tau_episode)/(850)
        self.X_general_state.append(time_left)
        #self.X_general_state.extend(self.fourier_features(time_left))
        clients_left = len(self.state.clients_not_visited)/150
        
        if time_left <1:
            episode_finalized = 1
            self.X_general_state.append(episode_finalized)
            #self.X_general_state.append(episode_finalized*clients_left)
            #self.X_general_state.extend(self.fourier_features(episode_finalized))

        
        else:
            episode_finalized = 0
            self.X_general_state.append(episode_finalized)
            #self.X_general_state.append(0)


        self.X_general_state.append(self.mean_depot_distance_normalized)
        self.X_general_state.append(self.std_dev_distance_normalized)
        self.X_general_state.append(self.std_dev_distance_normalized*self.mean_depot_distance_normalized)
        #if len(self.state.clients_not_visited) > 2:
        #    std_dev_depot_distance, mean_depot_distance, max_depot_distance, min_depot_distance = self.data_calculations.calculate_distance_metrics_to_depot(self.state.clients_not_visited)
        #    std_dev_distance, mean_distance, max_distance, min_distance = data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)

            
        #    if mean_depot_distance != 0:
        #        self.X_general_state.append(mean_depot_distance/std_dev_depot_distance)
            
        #    else:
        #        self.X_general_state.append(0)
            
        #    if mean_distance !=0:
        #        self.X_general_state.append(std_dev_distance/mean_distance)
            
        #    else:
        #        self.X_general_state.append(0)


        
        #else:
        #    self.X_general_state.append(0)
        #    self.X_general_state.append(0)
        
        

           # self.X_general_state.extend(self.fourier_features(episode_finalized))


        #if len(self.state.clients_not_visited)>2:
        #    std, mean, max_distance, min_distance = self.data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)
        #    self.X_general_state.append((mean-min_distance)/(max_distance-min_distance))
        #    self.X_general_state.append((std-min_distance)/(max_distance-min_distance))
        
        #else:
        #    self.X_general_state.append(0)
        #    self.X_general_state.append(0)


        #delay_costs = 0
        #earliness_costs = 0

        #3
        #Sumamos los costos de delay que ya hay
        self.already_late_clients = []
        for client in self.state.clients_not_visited:
            client_due_time = self.cg.clients[client][1]
            if tau_episode > client_due_time:
                self.already_late_clients.append(client)

        """
        vehicle_client_dict = {}
        total_cost = [0 for _ in range(4)]
        for vehicle in range(self.number_vehicles):
            vehicle_client_dict[vehicle] = [0,0]

        cont_earliness_vehicle = [0 for _ in range(self.number_vehicles)]
        cont_delay_vehicle = [0 for _ in range(self.number_vehicles)]


        for key in self.state.clients_arrival:
            client_earliness_time, client_due_time = self.cg.clients[key]
            max_earliness = float('-inf')
            max_delay = float('-inf')
            if self.state.clients_arrival[key][0] < client_earliness_time:
                earliness = (client_earliness_time - self.state.clients_arrival[key][0]) * self.earliness_cost_factor
                max_earliness = max(max_earliness, earliness)
                
            elif client_due_time < self.state.clients_arrival[key][0]:
                delay = (self.state.clients_arrival[key][0] - client_due_time) * self.delay_cost_factor
                max_delay = max(max_delay, delay)
        
        cont_delay = 0
        cont_earliness = 0
        for key in self.state.clients_arrival:
            client_earliness_time, client_due_time = self.cg.clients[key]

            if self.state.clients_arrival[key][0] < client_earliness_time:
                if max_earliness > 0:
                    vehicle_client_dict[self.state.clients_arrival[key][1]][0] += ((client_earliness_time-self.state.clients_arrival[key][0])*self.earliness_cost_factor)/max_earliness
                cont_earliness += 1
            
            elif client_due_time < self.state.clients_arrival[key][0]:
                if max_delay > 0:
                    vehicle_client_dict[self.state.clients_arrival[key][1]][1] += ((self.state.clients_arrival[key][0]-client_due_time)* self.delay_cost_factor)/max_delay
                cont_delay += 1
       
        if cont_earliness !=0:
            #self.X_general_state.append(vehicle_client_dict[self.state.clients_arrival[key][1]][0]/cont_earliness)
            total_cost.append(vehicle_client_dict[self.state.clients_arrival[key][1]][0]/cont_earliness)

        else:
            #self.X_general_state.append(0)
            total_cost.append(0)

        
        if cont_delay !=0:
            #self.X_general_state.append(vehicle_client_dict[self.state.clients_arrival[key][1]][1]/cont_delay)
            total_cost.append(vehicle_client_dict[self.state.clients_arrival[key][1]][1]/cont_delay)


        else:
            #self.X_general_state.append(0)
            total_cost.append(0)


        #total_cost_per_vehicle = [0 for _ in range(self.number_vehicles)]
        #for key in vehicle_client_dict:
            #total_cost += (vehicle_client_dict[key][0]*self.earliness_cost_factor) + (vehicle_client_dict[key][1]*self.delay_cost_factor)
        #    total_cost_per_vehicle[key] += vehicle_client_dict[key][0]


        #    total_cost_per_vehicle[key] += vehicle_client_dict[key][1]




        for vehicle in range(self.number_vehicles):
            if self.state.tau_episode > self.end_of_horizon and self.overtime_dictionary[vehicle][1] == 0:
                self.overtime_dictionary[vehicle][0] = (self.state.tau_episode-self.end_of_horizon)*self.overtime_cost

            if self.state.vehicle_position == 0 and self.state.tau_episode > 320 and self.overtime_dictionary[vehicle][1] == 0:
                self.overtime_dictionary[vehicle][1] = 1

            #total_cost_per_vehicle[vehicle] += self.overtime_dictionary[vehicle][0]
        max_overtime = max(self.overtime_dictionary[vehicle][0] for vehicle in range(self.number_vehicles))


        total_overtime = 0
        cont_overtime = 0
        if max_overtime != 0:
            for vehicle in range(self.number_vehicles):
                total_overtime += self.overtime_dictionary[vehicle][0]/max_overtime
                cont_overtime += 1
        
        if cont_overtime!= 0:
            #self.X_general_state.append(total_overtime/cont_overtime)
            total_cost.append(total_overtime/cont_overtime)
        else:
            #self.X_general_state.append(0)
            total_cost.append(0)


        max_distance = 0
        for vehicle in range(self.number_vehicles):
            distance = self.state.total_vehicle_distance_travelled[vehicle]*self.distance_cost_factor
            if max_distance < distance:
                max_distance = distance

        total_distance = 0
        for vehicle in range(self.number_vehicles):
            if max_distance != 0:
                total_distance += self.state.total_vehicle_distance_travelled[vehicle]*self.distance_cost_factor/max_distance
        
        #self.X_general_state.append(total_distance/self.number_vehicles)
        total_cost.append(total_distance/self.number_vehicles)

        mean = np.mean(total_cost)
        std = np.std(total_cost)

        self.X_general_state.append(mean)
        self.X_general_state.append(std)
        self.X_general_state.append(mean*time_left)
        self.X_general_state.append(std*time_left)
        number_clients_normalized = len(self.state.clients_not_visited)/150
        self.X_general_state.append(mean*number_clients_normalized)
        self.X_general_state.append(std*number_clients_normalized)
        """

        #Añadimos clientes a los que ya estamos llegando tarde
        """
        delay_not_visited_clients = 0
        cont = 0
        for client in self.state.clients_not_visited:
            client_due_time = self.cg.clients[client][1]
            if client_due_time < self.state.tau_episode:
                #Este servía antes
                #Se calcula el vehículo más cercano
                best_travel_time = float('inf')
                worst_travel_time = -1
                for vehicle in range(self.number_vehicles):
                    travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1]
                    if travel_time < best_travel_time:
                        best_travel_time = travel_time
                    
                    if travel_time > worst_travel_time:
                        worst_travel_time = travel_time

                delay_not_visited_clients += (worst_travel_time+self.state.tau_episode-client_due_time)*self.delay_cost_factor/((worst_travel_time-best_travel_time+self.state.tau_episode-client_due_time)*self.delay_cost_factor)
                cont += 1

        if cont != 0:
            delay_not_visited_clients = delay_not_visited_clients/cont
            self.X_general_state.append(delay_not_visited_clients)
            #self.X_general_state.extend(self.fourier_features(delay_not_visited_clients))

        
        else:
            delay_not_visited_clients = 0
            self.X_general_state.append(0)
            #self.X_general_state.extend(self.fourier_features(0))
        """
        #4
        # Numero de vehiculos activos
        #active_vehicles = sum(1 for pos in self.state.vehicle_position if pos != 0 or self.state.tau_episode <310)
        #print(active_vehicles)
        #active_vehicles_normalized = active_vehicles/ self.number_vehicles
        #self.X_general_state.append(active_vehicles_normalized)
        #self.X_general_state.extend(self.fourier_features(active_vehicles_normalized))

        #Este servía antes
        #self.X_general_state.append((len(self.state.clients_not_visited)/20)*active_vehicles_normalized*0.04)
        #Normalizamos
        number_clients_normalized = len(self.state.clients_not_visited)/150
        #self.X_general_state.append(number_clients_normalized*active_vehicles_normalized)
        #self.X_general_state.extend(self.fourier_features(number_clients_normalized*active_vehicles_normalized))


        #self.X_general_state.append(time_left*number_clients_normalized)
        #self.X_general_state.extend(self.fourier_features(time_left*number_clients_normalized))


        #Agregamos raiz cuadrada del numero de clientes que faltan por visitar
        #5

        self.X_general_state.append(np.sqrt(number_clients_normalized))
        #self.X_general_state.extend(self.fourier_features(np.sqrt(number_clients)/np.sqrt(150)))

        

        

        #6
        #Feature de numero clientes y el tiempo del episodio
        #Esto servia antes
        #clients_normalized = len(self.state.clients_not_visited)/150
        #number_clients_tau_feature = clients_normalized/(max(1150-tau_episode, 1))

        #self.X_general_state.append(number_clients_tau_feature/60)

        #Normalizamos
        #self.X_general_state.append(number_clients_tau_feature)
        #self.X_general_state.extend(self.fourier_features(number_clients_tau_feature))

        



        #Agregamos el promedio de las ultimas velocidades observadas por vehículo
        mean_velocity = 0
        self.mean_velocities = []
        for vehicle_velocities in self.state.observed_velocity:
            for velocity in vehicle_velocities:
                mean_velocity += velocity

            mean_velocty = mean_velocity/len(vehicle_velocities)
            #self.X_general_state.append(mean_velocty)
            self.mean_velocities.append(mean_velocty)
            #self.X_general_state.append(mean_velocty)


        self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions)

        #Creamos un diccionar rankeando los peores casos para cada vehículo dentro de las posibles acciones
        #Guarda el peor travel time, el peor delay y el peor earliness, en ese orden.
        self.worst_case_scenario_dict = {}
        self.best_case_scenario_dict = {}
        #print("Las acciones de los vehiculos son", self.possible_actions)
        
        for vehicle in range(self.number_vehicles):
            worst_earliness = 0
            worst_delay = 0
            worst_travel_time = 0
            best_earliness = float("inf")
            best_delay = float("inf")
            best_travel_time = float("inf")
            for action in self.possible_actions[vehicle]:
                est_arrival = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action)][1] + self.state.tau_episode
                if self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action)][1] > worst_travel_time:
                    worst_travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action)][1]
                
                if self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action)][1] < best_travel_time:
                    best_travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action)][1]


                if action!= 0:
                    earliness_tw, delay_tw = self.cg.clients[action]
                    if est_arrival < earliness_tw:
                        earliness_cost = (earliness_tw-est_arrival)*self.earliness_cost_factor
                        if earliness_cost > worst_earliness:
                            worst_earliness = earliness_cost
                        
                        if earliness_cost < best_earliness:
                            best_earliness = earliness_cost

                    elif est_arrival > delay_tw:
                        delay_cost = (est_arrival-delay_tw)*self.delay_cost_factor
                        if delay_cost > worst_delay:
                            worst_delay = delay_cost
                        
                        if delay_cost < best_delay:
                            best_delay = delay_cost
            if best_delay == float("inf"):
                best_delay = 0
            
            if best_earliness == float("inf"):
                best_earliness = 0
            
            if best_travel_time == float("inf"):
                best_travel_time = 0

            self.worst_case_scenario_dict[vehicle] = (worst_travel_time, worst_delay, worst_earliness)
            self.best_case_scenario_dict[vehicle] = (best_travel_time, best_delay, best_earliness )
        



        """
        vehicle_positions = self.state.vehicle_position
        vehicle_distance_sum = 0
        vehicle_distance_count = 0
        for i in range(len(vehicle_positions)):
            for j in range(i + 1, len(vehicle_positions)):
                distance = self.spm.shortest_paths[(vehicle_positions[i], vehicle_positions[j])][2]
                vehicle_distance_sum += distance
                vehicle_distance_count += 1
        if vehicle_distance_count > 0:
            average_vehicle_distance = vehicle_distance_sum / vehicle_distance_count
            average_vehicle_distance_normalized = average_vehicle_distance  # Normalizar si es necesario
        else:
            average_vehicle_distance_normalized = 0

        average_vehicle_distance_normalized = (average_vehicle_distance_normalized)
        self.X_general_state.append(average_vehicle_distance_normalized)  # Añadimos la nueva característica


        #Agregamos los menores travel times a todo cliente y luego agregamos el promedio
        travel_time_dict = {}
        min_travel_time = float('inf')
        for client in self.state.clients_not_visited:
            for idx, vehicle_position in enumerate(self.state.vehicle_position):
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                distance = self.spm.shortest_paths[(vehicle_position, client)][2]
                if travel_time < min_travel_time:
                    travel_time_dict[client] = (travel_time, distance, idx)

        average_travel_time_to_client = 0
        min_estimated_delay = 0
        average_distance_vehicle_to_client = 0
        min_estimated_overtime = 0
        overtime_dict = {}
        for key in travel_time_dict:
            average_travel_time_to_client+=travel_time_dict[key][0]
            average_distance_vehicle_to_client += travel_time_dict[key][1]

            client_due_time = self.cg.clients[key][1]
            min_est_arrival_time = travel_time_dict[key][0] + self.state.tau_episode
            if min_est_arrival_time > client_due_time:
                min_estimated_delay += (min_est_arrival_time-client_due_time)*self.delay_cost_factor

            if min_est_arrival_time > self.end_of_horizon:
                vehicle = travel_time_dict[key][2]
                overtime_cost = (min_est_arrival_time - self.end_of_horizon)*self.overtime_cost
                if vehicle not in overtime_dict:
                    overtime_dict[vehicle] = overtime_cost

                else:
                    if overtime_cost > overtime_dict[vehicle]:
                        overtime_dict[vehicle] = overtime_cost


        if average_travel_time_to_client != 0:
            average_travel_time_to_client = average_travel_time_to_client/len(travel_time_dict)
            average_distance_vehicle_to_client = average_distance_vehicle_to_client/len(travel_time_dict)

        for key in overtime_dict:
            min_estimated_overtime += overtime_dict[key]

        #8
        self.X_general_state.append(average_travel_time_to_client)
        #9
        self.X_general_state.append(min_estimated_delay)
        #10
        self.X_general_state.append(min_estimated_overtime)
        """

    def extract_general_state_features_routes(self):
        
        self.X_general_state = []
        #self.X_general_state.append(0.1)
        

        #1
        #self.X_general_state.append(0.1)
        clients_left = len(self.state.clients_not_visited)/150


        if clients_left != 0:
            time_left = (1150-self.state.tau_episode)/850
            time = (self.state.tau_episode-300)/850
        
        else:
            time_left = 0
            time = 0
        
        number_vehicles = 0
        if self.state.tau_episode > self.end_of_horizon:
            for vehicle in range(self.number_vehicles):
                if self.state.vehicle_position[vehicle] != 0:
                    number_vehicles += 1
        
        
        number_vehicles_normalized = number_vehicles/self.number_vehicles
        self.X_general_state.append(number_vehicles_normalized)


        #2
        self.X_general_state.append(time_left)
        #self.X_general_state.append(self.fourier_features(time_episode))


        #3
        self.X_general_state.append(np.sqrt(clients_left))

        self.X_general_state.append(number_vehicles_normalized*clients_left)

        self.X_general_state.append(clients_left*time)

        self.X_general_state.append(np.sqrt(clients_left/self.number_vehicles))
    
        mean_distance, std, max_distance, min_distance = self.data_calculations.calculate_distance_metrics_to_depot(self.state.clients_not_visited)
        #std, mean, max_distance, min_distance = self.data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)
        
        total_distance = mean_distance*len(self.state.clients_not_visited)
        self.X_general_state.append(total_distance/2300)
        self.X_general_state.append(std/5)
        
        #self.X_general_state.append(mean/10)
        
        #Estos sirven
        #self.X_general_state.append(std/5)
        #self.X_general_state.append(std*clients_left/5)
        
        #self.X_general_state.append(std)
        #self.X_general_state.append(std*clients_left)



        #client_earliness_tw= []
        client_earliness_value = []
        client_delay_value = []
        self.already_late_clients = []
        number_of_already_late_clients = 0
        for client in self.state.clients_not_visited:
            client_earliness, client_due_time = self.cg.clients[client]
            if self.state.tau_episode > client_due_time:
                self.already_late_clients.append(client)
                number_of_already_late_clients += 1
            
            client_earliness_value.append(client_earliness)
            client_delay_value.append(client_due_time)

        number_of_already_late_clients_normalized = number_of_already_late_clients/60
        self.X_general_state.append(number_of_already_late_clients_normalized)
        #self.X_general_state.append(number_of_already_late_clients_normalized*std/5)




        client_counts_earliness = [0 for _ in range(4)]
        client_counts_delay = [0 for _ in range(4)]


        for i in client_earliness_value:
            if i < 400:
                client_counts_earliness[0] += 1
            elif 400 <= i < 500:
                 client_counts_earliness[1] += 1
            elif 500 <= i < 600:
                 client_counts_earliness[2]+= 1
            elif 600 <= i < 700:
                 client_counts_earliness[3] += 1
        


        for count in client_counts_earliness:
            self.X_general_state.append(count/150)

        #for count in client_counts_earliness:
            #self.X_general_state.append(count*time_left/15)
        
        
        #Agregamos el promedio de las ultimas velocidades observadas por vehículo
        self.mean_velocities = []            
        for vehicle_velocities in self.state.observed_velocity:
            mean_velocity = 0
            for velocity in vehicle_velocities:
                mean_velocity += velocity

            mean_velocity = mean_velocity/len(vehicle_velocities)
            #if mean_velocity < 0.06 and self.state.tau_episode > 310:
                
            self.mean_velocities.append(mean_velocity)
            if mean_velocity < 0.06 and self.state.tau_episode > 308:
                self.X_general_state.append(1)
            else:
                self.X_general_state.append(0)
    
        


    def extract_general_state_features(self):
        self.X_general_state = []

        #1
        #self.X_general_state.append(0.1)
        clients_left = len(self.state.clients_not_visited)/150


        
        if clients_left != 0:
            time_left = (1150-self.state.tau_episode)/(850)
            time = (self.state.tau_episode-300)/850
            active_vehicles = 0



        
        else:
            time_left = 0
            time = 0
            number_vehicles_normalized = 0
            active_vehicles_normalized = 0



        #2
        #self.X_general_state.append(time_left)
        #print("Time_left es", time_left)
        #print("Clients_left es", clients_left)
        #self.X_general_state.append(self.fourier_features(time_episode))


        #3
        self.X_general_state.append(np.sqrt(clients_left))
        #self.X_general_state.append(clients_left)


        ##Cuadraticosss
        self.X_general_state.append(time_left)
        self.X_general_state.append(time_left**2)
        self.X_general_state.append(clients_left**2)
        self.X_general_state.append((clients_left**2)*time)
        self.X_general_state.append((time**2)*clients_left)
        self.X_general_state.append((time**2)*(clients_left**2))

        #6
    
        #mean_distance, std, max_distance, min_distance = self.data_calculations.calculate_distance_metrics_to_depot(self.state.clients_not_visited)
        #std, mean, max_distance, min_distance = self.data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)
        
        #total_distance = mean_distance*len(self.state.clients_not_visited)
        #total_distance_normalized = total_distance/2100

        #7
        #self.X_general_state.append(total_distance_normalized)
        #8
        #self.X_general_state.append(min_distance/12.9)
        #9
        #self.X_general_state.append(max_distance/29)


        
    
        #client_earliness_tw= []

        #number_of_already_late_clients_normalized = number_of_already_late_clients/50


        client_earliness_value = []
        client_delay_value = []
        self.already_late_clients = []
        number_of_already_late_clients = 0
        for client in self.state.clients_not_visited:
            client_earliness, client_due_time = self.cg.clients[client]
            
            client_earliness_value.append(client_earliness)

            client_delay_value.append(client_due_time)
        
        client_counts_earliness = [0 for _ in range(4)]
        client_counts_delay = [0 for _ in range(4)]

        for i in client_earliness_value:
            if i < 400 and self.state.tau_episode < 400:
                client_counts_earliness[0] += 1
            elif 400 <= i < 500 and self.state.tau_episode < 500:
                client_counts_earliness[1] += 1
            elif 500 <= i < 600 and self.state.tau_episode < 600:
                 client_counts_earliness[2]+= 1
        
        for count in client_counts_earliness:
            self.X_general_state.append(count/self.number_clients)
        
        
        time_left_for_earliness = (580-self.state.tau_episode)/(280)
        mean_earliness_diff = 0
        if time_left_for_earliness > 0:
           #Calculamos el promedio
            mean_earliness = 0
            for i in client_earliness_value: 
                mean_earliness += i
            
            if len(self.state.clients_not_visited) != 0:
                mean_earliness = mean_earliness/len(self.state.clients_not_visited)
                if mean_earliness > self.state.tau_episode:
                    mean_earliness_diff = (mean_earliness-self.state.tau_episode)/120


        
        #mean_delay = 0
        #mean_delay_diff = 0
        #for x in client_delay_value:
        #    mean_delay += x

        #if len(self.state.clients_not_visited) >0 :
        #    mean_delay = mean_delay/len(self.state.clients_not_visited)
        
        #mean_delay_diff = (mean_delay-self.state.tau_episode)/350

        
        self.X_general_state.append(mean_earliness_diff)
        #self.X_general_state.append(mean_delay_diff*len(self.state.clients_not_visited)/self.number_clients)

                    


        
        #Agregamos el promedio de las ultimas velocidades observadas por vehículo
        self.mean_velocities = []
        for vehicle_velocities in self.state.observed_velocity:
            mean_velocity = 0
            for velocity in vehicle_velocities:
                mean_velocity += velocity

            mean_velocity = mean_velocity/len(vehicle_velocities)
            #if mean_velocity < 0.06 and self.state.tau_episode > 310:
                
            self.mean_velocities.append(mean_velocity)

            #if mean_velocity < 0.06 and self.state.tau_episode > 308:
            #    self.X_general_state.append(1)
            #else:
                #self.X_general_state.append(0)
        
        #for vehicle in range(self.number_vehicles):
            #self.mean_velocities.append(self.state.observed_velocity[vehicle][2])
            #if self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != self.random_depot and self.state.vehicles_direction[vehicle] != self.state.vehicle_position[vehicle] and self.state.vehicle_position[vehicle] != self.random_depot:
            #    next_node = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.state.vehicles_direction[vehicle])][0][1]
            #    state_tau = int(self.state.tau_episode)
            #    key = (self.state.vehicle_position[vehicle], next_node, state_tau)
            #    avg_velocity = self.data_calculations.travel_data[key][1]
            #    speed_diff_normalized = self.mean_velocities[vehicle] - avg_velocity
             #   self.X_general_state.append(speed_diff_normalized)
            
            #else:
             #   self.X_general_state.append(0)





        self.clasify_delayed_clients()

        #print(len(self.X_general_state))
        #self.X_general_state = [0]*16

        #print(self.X_general_state)


        
        #mean_distance_to_late_clients = 0
        #cont = 0
        #for vehicle in range(self.number_vehicles):
        #    for tuple in self.vehicle_to_clients[vehicle]:
        #        client = tuple[1]
        #        delay_tw = self.cg.clients[client][1]
        #        if delay_tw < self.state.tau_episode:
        #            mean_distance_to_late_clients += self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][2]
        #            cont += 1
        
        #if cont != 0:
        #    mean_distance_to_late_clients = mean_distance_to_late_clients/cont
        
        #self.X_general_state.append(mean_distance_to_late_clients/15)
        


    def extract_general_state_features_fourier(self):
        self.X_general_state = []

        #1
        #self.X_general_state.append(0.1)
        clients_left = len(self.state.clients_not_visited)/150


        
        if clients_left != 0:
            time_left = (1150-self.state.tau_episode)/(850)
            time = (self.state.tau_episode-300)/850
            active_vehicles = 0



        
        else:
            time_left = 0
            time = 0
            number_vehicles_normalized = 0
            active_vehicles_normalized = 0



        #2
        #self.X_general_state.append(time_left)
        #print("Time_left es", time_left)
        #print("Clients_left es", clients_left)
        #self.X_general_state.append(self.fourier_features(time_episode))


        #3
        self.X_general_state.extend(self.fourier_features(np.sqrt(clients_left)))
        self.X_general_state.extend(self.fourier_features(time_left))
        self.X_general_state.extend(self.fourier_features(clients_left * time_left))


        #self.X_general_state.append(number_vehicles_normalized*clients_left)


        #self.X_general_state.append(np.sqrt(clients_left/self.number_vehicles))
    
        mean_distance, std, max_distance, min_distance = self.data_calculations.calculate_distance_metrics_to_depot(self.state.clients_not_visited)
        #std, mean, max_distance, min_distance = self.data_calculations.calculate_nodes_caracteristics(self.state.clients_not_visited)
        
        total_distance = mean_distance*len(self.state.clients_not_visited)
        total_distance_normalized = total_distance/2100
        #total_distance_squared = total_distance_normalized**2
        self.X_general_state.extend(self.fourier_features(total_distance_normalized))
        self.X_general_state.extend(self.fourier_features(min_distance/12.9))
        self.X_general_state.extend(self.fourier_features(max_distance/29))

        
    
        #client_earliness_tw= []

        #number_of_already_late_clients_normalized = number_of_already_late_clients/50


        client_earliness_value = []
        client_delay_value = []
        self.already_late_clients = []
        number_of_already_late_clients = 0
        for client in self.state.clients_not_visited:
            client_earliness, client_due_time = self.cg.clients[client]
            
            client_earliness_value.append(client_earliness)

            client_delay_value.append(client_due_time)
        
        client_counts_earliness = [0 for _ in range(4)]
        client_counts_delay = [0 for _ in range(4)]

        for i in client_earliness_value:
            if i < 400 and self.state.tau_episode < 400:
                client_counts_earliness[0] += 1
            elif 400 <= i < 500 and self.state.tau_episode < 500:
                client_counts_earliness[1] += 1
            elif 500 <= i < 600 and self.state.tau_episode < 600:
                 client_counts_earliness[2]+= 1
        
        for count in client_counts_earliness:
            #self.X_general_state.append(count/self.number_clients)
            self.X_general_state.extend(self.fourier_features(count/self.number_clients))

        
        time_left_for_earliness = (580-self.state.tau_episode)/(280)
        mean_earliness_diff = 0
        if time_left_for_earliness > 0:
           #Calculamos el promedio
            mean_earliness = 0
            for i in client_earliness_value: 
                mean_earliness += i
            
            if len(self.state.clients_not_visited) != 0:
                mean_earliness = mean_earliness/len(self.state.clients_not_visited)
                if mean_earliness > self.state.tau_episode:
                    mean_earliness_diff = (mean_earliness-self.state.tau_episode)/120




        
        self.X_general_state.extend(self.fourier_features(mean_earliness_diff))

                    


        
        #Agregamos el promedio de las ultimas velocidades observadas por vehículo
        self.mean_velocities = []
        for vehicle_velocities in self.state.observed_velocity:
            mean_velocity = 0
            for velocity in vehicle_velocities:
                mean_velocity += velocity

            mean_velocity = mean_velocity/len(vehicle_velocities)
            #if mean_velocity < 0.06 and self.state.tau_episode > 310:
                
            self.mean_velocities.append(mean_velocity)

            #if mean_velocity < 0.06 and self.state.tau_episode > 308:
            #    self.X_general_state.append(1)
            #else:
                #self.X_general_state.append(0)
        
        #for vehicle in range(self.number_vehicles):
            #self.mean_velocities.append(self.state.observed_velocity[vehicle][2])
            #if self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != self.random_depot and self.state.vehicles_direction[vehicle] != self.state.vehicle_position[vehicle] and self.state.vehicle_position[vehicle] != self.random_depot:
            #    next_node = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.state.vehicles_direction[vehicle])][0][1]
            #    state_tau = int(self.state.tau_episode)
            #    key = (self.state.vehicle_position[vehicle], next_node, state_tau)
            #    avg_velocity = self.data_calculations.travel_data[key][1]
            #    speed_diff_normalized = self.mean_velocities[vehicle] - avg_velocity
             #   self.X_general_state.append(speed_diff_normalized)
            
            #else:
             #   self.X_general_state.append(0)

        total_congestions = 0
        self.vehicle_congestions = [0 for _ in range(self.number_vehicles)]
        for vehicle in range(self.number_vehicles):
            if self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != self.random_depot and self.state.vehicles_direction[vehicle] != self.state.vehicle_position[vehicle] and self.state.vehicle_position[vehicle] != self.random_depot:
                state_tau = int(self.state.tau_episode)
                if state_tau % 2 == 1:
                    state_tau -= 1
                next_node = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.state.vehicles_direction[vehicle])][0][1]
                key = (self.state.vehicle_position[vehicle], next_node, state_tau)
                congestion_penalization = self.congestion_upper_bound/0.78
                avg_congestion_velocity = congestion_penalization * self.data_calculations.travel_data[key][1]
                #avg_congestion_velocity = 0.14 * self.data_calculations.travel_data[key][1]   
                if self.mean_velocities[vehicle] < avg_congestion_velocity:
        #            self.X_general_state.append(1)
                    total_congestions += 1
                #    self.vehicle_congestions[vehicle] = 1
        self.total_congestions_normalized = total_congestions/self.number_vehicles
        self.X_general_state.extend(self.fourier_features(self.total_congestions_normalized))



        self.clasify_delayed_clients()

        self.fourier_features()
        
    def extract_state_action_features_fourier(self, action):
        self.X_state_action = []
        clients_left = copy.deepcopy(self.state.clients_not_visited)

        if len(self.state.clients_not_visited) != 0:
            time_left = (1150-self.state.tau_episode)/(850)
            time = (self.state.tau_episode-300)/850
            active_vehicles = 0



        
        else:
            time_left = 0
            time = 0
            number_vehicles_normalized = 0
            active_vehicles_normalized = 0

        
        for a in action:
            if a != self.random_depot and a in clients_left:
                #print(action)
                #print(len(self.state.clients_not_visited))
                clients_left.remove(a)
    


        #3

        #client_earliness_tw= []
        self.already_late_clients = []
        number_of_already_late_clients = 0
        #number_of_early_clients = 0
        #number_of_clients_in_time_window = 0



        client_earliness_value = []
        client_delay_value = []
        for client in clients_left:
            client_earliness, client_due_time = self.cg.clients[client]
            client_earliness_value.append(client_earliness)

            client_delay_value.append(client_due_time)
            if self.state.tau_episode > client_due_time:
                self.already_late_clients.append(client)
                number_of_already_late_clients += 1
            
            #elif self.state.tau_episode < client_earliness:
            #    number_of_early_clients += 1
            
            #else:
            #    number_of_clients_in_time_window += 1
            

        number_of_already_late_clients_normalized = number_of_already_late_clients/13


        self.X_state_action.extend(self.fourier_features(number_of_already_late_clients_normalized))

    
        vehicle_positions = self.state.vehicle_position

        
        #Inicializamos los vehículos ya marcados y el cambio de ruta
        already_marked_vehicles = [0 for _ in range(self.number_vehicles)]
        self.change_of_path = [0 for _ in range(self.number_vehicles)]

        # Verificamos los vehículos que ya tienen clientes marcados
        if len(self.state.clients_not_visited) != 0:
            for key, value in self.change_client_dictionary_1.items():
                vehicle, marked_time = key
                if marked_time <= self.state.tau_episode < marked_time + 20 and action[vehicle] == self.change_client_dictionary_1[key]:
                    already_marked_vehicles[vehicle] = 1
                    # Revisamos si algun vehiculo ha elegido el cliente previamente marcado
                    #for all_vehicles in range(self.number_vehicles):
                    #    if action[all_vehicles] == value:
                    #        self.change_of_path[all_vehicles] = 1
            
        
        #REVISAR ESTA VARIABLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # Marcamos nuevos clientes si el vehículo aún no tiene uno y su velocidad es menor a 0.12
        for vehicle in range(self.number_vehicles):
            if self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != self.random_depot and self.state.vehicles_direction[vehicle] != self.state.vehicle_position[vehicle]:
                state_tau = int(self.state.tau_episode)
                if state_tau % 2 == 1:
                    state_tau -= 1
                next_node = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.state.vehicles_direction[vehicle])][0][1]
                key = (self.state.vehicle_position[vehicle], next_node, state_tau)
                congestion_penalization = self.congestion_upper_bound/0.78
                avg_congestion_velocity = congestion_penalization * self.data_calculations.travel_data[key][1]
                #avg_congestion_velocity = 0.14 * self.data_calculations.travel_data[key][1]
                if already_marked_vehicles[vehicle] == 0 and avg_congestion_velocity >= self.mean_velocities[vehicle]:
                    key = (vehicle, self.state.tau_episode)
                    self.change_client_dictionary_1[key] = self.state.vehicles_direction[vehicle]
                    for vehicle in range(self.number_vehicles):
                        if action[vehicle] == self.change_client_dictionary_1[key]:
                            self.change_of_path[vehicle] = 1

        total_value = 0
        for value in self.change_of_path:
            total_value += value
            #self.X_state_action.append(value)
        re_route_variable = total_value/self.number_vehicles
        self.X_state_action.extend(self.fourier_features(re_route_variable))

        #self.X_state_action.append((re_route_variable)**2)
        
        distance = 0
        for vehicle in range(self.number_vehicles):
            #Servia
            distance += self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action[vehicle])][2]


                #if self.mean_velocities[vehicle] != 0:
                #    self.X_state_action.append(self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][2]/(2*self.mean_velocities[vehicle]))
                
                #else:
                #    self.X_state_action.append(0)

            #distance_cost = distance/100
            #distance_cost = distance/40
            #Este siii
        distance_cost = distance/100
            #distance_cost = distance*12.5/120
            
        self.X_state_action.extend(self.fourier_features(distance_cost))

        #self.X_state_action.append(distance_cost**2)



        #Variable binaria de si existe congestión o no



        
        
        earliness_cost = 0
        earliness_cost_per_vehicle_congestion = 0
        delay_cost = 0
        delay_cost_per_vehicle_congestion = 0
        overtime_cost = 0
        for vehicle in range(self.number_vehicles):
            #est_arrival = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle])
            est_arrival = self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1] + self.state.tau_episode

                    #est_arrival_depot = self.calculate_real_mean_travel_time(est_arrival, action[vehicle], 0, vehicle)
            if action[vehicle] != self.random_depot and action[vehicle] != self.state.vehicle_position[vehicle]:
                earliness_tw, delay_tw = self.cg.clients[action[vehicle]]
                            
                        #if self.change_of_path[vehicle] == 0:
                        #est_arrival = self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1] + self.state.tau_episode
                        #est_arrival = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle], vehicle)

                if est_arrival > 1150:
                        est_arrival = 1150                
                        #else:
                        #    est_arrival = (self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][2]/0.1) + self.state.tau_episode

                if est_arrival < earliness_tw:
                    earliness_cost += (earliness_tw-est_arrival)*self.earliness_cost_factor
                        
                elif est_arrival > delay_tw:
                    if self.state.tau_episode < delay_tw:
                        delay_cost += (est_arrival-delay_tw)*self.delay_cost_factor

                    elif self.state.tau_episode >= delay_tw:
                        delay_cost += (est_arrival-self.state.tau_episode)*self.delay_cost_factor
            
            


                

        #print(delay_cost/20)
        
        #self.X_state_action.append(overtime_cost/150)
        #Este si esss
        #earliness_cost/120
        #/55
        earliness_cost = earliness_cost/(60)
        self.X_state_action.extend(self.fourier_features(earliness_cost))





        #/80
        delay_cost = delay_cost/(80)

        self.X_state_action.extend(self.fourier_features(delay_cost))

        
        
        
        #print(total_delay_not_visited)
        
        

        
        #self.clasify_clients_by_vehicle_position(action)
        cost = 0
        distance_cost = 0
        cost_list = []
        total_max_distance = 0
        for vehicle in range(self.number_vehicles):
            distance = 0
            max_distance_vehicle = 0

            total_centroid_distance = 0
            #total_vehicle_travel_time = 0
            for tuple in self.vehicle_to_clients[vehicle]:
                
                client = tuple[1]



                if client not in action:
                    delay_tw = self.cg.clients[client][1]
                    travel_time = self.spm.shortest_paths[(action[vehicle], client)][1] + self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1]
                    #total_vehicle_travel_time += self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1] + self.service_time
                    #distance = self.spm.shortest_paths[(action[vehicle], client)][2]
                    #if distance > max_distance_vehicle:
                    #    max_distance_vehicle = distance

                    est_arrival_time = travel_time + self.state.tau_episode + self.service_time
                            
                    if est_arrival_time > 1150:
                        est_arrival_time = 1150
                            
                    if est_arrival_time > delay_tw and self.state.tau_episode <= delay_tw:
                        cost += (est_arrival_time-delay_tw)*self.delay_cost_factor
                            
                    elif est_arrival_time > delay_tw and self.state.tau_episode > delay_tw:
                        cost += (est_arrival_time-self.state.tau_episode)*self.delay_cost_factor
                    
                
            #total_max_distance += max_distance_vehicle
                
        
        #total_travel_time_normalized = (total_vehicle_travel_time+self.state.tau_episode)/3000
        #cost = cost/1500
        #cost = cost/2000
        cost = cost
        cost = cost/2000
        #if cost > 1:
        #    print(cost)
        self.X_state_action.extend(self.fourier_features(cost))
        self.X_state_action.extend(self.fourier_features(cost**2))



        


        #Costo overtime
        overtime_cost = 0
        for vehicle in range(self.number_vehicles):
            if action[vehicle] != self.random_depot:
                #if self.state.vehicle_completing_service[vehicle] == 1:
                #    est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action[vehicle])][1] + self.spm.shortest_paths[(action[vehicle], self.random_depot)][1] + self.state.tau_episode + self.service_time
                
                #else:
                est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action[vehicle])][1] + self.spm.shortest_paths[(action[vehicle], self.random_depot)][1] + self.state.tau_episode + self.service_time

                if est_arrival > self.end_of_horizon and self.end_of_horizon > self.state.tau_episode:
                    overtime_cost += self.overtime_cost * (est_arrival-self.end_of_horizon)
               
                elif est_arrival > self.end_of_horizon and self.state.tau_episode >= self.end_of_horizon:
                    overtime_cost += self.overtime_cost * (est_arrival-self.state.tau_episode)
            
            else:
                est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.random_depot)][1] + self.state.tau_episode
                
                if est_arrival > self.end_of_horizon and self.end_of_horizon > self.state.tau_episode:
                    overtime_cost += self.overtime_cost * (est_arrival-self.end_of_horizon)
                
                elif est_arrival > self.end_of_horizon and self.state.tau_episode >= self.end_of_horizon:
                    overtime_cost += self.overtime_cost * (est_arrival-self.state.tau_episode)

        overtime_cost_normalized = overtime_cost/(180)

        self.X_state_action.append(overtime_cost_normalized)
        self.X_state_action.extend(self.fourier_features(overtime_cost_normalized))


    def extract_state_action_features_Routes(self, action):
        self.X_state_action = []
        self.routes = copy.deepcopy(self.state.greedy_insertion_routes)
        for vehicle in range(self.number_vehicles):
            self.routes[vehicle][0] = self.state.vehicle_position[vehicle]

        for vehicle in range(self.number_vehicles):
            for i, sub_list in enumerate(self.routes):
                if action[vehicle] in sub_list and action[vehicle] != 0:
                    position = sub_list.index(action[vehicle])
                    sub_list.pop(position)
                    sub_list.insert(1, action[vehicle])
                    break
                
                elif action[vehicle] == 0:
                    self.routes[vehicle] = [self.state.vehicle_position[vehicle], 0]

        
        #print(self.routes)
        total_cost, earliness_cost, delay_cost, distance_cost, overtime_cost = self.evaluate_greedy_routes_cost(self.routes)

        self.X_state_action.append(earliness_cost/5000)
        self.X_state_action.append(earliness_cost/479)
        self.X_state_action.append(delay_cost/5300)
        self.X_state_action.append(distance_cost/820)
        self.X_state_action.append(overtime_cost/2200)
        #print(overtime_cost)
        already_marked_vehicles = [0 for _ in range(self.number_vehicles)]
        self.change_of_path = [0 for _ in range(self.number_vehicles)]


        if len(self.state.clients_not_visited) != 0:
            for key, value in self.change_client_dictionary_1.items():
                vehicle, marked_time = key
                if marked_time <= self.state.tau_episode < marked_time + 60:
                    already_marked_vehicles[vehicle] = 1
                    # Revisamos si algun vehiculo ha elegido el cliente previamente marcado
                    for all_vehicles in range(self.number_vehicles):
                        if action[all_vehicles] == value:
                            self.change_of_path[all_vehicles] = 1
            
        
        #REVISAR ESTA VARIABLEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        # Marcamos nuevos clientes si el vehículo aún no tiene uno y su velocidad es menor a 0.12
        for vehicle in range(self.number_vehicles):
            if already_marked_vehicles[vehicle] == 0 and 0.06 > self.mean_velocities[vehicle] and self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != 0:
                key = (vehicle, self.state.tau_episode)
                self.change_client_dictionary_1[key] = self.state.vehicles_direction[vehicle]
                for vehicle in range(self.number_vehicles):
                    if action[vehicle] == self.change_client_dictionary_1[key]:
                        self.change_of_path[vehicle] = 1

                
        for value in self.change_of_path:
            self.X_state_action.append(value)

    
    def extract_state_action_features(self, action):
        """
        Versión optimizada de extracción de features de estado-acción:
        - Usa variables locales para minimizar lookups en self.
        - Un solo bucle para cálculos de clientes restantes.
        - Pre-caché de distancias y tiempos de viaje.
        - Reemplazo de múltiples appends por operaciones combinadas.
        """
        # ——————————————————————————————
        # 1) Cache de atributos y estructuras
        cg_clients    = self.cg.clients
        spm_paths     = self.spm.shortest_paths
        travel_data   = self.data_calculations.travel_data
        state         = self.state
        tau           = state.tau_episode
        clients_all   = state.clients_not_visited
        
        random_depot  = self.random_depot
        n_veh         = self.number_vehicles
        service_time  = self.service_time
        end_horizon   = self.end_of_horizon
        earl_fact     = self.earliness_cost_factor
        delay_fact    = self.delay_cost_factor
        overtime_fact = self.overtime_cost

        # ——————————————————————————————
        # 2) Determinar clientes aún no atendidos tras la acción
        selected = {a for a in action if a != random_depot}
        clients_left = [c for c in clients_all if c not in selected]
        

        features = []

        # ——————————————————————————————
        # 3) Número de clientes ya atrasados (normalizado)
        late_count = sum(1 for c in clients_left if tau > cg_clients[c][1])
        #features.append(late_count / max(1, len(clients_all)))
        features.append(late_count / 13)
        # ——————————————————————————————
        features.append(0)
        # 4) Coste de distancia de la acción actual
        total_dist = sum(
            spm_paths[(state.vehicle_position[i], action[i])][2]
            for i in range(n_veh)
        )
        features.append(total_dist / 100.0)

        # ——————————————————————————————
        # 5) Costes de earliness y delay por vehículo
        earliness_cost = 0.0
        delay_cost     = 0.0
        for i, a in enumerate(action):
            if a in clients_all and a != random_depot:
                travel_time = spm_paths[(state.vehicle_position[i], a)][1]
                est_arrival = tau + travel_time
                earl_tw, due_tw = cg_clients[a]

                if est_arrival < earl_tw:
                    earliness_cost += (earl_tw - est_arrival) * earl_fact
                elif est_arrival > due_tw:
                    delay_cost += (est_arrival - max(due_tw, tau)) * delay_fact

        features.append(earliness_cost / 60.0)
        features.append(delay_cost     / 60.0)

        # ——————————————————————————————
        # 6) Coste por retrasos futuros de clientes no visitados
        future_delay = 0.0
        for veh in range(n_veh):
            for _, client in self.vehicle_to_clients[veh]:
                if client not in action:
                    # tiempo hasta client tras seleccionar action[veh]
                    t1 = spm_paths[(state.vehicle_position[veh], action[veh])][1]
                    t2 = spm_paths[(action[veh], client)][1]
                    est = tau + t1 + t2 + service_time
                    _, due_tw = cg_clients[client]
                    if est > due_tw:
                        future_delay += (est - max(due_tw, tau)) * delay_fact

        norm_future = future_delay / 2500.0
        features.append(norm_future)
        #features.append(norm_future**2)

        # ——————————————————————————————
        # 7) Coste de overtime al volver al depósito
        overtime_cost = 0.0
        for i, a in enumerate(action):
            if a != random_depot:
                t1 = spm_paths[(state.vehicle_position[i], a)][1]
                t2 = spm_paths[(a, random_depot)][1]
                est_ret = tau + t1 + t2 + service_time
            else:
                est_ret = tau + spm_paths[(state.vehicle_position[i], random_depot)][1]

            if est_ret > end_horizon:
                base = end_horizon if tau < end_horizon else tau
                overtime_cost += (est_ret - base) * overtime_fact

        features.append(overtime_cost / 180.0)

        # ——————————————————————————————
        # 8) Asignar vector de features
        self.X_state_action = features

    def extract_state_action_features_real(self, action):
        self.X_state_action = []
        #clients_left = set(self.state.clients_not_visited)

        #print("tiempo", self.state.tau_episode)

        #clients_left.difference_update([a for a in action if a != self.random_depot])

                # 2) Determinar clientes aún no atendidos tras la acción
        selected = {a for a in action if a != random_depot}
        clients_left = [c for c in self.state.clients_not_visited if c not in selected]
        #print("Numero de clientes", len(clients_left))
        

    


        #3

        #client_earliness_tw= []
        #self.already_late_clients = []
        number_of_already_late_clients = 0
        #number_of_early_clients = 0
        #number_of_clients_in_time_window = 0



        client_earliness_value = []
        client_delay_value = []
        for client in clients_left:
            client_earliness, client_due_time = self.cg.clients[client]
            client_earliness_value.append(client_earliness)

            client_delay_value.append(client_due_time)
            if self.state.tau_episode > client_due_time:
                #self.already_late_clients.append(client)
                number_of_already_late_clients += 1
            
            #elif self.state.tau_episode < client_earliness:
            #    number_of_early_clients += 1
            
            #else:
            #    number_of_clients_in_time_window += 1
            

        number_of_already_late_clients_normalized = number_of_already_late_clients/13


        self.X_state_action.append(number_of_already_late_clients_normalized)

    
        vehicle_positions = self.state.vehicle_position

        
        #Inicializamos los vehículos ya marcados y el cambio de ruta
        #already_marked_vehicles = [0 for _ in range(self.number_vehicles)]
        #self.change_of_path = [0 for _ in range(self.number_vehicles)]

        # Verificamos los vehículos que ya tienen clientes marcados
        #if len(self.state.clients_not_visited) != 0:
        #    for key, value in self.change_client_dictionary_1.items():
        #        vehicle, marked_time = key
        #        if marked_time <= self.state.tau_episode < marked_time + 45:
        #            already_marked_vehicles[vehicle] = 1
        #            if action[vehicle] == self.change_client_dictionary_1[key]:
        #                self.change_of_path[vehicle] = 1
            
        
        #for vehicle in range(self.number_vehicles):
        #    if self.state.tau_episode > 304 and self.state.vehicles_direction[vehicle] != self.random_depot and self.state.vehicles_direction[vehicle] != self.state.vehicle_position[vehicle] and already_marked_vehicles[vehicle] == 0:
        #        cont = 0
        #        for idx,list in enumerate(self.state.transitioning_arc[vehicle]):
        #            key = []
        #            key.extend(list)
        #            key.append(int(self.state.time_of_transitioning[vehicle][idx]))
        #            key = builtins.tuple(key)
        #            mean_with_congestion = self.data_calculations.travel_data[key][1] * self.congestion_upper_bound
        #            if mean_with_congestion >= self.state.observed_velocity[vehicle][idx]:
        #                cont += 1
                    
        #        if cont == self.state.n_arcs:
        #            key = (vehicle, self.state.tau_episode)
        #            self.change_client_dictionary_1[key] = self.state.vehicles_direction[vehicle]
        #            for vehicle_2 in range(self.number_vehicles):
        #                if action[vehicle_2] == self.change_client_dictionary_1[key]:
        #                    self.change_of_path[vehicle_2] = 1

        #total_value = 0
        #for value in self.change_of_path:
        #    total_value += value
            #self.X_state_action.append(value)
        #re_route_variable = total_value/self.number_vehicles
        #self.X_state_action.append(re_route_variable)
        self.X_state_action.append(0)
        
        distance = 0
        for vehicle in range(self.number_vehicles):
            #Servia
            distance += self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][2]


                #if self.mean_velocities[vehicle] != 0:
                #    self.X_state_action.append(self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][2]/(2*self.mean_velocities[vehicle]))
                
                #else:
                #    self.X_state_action.append(0)

            #distance_cost = distance/100
            #distance_cost = distance/40
            #Este siii
        distance_cost = distance/100
            #distance_cost = distance*12.5/120
            
        self.X_state_action.append(distance_cost)
        #self.X_state_action.append(distance_cost**2)



        #Variable binaria de si existe congestión o no



        
        
        earliness_cost = 0
        earliness_cost_per_vehicle_congestion = 0
        delay_cost = 0
        delay_cost_per_vehicle_congestion = 0
        overtime_cost = 0
        for vehicle in range(self.number_vehicles):
            #est_arrival = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle])
            est_arrival = self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1] + self.state.tau_episode
            
            if action[vehicle] not in self.state.clients_not_visited:
                continue

            elif action[vehicle] != self.random_depot:
                earliness_tw, delay_tw = self.cg.clients[action[vehicle]]

                #if action[vehicle] == self.state.vehicle_position[vehicle] and action[vehicle] in self.state.clients_not_visited:
                #    print("Vamos al cliente", action[vehicle], "Nos encontramos en el nodo", self.state.vehicle_position[vehicle])

                #if est_arrival > 1150:
                #        est_arrival = 1150                

                if est_arrival < earliness_tw:
                    earliness_cost += (earliness_tw-est_arrival)*self.earliness_cost_factor
                        
                elif est_arrival > delay_tw:
                    if self.state.tau_episode < delay_tw:
                        delay_cost += (est_arrival-delay_tw)*self.delay_cost_factor

                    elif self.state.tau_episode >= delay_tw:
                        delay_cost += (est_arrival-self.state.tau_episode)*self.delay_cost_factor
            
            


                

        #print(delay_cost/20)
        
        #self.X_state_action.append(overtime_cost/150)
        #Este si esss
        #earliness_cost/120
        #/55
        earliness_cost = earliness_cost/(60)



        self.X_state_action.append(earliness_cost)

        #/80
        delay_cost = delay_cost/(60)
        
        #delay_cost = 0
        self.X_state_action.append(delay_cost)
         #self.X_state_action.append(delay_cost)

        
        

        #print(total_delay_not_visited)
        
        

        
        #self.clasify_clients_by_vehicle_position(action)
        cost = 0
        distance_cost = 0
        cost_list = []
        total_max_distance = 0
        for vehicle in range(self.number_vehicles):
            distance = 0
            max_distance_vehicle = 0

            total_centroid_distance = 0
            #total_vehicle_travel_time = 0
            for tuple in self.vehicle_to_clients[vehicle]:
                
                client = tuple[1]



                if client not in action:
                    delay_tw = self.cg.clients[client][1]
                    travel_time = self.spm.shortest_paths[(action[vehicle], client)][1] + self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1]
                    #total_vehicle_travel_time += self.spm.shortest_paths[(vehicle_positions[vehicle], action[vehicle])][1] + self.service_time
                    #distance = self.spm.shortest_paths[(action[vehicle], client)][2]
                    #if distance > max_distance_vehicle:
                    #    max_distance_vehicle = distance

                    est_arrival_time = travel_time + self.state.tau_episode + self.service_time
                            
                    if est_arrival_time > 1150:
                        est_arrival_time = 1150
                            
                    if est_arrival_time > delay_tw and self.state.tau_episode <= delay_tw:
                        cost += (est_arrival_time-delay_tw)*self.delay_cost_factor
                            
                    elif est_arrival_time > delay_tw and self.state.tau_episode > delay_tw:
                        cost += (est_arrival_time-self.state.tau_episode)*self.delay_cost_factor
                    
                
            #total_max_distance += max_distance_vehicle
                
        
        #total_travel_time_normalized = (total_vehicle_travel_time+self.state.tau_episode)/3000
        #cost = cost/1500
        #cost = cost/2000
        #cost = cost
        cost = cost/2500
        #if cost > 1:
        #    print(cost)
        #cost  = 0
        self.X_state_action.append(cost) 


        #self.X_state_action.append(0)
        self.X_state_action.append(cost**2)


        #Costo overtime
        overtime_cost = 0
        for vehicle in range(self.number_vehicles):
            if action[vehicle] != self.random_depot:
                #if self.state.vehicle_completing_service[vehicle] == 1:
                #    est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action[vehicle])][1] + self.spm.shortest_paths[(action[vehicle], self.random_depot)][1] + self.state.tau_episode + self.service_time
                
                #else:
                est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], action[vehicle])][1] + self.spm.shortest_paths[(action[vehicle], self.random_depot)][1] + self.state.tau_episode + self.service_time

                if est_arrival > self.end_of_horizon and self.end_of_horizon > self.state.tau_episode:
                    overtime_cost += self.overtime_cost * (est_arrival-self.end_of_horizon)
               
                elif est_arrival > self.end_of_horizon and self.state.tau_episode >= self.end_of_horizon:
                    overtime_cost += self.overtime_cost * (est_arrival-self.state.tau_episode)
            
            else:
                est_arrival =  self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.random_depot)][1] + self.state.tau_episode
                
                if est_arrival > self.end_of_horizon and self.end_of_horizon > self.state.tau_episode:
                    overtime_cost += self.overtime_cost * (est_arrival-self.end_of_horizon)
                
                elif est_arrival > self.end_of_horizon and self.state.tau_episode >= self.end_of_horizon:
                    overtime_cost += self.overtime_cost * (est_arrival-self.state.tau_episode)

        overtime_cost_normalized = overtime_cost/(180)
        #overtime_cost_normalized = 0

        self.X_state_action.append(overtime_cost_normalized)

        #print(self.X_state_action)








        #vehicle_distance_to_centroid, max_dist_to_centroid, mean_dist_to_centroid, std_dist_to_centroid = self.data_calculations.calculate_clients_dispersion(clients_left, action)

        #max_dist_to_centroid_normalized = max_dist_to_centroid/14
        #total_dist_to_centroid_normalized = mean_dist_to_centroid*len(clients_left)/650
        #print("total_dist_to_centroid_normalized", total_dist_to_centroid_normalized)
        #std_dist_to_centroid_normalized = std_dist_to_centroid/4.7
        #print("std_dist_to_centroid_normalized", std_dist_to_centroid_normalized)
        #vehicle_distance_to_centroid_normalized = vehicle_distance_to_centroid/(40/2)
        #print("vehicle_distance_to_centroid_normalized", vehicle_distance_to_centroid_normalized)

        #self.X_state_action.append(max_dist_to_centroid_normalized)
        #self.X_state_action.append(total_dist_to_centroid_normalized)
        #self.X_state_action.append(std_dist_to_centroid_normalized)

    
    def haversine_distance(self, lat1, lon1, lat2, lon2):

        # Radio de la Tierra en kilómetros
        R = 6371.0
        
        # Convertir grados a radianes
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Diferencias de latitud y longitud
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Fórmula de Haversine
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distancia
        distance = R * c
        return distance

        
        
        """
        total_overtime_for_action = 0
        cont = 0
        for a in action:
            if a== 0:
                cont+=1
        
        if cont == self.number_vehicles and len(self.state.clients_not_visited) != 0:
            for client in self.state.clients_not_visited:
                delay_tw = self.cg.clients[client][1]
                if self.state.tau_episode > delay_tw:
                    total_overtime_for_action += (1150-self.state.tau_episode)*self.delay_cost_factor
                else:
                    total_overtime_for_action += (1150-delay_tw)*self.delay_cost_factor
            
            for vehicle in range(self.number_vehicles):
                est_depot_arrival = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], 0, vehicle)
                if self.state.tau_episode > self.end_of_horizon:
                    total_overtime_for_action += (est_depot_arrival-self.state.tau_episode)*self.delay_cost_factor
                
                elif est_depot_arrival > self.end_of_horizon:
                    total_overtime_for_action += (est_depot_arrival-self.end_of_horizon)*self.delay_cost_factor
        
        else:
            for vehicle in range(self.number_vehicles):
                if self.state.vehicle_position[vehicle] == 0:
                    continue
                
                elif action[vehicle] == 0:
                    travel_time = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle], vehicle)
                    if self.state.tau_episode > self.end_of_horizon:
                        total_overtime_for_action += travel_time * self.overtime_cost



                else:
                    est_arrival_time = self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle], vehicle) + self.calculate_real_mean_travel_time(self.state.tau_episode, vehicle_positions[vehicle], action[vehicle], vehicle) - self.state.tau_episode
                    if est_arrival_time > self.end_of_horizon and self.state.tau_episode >= self.end_of_horizon:
                        total_overtime_for_action += (est_arrival_time-self.state.tau_episode)*self.overtime_cost
                    
                    elif est_arrival_time > self.end_of_horizon and self.state.tau_episode < self.end_of_horizon:
                        total_overtime_for_action += (est_arrival_time-self.end_of_horizon)*self.overtime_cost

        
        #self.X_state_action.append(total_overtime_for_action/20)
        self.X_state_action.append(total_overtime_for_action/400) 
        """
        
     
        """
        sum_distances = 0
        num_pairs = 0
        for i in range(self.number_vehicles):
            for j in range(i + 1, self.number_vehicles):
                node_i = action[i]
                node_j = action[j]
                distance = self.spm.shortest_paths[(node_i, node_j)][2]
                sum_distances += distance
                num_pairs += 1

        # Paso 3: Calcular la distancia promedio
        if num_pairs > 0:
            average_distance = sum_distances / num_pairs
        else:
            average_distance = 0


        self.X_state_action.append(average_distance)
        """

        """
        cont = 0
        for destination in action:
            if destination == 0:
                cont += 1

        percentage_vehicles_in_depot = cont/self.number_vehicles

        percentage_clients_left = len(self.state.clients_not_visited)/self.number_clients

        self.X_state_action.append(percentage_clients_left*percentage_vehicles_in_depot)

        total_extra_dist = 0
        dist = 0
        for vehicle in range(self.number_vehicles):
            min_dist = float("inf")
            est_arrival_time = 0
            if action[vehicle] == 0 and len(self.possible_actions[vehicle])>1:
                for possible_vehicle_action in self.possible_actions[vehicle]:
                    for vehicle_2 in range(self.number_vehicles):
                        dist = self.spm.shortest_paths[(action[vehicle_2], possible_vehicle_action)][1]
                        if dist < min_dist:
                            min_dist = dist
            if min_dist != float("inf"):
                total_extra_dist += min_dist


        self.X_state_action.append(total_extra_dist)
        """

    #Hay un beta que falta que se coma el ruido.
    #Raiz de n clientes
    #Tengo disponibles tanto % de vehículos y tanto % de clientes
    #¿Que cosas malas hace el monte carlo?
    #Ver que ocurre realmente con el modelo, ¿Donde falla?
    #Feature que penalice ir al depot?

    
    def generate_best_approximate_action(self):
        # Seleccionamos las posibles acciones
        self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions)

        vehicle_changed = set()

        client_occupied = {}


        self.predictions = [[] for _ in range(self.number_vehicles)]

        # Generamos los features generales del estado una vez
        self.extract_general_state_features()


        if self.W is None:
            self.extract_state_action_features(self.action)
            self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
            self.X = np.array(self.X)
            self.create_W(len(self.X))

        #self.extract_state_action_features(self.action)
        #self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
        #self.X = np.array(self.X)
        #Q_pred = np.dot(self.X, self.W)

        # Copiamos la acción actual como la mejor acción inicial
        best_action = self.action.copy()


        # Iteramos sobre cada vehículo
        #print("la primera accion es:", self.action)
        for _ in range(self.number_vehicles):
            vehicle_change = None
            Q_pred = float('inf')
            for vehicle in range(self.number_vehicles):
                if vehicle in vehicle_changed:
                    continue
                # Iteramos sobre las posibles acciones para el vehículo
                for client in self.possible_actions[vehicle]:
                    #if client in client_occupied:
                        #continue


                    #if client == 0 or client not in self.action:
                        # Guardamos el valor original de la acción del vehículo
                    original_action = self.action[vehicle]
                        # Modificamos temporalmente la acción del vehículo
                    self.action[vehicle] = client
                        # Extraemos las características de estado-acción para la nueva acción
                    self.extract_state_action_features(self.action)
                    self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
                    self.X = np.array(self.X)

                    Q_pred_estimation = np.dot(self.X, self.W)
                    self.predictions[vehicle].append(Q_pred_estimation)

                        # Si el Q_pred_estimado es mejor, actualizamos la mejor acción
                    if Q_pred_estimation < Q_pred:
                        best_action = self.action.copy()
                        Q_pred = Q_pred_estimation
                            #El vehiculo que cambiamos lo guardamos
                        vehicle_change = vehicle


                        #Restauramos la acción original del vehículo
                    self.action[vehicle] = original_action


            # Actualizamos la acción con la mejor acción encontrada

            self.action = best_action
            if vehicle_change is not None:
                vehicle_changed.add(vehicle_change)
                if self.action[vehicle_change] != 0:
                    client_occupied[self.action[vehicle_change]] = 0

        #print("la accion escogida es, ", self.action, "con una preddicion de", Q_pred)

    """
    def generate_best_approximate_action(self):
        #self.possible_actions = self.select_vehicle_possible_actions(self.number_of_actions)
        vehicle_changed = set()

        self.extract_general_state_features()
        self.extract_state_action_features(self.action)
        if self.W is None:
            self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
            self.X = np.array(self.X)
            self.create_W(len(self.X))

        #for i in range(self.number_vehicles):
        #    if self.action[i] not in self.possible_actions[i]:
        #        self.action[i] = random.choice(self.possible_actions[i])

        #min_Q_pred = np.dot(self.X, self.W)

        # Initialize the best action as the current action
        best_action = self.action.copy()

        for _ in range(self.number_vehicles):
            improvement = False
            min_Q_pred = float('inf')
            vehicle_change = None

            #Iteramos sobre todo vehículo
            for vehicle in range(self.number_vehicles):
                if vehicle in vehicle_changed:
                    continue
                #Guardamos la acción que estamos cambiando
                original_action = self.action[vehicle]

                #Iteramos sobre acciones candidatas para ese vehículo
                action_candidates = self.possible_actions[vehicle]
                #print(action_candidates)

                #Removemos las acciones que ya están asignadas a otro vehiculos utilizando set.
                assigned_actions = set(self.action)
                # Permitimos que la acción original si sea tomada
                assigned_actions.discard(original_action)
                #Guardamos toda posible acción
                action_candidates = [a for a in action_candidates if a == 0 or a not in assigned_actions]

                #Si no quedan acciones candidatas, entonces mandamos el vehículo al depot
                if not action_candidates:
                    action_candidates = [0]
                    self.possible_actions[vehicle].append(0)

                #Iteramos sobre las posibles acciones
                for client in action_candidates:
                    self.action[vehicle] = client

                    # Extraemos features del estado y realizamos la predicción
                    self.extract_state_action_features(self.action)
                    self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
                    self.X = np.array(self.X)
                    Q_pred_estimation = np.dot(self.X, self.W)
                    #print(f"Vehicle: {vehicle}, Action: {client}, Q_pred_estimation: {Q_pred_estimation}")


                    #Guardamos la mejor acción si se encuentra un mejor Q_pred
                    if Q_pred_estimation < min_Q_pred:
                        min_Q_pred = Q_pred_estimation
                        best_action_candidate = self.action.copy()
                        vehicle_change = vehicle
                        improvement = True
                        #print(f"New best action for Vehicle {vehicle}: {client} with Q_pred_estimation: {Q_pred_estimation}")


                #Restauramos el valor original de la acción
                self.action[vehicle] = original_action

            #Si se encontró una mejora, entonces cambiamos la acción
            if improvement and vehicle_change is not None:
                self.action = best_action_candidate
                vehicle_changed.add(vehicle_change)
            else:
                #No se puede mejorar más
                break

            #print(f"Best action for Vehicle {vehicle} in this iteration: {best_action_candidate[vehicle]} with Q_pred_estimation: {min_Q_pred}")



        #Nos aseguramos que cada acción sea factible
        for vehicle in range(self.number_vehicles):
            if self.action[vehicle] not in self.possible_actions[vehicle]:
                print("hola")
                # Prepare the list of candidate actions
                assigned_actions = set(self.action)
                assigned_actions.discard(self.action[vehicle])  # Remove the vehicle's current (invalid) action
                action_candidates = [a for a in self.possible_actions[vehicle] if a == 0 or a not in assigned_actions]

                min_Q_pred = float('inf')
                best_action_candidate = None

                # Iterate over possible actions to find the best one
                for client in action_candidates:
                    self.action[vehicle] = client

                    # Extract state-action features and compute Q_pred_estimation
                    self.extract_state_action_features(self.action)
                    self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
                    self.X = np.array(self.X)
                    Q_pred_estimation = np.dot(self.X, self.W)

                    # Update the best action if a lower Q_pred_estimation is found
                    if Q_pred_estimation < min_Q_pred:
                        min_Q_pred = Q_pred_estimation
                        best_action_candidate = client

                #Asignamos la mejor acción
                if best_action_candidate is not None:
                    self.action[vehicle] = best_action_candidate
                else:
                    #Si no se encuentra una accion valida, lo mandamos al depot
                    self.action[vehicle] = 0
    """


    def create_W(self, number_features):
        #self.W = [0]*number_features
        self.W = np.zeros(number_features)
        #6 y 7 posicion 5 y 6
        #self.W[2] = 1
        #self.W[4] = 1
        #self.W[7] = 1
        #self.W[8] = 1
        #self.W = np.ones(number_features)
        #self.W[13] = 1


    def actualize_W(self, states, actions, rewards):
        self.rewards = rewards
    
        T = len(actions)

        #print(T)
        #print(len(rewards))

#        print(len(states))

        self.Q_preds = []


        cost = 0

        #Deberia ser e-5
        U_t = 0
        lr = self.learning_rate
        self.error = 0

        #print(len(states))
        #self.Q_error_list = []
        #estandarizamos
        cont = 0
        for t in range(T-1, -1, -1):
            #Valor real
            #self.total_cost_acquired = 0
            U_t += rewards[t+1]
            
            cont += 1
            #print(rewards[t+1])

            #print("el reward es", U_t)

            
            #print("Al entrengar el tiempo es", self.state.tau_episode)
            #print(rewards[t])
            #Guardamos los features del estado y de la acción de cada vehícul
            self.X = []
            self.state = states[t]
            #print("Tiempo episodio entrenamiento", self.state.tau_episode)
            
            #print("Durante entrenamiento tiempo", self.state.tau_episode)
            self.calculate_already_acquired_cost()
            
            #print(self.state.greedy_insertion_routes)
            #print("episode tau", self.state.tau_episode)
    



            self.extract_general_state_features()
            action = actions[t]
            #print("Action entrenamiento", action)
            self.extract_state_action_features(action)
            self.X = list(itertools.chain(self.X_general_state, self.X_state_action))

            #print(self.X)

            #print(self.X_state_action)

            
            self.X = np.array(self.X)
            #print("El vector de features es", self.X)
            #print("el vector de features es", self.X)


            Q_pred = np.dot(self.X, self.W)


            gradient = lr * ((U_t - self.total_cost_acquired - Q_pred) * self.X)




            self.W = self.W + gradient

            #print("El vector de betas es", self.W)
            #self.W = self.W + (lr * (U_t - Q_pred) * self.X)
            #print("Los pesos son", self.W)


            self.error += (U_t - self.total_cost_acquired - Q_pred)

            #if U_t - self.total_cost_acquired < 0:
            #    print(U_t - self.total_cost_acquired)

            #if self.total_cost_acquired < 0:
            #    print(self.total_cost_acquired)



        #print(cont)
        self.error = (self.error/T)

    def calculate_variable_significance(self,states,actions,rewards):
        T = len(actions)
        U_t = 0
        self.error = 0
        cost = []
        x = []
        for t in range(T-1, -1, -1):
            #Valor real
            self.calculate_already_acquired_cost()
            U_t += rewards[t+1] -self.total_cost_acquired

            cost.append(U_t)

            #print(rewards[t])
            #Guardamos los features del estado y de la acción de cada vehícul
            self.X = []
            self.state = states[t]



            self.extract_general_state_features()
            action = actions[t]

            self.extract_state_action_features(action)
            self.X = list(itertools.chain(self.X_general_state, self.X_state_action))

            self.X = np.array(self.X)
            
            x.append(self.X)
            # Convierte las listas en arrays de numpy
    
        y = np.array(cost)

        return y, x
    
        # Ajusta el modelo de regresión lineal con statsmodels
        #X_with_intercept = sm.add_constant(X)  # Agrega el término de intercepto
        #model = sm.OLS(y, X).fit()

        # Imprime el resumen del modelo
        #print(model.summary())


            #Q_pred = np.dot(self.X, self.W)
            #self.Q_preds.append(self.X)


            #for i in range(len(lr)):
            #   self.W[i] = self.W[i] + (lr[i]*(U_t - Q_pred)* self.X[i])
            #self.calculate_already_acquired_cost()


            #self.error += (U_t - Q_pred)


    def select_vehicle_possible_actions(self, number_of_actions, vehicle):
        possible_actions = []
        forbidden_actions = []

        for v in range(self.number_vehicles):
            if v == vehicle:
                continue

            else:
                forbidden_actions.append(self.action[v])
        
        
        if self.state.vehicle_position[vehicle] == self.random_depot and self.state.tau_episode > 350:
            possible_actions.append(self.random_depot)

        elif len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.random_depot)
        
        elif len(self.state.clients_not_visited) < 3:
            #self.clasify_delayed_clients()
            self.clasify_shortest_distance_clients()
            if self.shortest_distance_clients[vehicle]:
                for i in range(len(self.shortest_distance_clients[vehicle])):
                    #if self.vehicle_to_clients[vehicle][i][1] not in forbidden_actions:
                    possible_actions.append(self.shortest_distance_clients[vehicle][i][1])
                        


            else:
                possible_actions.append(self.random_depot)
            
        
        #elif self.state.vehicle_completing_service[vehicle] == 1:
        #    possible_actions.append(self.state.vehicle_position[vehicle])

            
        else:
            clients = self.state.clients_not_visited
                #Obtenemos los clientes que poseen el menor travel time
            travel_times = [
                (self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1], client) for client in clients]
            top_vehicle_actions = [vehicle_action for vehicle_action in travel_times if vehicle_action[1] not in forbidden_actions]
            
            possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]


            #all_actions = [client for _, client in heapq.nsmallest(20, top_vehicle_actions)]

            #for x in all_actions:
            #    if x not in possible_actions:
            #        earliness_tw, delay_tw = self.cg.clients[x]
            #        travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], x)][1] + self.state.tau_episode
            #        if earliness_tw<= travel_time <= delay_tw:
            #            possible_actions.append(x)
                #possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]


            possible_actions = list(set(possible_actions))

        


        

            if self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.random_depot)][1] + self.state.tau_episode > self.end_of_horizon:
                possible_actions.append(self.random_depot)
        
        #if len(self.state.clients_not_visited) < self.number_vehicles:
        #    possible_actions.append(0)
            
            #cont = 0
            for delayed_client in self.delayed_clients[vehicle]:

                #if cont == 2:
                #    break

                if delayed_client not in possible_actions and delayed_client not in forbidden_actions:
                    possible_actions.append(delayed_client)
            #        cont+= 1
                
        
        
            if len(possible_actions) == 0:
                possible_actions.append(self.random_depot)

        #print("Las posibles acciones son", possible_actions)        
        
        #if len(self.state.clients_not_visited) != 0 and self.state.vehicle_position[vehicle] in self.cg.clients_list_2:
            #possible_actions.append(self.state.vehicle_position[vehicle])
        
        
        return possible_actions
    

    def select_vehicle_possible_actions_Slack(self, number_of_actions, vehicle):
        possible_actions = []
        forbidden_actions = []

        for v in range(self.number_vehicles):
            if v == vehicle:
                continue

            else:
                forbidden_actions.append(self.action[v])
        
        
        if self.state.vehicle_position[vehicle] == self.random_depot and self.state.tau_episode > 350:
            possible_actions.append(self.random_depot)

        elif len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.random_depot)
        
        elif len(self.state.clients_not_visited) < 2:
            self.clasify_delayed_clients()
            if self.vehicle_to_clients[vehicle]:
                for i in range(len(self.vehicle_to_clients[vehicle])):
                    if self.vehicle_to_clients[vehicle][i][1] not in forbidden_actions:
                        possible_actions.append(self.vehicle_to_clients[vehicle][i][1])


            else:
                possible_actions.append(self.random_depot)

            
        else:
            clients = self.state.clients_not_visited
                #Obtenemos los clientes que poseen el menor travel time
            travel_times = [
                [self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1], client] for client in clients]
            


            top_vehicle_actions = [vehicle_action for vehicle_action in travel_times if vehicle_action[1] not in forbidden_actions]
            
            alpha = 1
            beta = 0

            for x in top_vehicle_actions:
                x[0] += self.state.tau_episode
                delay_tw = self.cg.clients[x[1]][1]
                slack = delay_tw - x[0]
                priority_score = alpha * x[0] + beta * slack
                x[0] = priority_score
                
            
            possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]



            #all_actions = [client for _, client in heapq.nsmallest(20, top_vehicle_actions)]

            #for x in all_actions:
            #    if x not in possible_actions:
            #        earliness_tw, delay_tw = self.cg.clients[x]
            #        travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], x)][1] + self.state.tau_episode
            #        if earliness_tw<= travel_time <= delay_tw:
            #            possible_actions.append(x)
                #possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]


            possible_actions = list(set(possible_actions))

        


        

            if self.spm.shortest_paths[(self.state.vehicle_position[vehicle], self.random_depot)][1] + self.state.tau_episode > self.end_of_horizon:
                possible_actions.append(self.random_depot)
        
        #if len(self.state.clients_not_visited) < self.number_vehicles:
        #    possible_actions.append(0)
            
            #cont = 0
            for delayed_client in self.delayed_clients[vehicle]:

                #if cont == 2:
                #    break

                if delayed_client not in possible_actions and delayed_client not in forbidden_actions:
                    possible_actions.append(delayed_client)
            #        cont+= 1
                
        
        
        if len(possible_actions) == 0:
            possible_actions.append(self.random_depot)
        
        
        
        #if len(self.state.clients_not_visited) != 0 and self.state.vehicle_position[vehicle] in self.cg.clients_list_2:
            #possible_actions.append(self.state.vehicle_position[vehicle])
        
        
        return possible_actions
    
    def select_vehicle_possible_actions_K_Means(self, number_of_actions, vehicle):
        possible_actions = []
        forbidden_actions = []
        top_vehicle_actions = []
        clients = []
        for v in range(self.number_vehicles):
            if v == vehicle:
                continue

            else:
                forbidden_actions.append(self.action[v])
        
        
        if self.state.vehicle_position[vehicle] == self.random_depot and self.state.tau_episode > 350:
            possible_actions.append(self.random_depot)

        elif len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.random_depot)
        
        else:
            for client in self.state.clients_not_visited:
                if self.cluster_dictionary[client] == vehicle:
                    clients.append(client)

            travel_times = [
                (self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1], client) for client in clients]
                #Obtenemos los clientes que poseen el menor travel time
            
            top_vehicle_actions = [vehicle_action for vehicle_action in travel_times if vehicle_action[1] not in forbidden_actions]

            possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]


            possible_actions = list(set(possible_actions))    
            
            clients = self.state.clients_not_visited
            travel_times = [
                (self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1], client) for client in clients]

            top_vehicle_actions = [vehicle_action for vehicle_action in travel_times if vehicle_action[1] not in forbidden_actions]
            
            more_actions = [client for _, client in heapq.nsmallest(number_of_actions, top_vehicle_actions)]

            more_actions = list(set(more_actions))

            for x in more_actions:
                if x not in possible_actions:
                    possible_actions.append(x)

        if len(possible_actions) == 0:
            possible_actions.append(self.random_depot)
        
        
        return possible_actions

    
    def select_vehicle_possible_actions_Rutas_Multiarmed(self, number_of_actions, vehicle):
        #Si el estado no es terminal se elige un acción para cada vehículo
        #Si no queda clientes por visitar, entonces el vehículo viaja al depot
        self.number_static_route_actions = 1
        
        number_closest_clients = 2
        possible_actions = []
        if len(self.state.clients_not_visited) == 0:
            possible_actions.append(self.random_depot)

        #Si le quedan dos nodos a la ruta son los depots
        elif len(self.static_routes[vehicle]) == 0:
            #print(self.static_routes[vehicle])
            possible_actions.append(self.random_depot)

                #Si se encuentra el siguiente nodo dentro de los clientes no visitados, entonces se viaja a el
        else:
            #print(self.static_routes)
            for client in self.static_routes[vehicle]:
                if client not in self.state.clients_not_visited:
                    #possible_actions.append(client)
                    self.static_routes[vehicle].remove(client)
            #print("La ruta es", self.static_routes[vehicle])


            if len(self.static_routes[vehicle]) >= self.number_static_route_actions:
                for i in range(self.number_static_route_actions):
            #        print(i)
                    possible_actions.append(self.static_routes[vehicle][i])
            
            elif 0< len(self.static_routes[vehicle]) < self.number_static_route_actions:
                number_route_actions = len(self.static_routes[vehicle])
                for i in range(number_route_actions):
                    possible_actions.append(self.static_routes[vehicle][i])
            
            elif len(self.static_routes[vehicle]) <= 0:
                possible_actions.append(0)

            closest_clients = self.get_closest_clients_of_centroid_of_list(possible_actions, number_closest_clients)

            possible_actions.extend(closest_clients)
            
            
            for v in range(self.number_vehicles):
                if v == vehicle:
                    continue
                else:
                    if self.action[v] in possible_actions and self.action[v] != self.random_depot:
                        possible_actions.remove(self.action[v])

        

        return possible_actions
        
    """
    def select_vehicle_possible_actions(self, number_of_actions, vehicle):
        remaining_clients = {}
        possible_actions = []
        vehicle_position = self.state.vehicle_position[vehicle]
        for client in self.state.clients_not_visited:
            remaining_clients[client] = []

        if vehicle_position == 0 and self.state.tau_episode > 320:
            possible_actions.append(0)

        elif len(self.state.clients_not_visited) == 0:
            possible_actions.append(0)

        else:
            travel_times = []
            for key in remaining_clients:
                if self.cluster_dictionary[key] == vehicle:
                    travel_times.append((self.spm.shortest_paths[(vehicle_position, key)][1], key))

                #Obtenemos los clientes que poseen el menor travel time
            if len(travel_times) != 0:
                possible_actions = [client for _, client in heapq.nsmallest(number_of_actions, travel_times)]

            else:
                possible_actions = [0]

        return possible_actions
    """
    def clasify_delayed_clients(self):
        #Guardamos los clientes mas cercanos a los vehiculos.
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for client in self.state.clients_not_visited:
            assigned_vehicle = None
            min_travel_time = float('inf')
            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.random_depot and self.state.tau_episode > 310:
                    continue
                
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx
                
                if assigned_vehicle is not None:
                    self.vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))
        
        
        for vehicle_idx, client_list in self.vehicle_to_clients.items():
            client_list.sort()
            for travel_time, client in client_list:
                if len(self.delayed_clients[vehicle_idx]) >=  2:
                    break  #Dejamos de añadir acciónes cuando ya pasamos el limite

                delay_tw = self.cg.clients[client][1]
                if travel_time + self.state.tau_episode >= delay_tw:
                    self.delayed_clients[vehicle_idx].append(client)


    def clasify_shortest_distance_clients(self):
        self.shortest_distance_clients = defaultdict(list)

        clients_remaining = len(self.state.clients_not_visited)

        if clients_remaining == 2:
            # Caso especial: exactamente dos clientes restantes
            vehicle_distances = []

            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.random_depot and self.state.tau_episode > 310:
                    continue

                total_distance = sum(
                    self.spm.shortest_paths[(vehicle_position, client)][1]
                    for client in self.state.clients_not_visited
                )
                vehicle_distances.append((total_distance, vehicle_idx))

            closest_two_vehicles = heapq.nsmallest(2, vehicle_distances)

            for _, vehicle_idx in closest_two_vehicles:
                for client in self.state.clients_not_visited:
                    travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle_idx], client)][1]
                    self.shortest_distance_clients[vehicle_idx].append((travel_time, client))

        elif clients_remaining == 1:
            # Caso especial: exactamente un cliente restante
            client = next(iter(self.state.clients_not_visited))
            distances = []

            for vehicle_idx, vehicle_position in enumerate(self.state.vehicle_position):
                if vehicle_position == self.random_depot and self.state.tau_episode > 310:
                    continue

                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                distances.append((travel_time, vehicle_idx))

            closest_vehicle = min(distances)
            assigned_vehicle_idx = closest_vehicle[1]
            self.shortest_distance_clients[assigned_vehicle_idx].append((closest_vehicle[0], client))


    def clasify_clients_by_vehicle_position(self, action):
        #Guardamos los clientes mas cercanos a los vehiculos.
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for client in self.state.clients_not_visited:
            assigned_vehicle = None
            min_travel_time = float('inf')
            for vehicle_idx, vehicle_position in enumerate(action):
                if vehicle_position == self.random_depot and self.state.tau_episode > 310:
                    continue
                
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx
                
                if assigned_vehicle is not None:
                    self.vehicle_to_clients[assigned_vehicle].append((min_travel_time, client))
        
        
        #for vehicle_idx, client_list in self.vehicle_to_clients.items():
        #    client_list.sort()
        #    for travel_time, client in client_list:
        #        if len(self.delayed_clients[vehicle_idx]) >=  2:
        #            break  #Dejamos de añadir acciónes cuando ya pasamos el limite

        #        delay_tw = self.cg.clients[client][1]
        #        if travel_time + self.state.tau_episode >= delay_tw:
        #            self.delayed_clients[vehicle_idx].append(client)
    

    def clasify_delayed_clients_rutas(self):
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for vehicle in range(self.number_vehicles):
            for client in self.static_routes[vehicle]:
                if client in self.state.clients_not_visited:
                    travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1]
                    self.vehicle_to_clients[vehicle].append((travel_time, client))
                    
    def clasify_delayed_clients_K_means(self):
        self.delayed_clients = [[] for _ in range(self.number_vehicles)]
        self.vehicle_to_clients = defaultdict(list)
        for client in self.state.clients_not_visited:
            vehicle = self.cluster_dictionary[client]
            travel_time = self.spm.shortest_paths[(self.state.vehicle_position[vehicle], client)][1]
            self.vehicle_to_clients[vehicle].append((travel_time, client))

        
        for vehicle_idx, client_list in self.vehicle_to_clients.items():
            client_list.sort()
            for travel_time, client in client_list:
                if len(self.delayed_clients[vehicle_idx]) >=  2:
                    break  #Dejamos de añadir acciónes cuando ya pasamos el limite

                delay_tw = self.cg.clients[client][1]
                if travel_time + self.state.tau_episode >= delay_tw:
                    self.delayed_clients[vehicle_idx].append(client)
        
        #print(self.vehicle_to_clients)

    def get_closest_clients(self, clients_left, vehicle_positions):
        self.vehicle_closest_clients = defaultdict(list)  # Diccionario para almacenar clientes por vehículo
        assigned_clients = set()  # Conjunto para rastrear los clientes ya asignados

        # Iterar sobre cada cliente y asignarlo al vehículo más cercano
        for client in clients_left:
            assigned_vehicle = None
            min_travel_time = float('inf')

            # Encontrar el vehículo más cercano al cliente actual
            for vehicle_idx, vehicle_position in enumerate(vehicle_positions):
                # Ignorar vehículos en la posición inicial después de cierto tiempo
                if vehicle_position == self.random_depot and self.state.tau_episode > 310:
                    continue

                # Calcular el tiempo de viaje
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    assigned_vehicle = vehicle_idx

            # Asignar el cliente al vehículo más cercano y marcarlo como asignado
            if assigned_vehicle is not None and client not in assigned_clients:
                self.vehicle_closest_clients[assigned_vehicle].append((min_travel_time, client))
                assigned_clients.add(client)
        
    def calculate_already_acquired_cost(self):
        #Delay de clientes que ya están tarde y no hemos llegado
        self.total_cost_acquired = 0
        for client in self.state.clients_not_visited:
            earliness_tw, delay_tw = self.cg.clients[client]
            if delay_tw  < self.state.tau_episode:
                self.total_cost_acquired += (self.state.tau_episode-delay_tw)*self.delay_cost_factor
        #Costo overtime
        for vehicle in range(self.number_vehicles):
            if self.state.vehicle_position[vehicle] != self.random_depot and self.state.tau_episode > self.end_of_horizon:
                self.total_cost_acquired += (self.state.tau_episode-self.end_of_horizon)*self.overtime_cost
        
        
    def get_closest_clients_of_centroid_of_list(self, clients, number_actions):
        latitudes = [self.data_calculations.latitude_and_longitude[client][0] for client in clients]
        longitudes = [self.data_calculations.latitude_and_longitude[client][1] for client in clients]
        centroid_lat = np.mean(latitudes)
        centroid_long = np.mean(longitudes)

        distances = []
        for client in self.state.clients_not_visited:
            client_lat, client_long = self.data_calculations.latitude_and_longitude[client]
            distance = self.haversine_distance(centroid_lat, centroid_long, client_lat, client_long)
            distances.append((client, distance))
            
        distances.sort(key=lambda x: x[1])
        closest_clients = [client for client, _ in distances[:number_actions]]

        return closest_clients


    """ 
    def calculate_geographical_clusters(self):
        # Agrupamos a los clientes utilizando KMeans y agregamos la etiqueta del clúster como característica
        coordenadas = [self.data_calculations.latitude_and_longitude[node] for node in self.state.clients_not_visited]
        data = np.array(coordenadas)
        k = self.number_vehicles  # Asumimos un clíster por cada vehículo
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        
        self.cluster_dictionary = {self.state.clients_not_visited[i]: labels[i] for i in range(len(self.state.clients_not_visited))}
        self.cluster_centers = kmeans.cluster_centers_

    def select_vehicle_possible_actions(self, num_actions):
        # Seleccionar las acciones más cercanas considerando la proximidad de los clientes y las ventanas de tiempo
        possible_actions = []
        for vehicle_position in self.state.vehicle_position:
            actions_for_vehicle = []
            client_distances = []
            for client in self.state.clients_not_visited:
                travel_time = self.spm.shortest_paths[(vehicle_position, client)][1]
                earliness_tw, due_time = self.cg.clients[client]
                estimated_arrival = travel_time + self.state.tau_episode
                client_distances.append((travel_time, client))
            
            # Ordenar clientes por tiempo de viaje y seleccionar los más cercanos
            client_distances.sort()  # Orden ascendente por tiempo de viaje
            actions_for_vehicle = [client for _, client in client_distances[:num_actions]]            
            if not actions_for_vehicle:
                actions_for_vehicle.append(0)  # Si no hay acciones posibles, ir al dépósito
            possible_actions.append(actions_for_vehicle)
            
        return possible_actions

    def monte_carlo_policy(self, state):
        self.state = state
        self.extract_general_state_features()
        possible_actions = self.select_vehicle_possible_actions(5)
        # Selección ε-greedy de las acciones para cada vehículo
        for vehicle in range(self.number_vehicles):
            if random.random() < self.epsilon:
                # Exploración: seleccionamos una acción aleatoria
                self.action[vehicle] = random.choice(possible_actions[vehicle])
            else:
                # Explotación: seleccionamos la mejor acción según Q-pred
                min_q_value = float('inf')
                best_action = 0
                for client in possible_actions[vehicle]:
                    self.extract_state_action_features([client if i == vehicle else 0 for i in range(self.number_vehicles)])
                    X = np.array(self.X_general_state + self.X_state_action)
                    if self.W is None:
                       self.create_W(len(X))
                    q_value = np.dot(X, self.W)
                    if q_value < min_q_value:
                        min_q_value = q_value
                        best_action = client
                self.action[vehicle] = best_action

        return self.action
    """ 
    def actualize_W_Q_learning(self, Reward_t_1, state_t, action_t, state_t_1):
        self.Q_real
        self.state = state_t
        lr = 0.000001
        self.extract_general_state_features()
        self.extract_state_action_features(action_t)
        self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
        self.X = np.array(self.X)
        Q_pred_t = np.dot(self.X, self.W)

        self.state = state_t_1
        if self.state.terminal:
            Q_pred_t_1 = 0

        else:
            self.monte_carlo_policy(self.state)
            action_t_1 = self.action
            self.extract_general_state_features()
            self.extract_state_action_features(action_t_1)
            self.X = list(itertools.chain(self.X_general_state, self.X_state_action))
            self.X = np.array(self.X)
            Q_pred_t_1 = np.dot(self.X, self.W)

        self.W = self.W - (lr * (Reward_t_1 + Q_pred_t_1 - Q_pred_t) * self.X)

    
    def calculate_real_mean_travel_time(self, state_tau, Node_start, Node_end):

        shortest_path = self.spm.shortest_paths[(Node_start, Node_end)][0]
        real_state_tau = state_tau
        
        # Truncar state_tau a un número par
        state_tau = int(real_state_tau)
        if state_tau % 2 == 1:
            state_tau -= 1

        for i in range(len(shortest_path) - 1):
            key = (shortest_path[i], shortest_path[i + 1], state_tau)
            length, speed = self.data_calculations.travel_data[key]
            real_state_tau += length / speed
            # Actualizar state_tau para la próxima iteración y truncarlo a un número par
            state_tau = int(real_state_tau)
            if state_tau > 1150:
                real_state_tau = 1150
                break
            
            if state_tau % 2 == 1:
                state_tau -= 1
            
        
        return real_state_tau

    def cheapest_inertion_route(self):
        vehicle_routes = [[self.state.vehicle_position[vehicle],0] for vehicle in range(self.number_vehicles)]
        clients = copy.deepcopy(self.state.clients_not_visited)

        for i in range(len(self.state.clients_not_visited)):
            random_index = random.randint(0, len(clients) - 1)
            random_client = clients[random_index]
            best_cost = float("inf")
            best_vehicle = None
            best_position = None

            for vehicle in range(self.number_vehicles):
                # Solo permite inserción entre los dos ceros
                for position in range(1, len(vehicle_routes[vehicle])):  
                    vehicle_routes[vehicle].insert(position, random_client)
                    cost = self.evaluate_greedy_routes_cost(vehicle_routes)[0]

                    if cost < best_cost:
                        best_cost = cost
                        best_position = position
                        best_vehicle = vehicle

                    vehicle_routes[vehicle].pop(position)

            # Si no encontró mejor posición (por ejemplo porque solo hay [0,0]), inserta en el medio por defecto
            if best_vehicle is None:
                best_vehicle = 0
                best_position = 1

            vehicle_routes[best_vehicle].insert(best_position, random_client)

            clients.pop(random_index)
        
        return vehicle_routes
    
    def evaluate_greedy_routes_cost(self, routes):
        total_routes_cost = 0
        earliness_cost = 0
        delay_cost = 0
        distance_cost = 0
        overtime_cost = 0
        for route in routes:
            if len(route) > 1:
                time = copy.deepcopy(self.state.tau_episode)
                for node in range(len(route)-1):
                    node_i = route[node]
                    node_j = route[node + 1]
                    #print(node_j)

                    if node_j != 0:
                        #if time > 1100:
                            #time = 1100
                        #est_time_arrival = self.calculate_real_mean_travel_time(time, node_i, node_j) + self.service_time
                        est_time_arrival = self.spm.shortest_paths[(node_i, node_j)][1] + time
                        time = est_time_arrival
                        if est_time_arrival > 1150:
                            est_time_arrival = 1145
                        earliness_tw, delay_tw = self.cg.clients[node_j]
                        if est_time_arrival < earliness_tw:
                            total_routes_cost += self.earliness_cost_factor*(earliness_tw-est_time_arrival)
                            earliness_cost += self.earliness_cost_factor*(earliness_tw-est_time_arrival)
                        if est_time_arrival > delay_tw and delay_tw < self.state.tau_episode:
                            total_routes_cost += self.delay_cost_factor*(est_time_arrival-self.state.tau_episode)
                            delay_cost += self.delay_cost_factor*(est_time_arrival-self.state.tau_episode)
                        elif est_time_arrival > delay_tw and delay_tw > self.state.tau_episode:
                            total_routes_cost += self.delay_cost_factor*(est_time_arrival-delay_tw)
                            delay_cost += self.delay_cost_factor*(est_time_arrival-delay_tw)

                        time += self.service_time
                        distance = self.spm.shortest_paths[(node_i, node_j)][2]
                        total_routes_cost += distance * self.distance_cost_factor
                        distance_cost += distance * self.distance_cost_factor

                
                    else:
                        depot = 0
                        est_time_arrival = self.spm.shortest_paths[(node_i, node_j)][1] + time
                        
                        if est_time_arrival > 1150:
                            est_time_arrival = 1145

                        if est_time_arrival > self.end_of_horizon and self.end_of_horizon < self.state.tau_episode:
                            total_routes_cost += (est_time_arrival - self.end_of_horizon)*self.overtime_cost
                            overtime_cost += (est_time_arrival - self.end_of_horizon)*self.overtime_cost
                        elif est_time_arrival > self.end_of_horizon and self.state.tau_episode > self.end_of_horizon:
                            total_routes_cost += (est_time_arrival - self.state.tau_episode)*self.overtime_cost
                            overtime_cost += (est_time_arrival - self.state.tau_episode)*self.overtime_cost
                        

                        distance_to_depot = self.spm.shortest_paths[(node_i, node_j)][2]
                        distance_cost += distance_to_depot
                        total_routes_cost += distance_to_depot*self.distance_cost_factor     
            #else:
            #    total_routes_cost+= 1000
            #print("total distance cost", distance_cost + distance_to)
        
        return total_routes_cost, earliness_cost, delay_cost, distance_cost, overtime_cost

    def compare(self,state, action):
        # asume que policy contiene ambas funciones
        self.state = state          # mismo estado de entrada
        self.extract_state_action_features(action)
        x_old = np.array(self.X_state_action)

        self.extract_state_action_features_real(action)
        x_new = np.array(self.X_state_action)

        delta = x_new - x_old
        print("Δ features:", np.round(delta, 4))
        print("Δ · w      :", np.round(np.dot(self.W[:8], delta[:8]), 2))
        
#Model con random depot
class model():
    def __init__(self,state, policy, DataCalculations, shortest_path_memory, client_generator,  number_vehicles,
                  horizon_start_time, horizon_end_time, random_depot, lower_congestion_bound, upper_congestion_bound, congestion_max_duration):

        self.clients_travel_time_dict = {}

        #self.service_times = service_times

        self.total_distance_travelled = 0

        self.delay_client = 0
        
        self.total_cost_2 = 0
        #Factores de multiplicación de cada costo
        self.earliness_cost = 0.1
        #self.earliness_cost = 1
        self.distance_cost = 1
        self.delay_cost = 1
        #self.delay_cost = 2
        self.overtime_cost = 5/6
        #self.overtime_cost = 2

        self.service_time = 5

        self.random_depot = random_depot
        
        self.clients = {}

        self.premature_ending = 0


        #Inicializamos clases
        self.state = state
        #self.state.greedy_insertion_routes = cheapest_insertion_routes

        #self.env = environment
        self.policy = policy
        self.data_calc = DataCalculations
        self.spm = shortest_path_memory
        self.cg = client_generator

        self.number_vehicles = number_vehicles
        self.horizon_start_time = horizon_start_time
        self.horizon_end_time = horizon_end_time

        self.lower_congestion_bound = lower_congestion_bound
        self.upper_congestion_bound = upper_congestion_bound
        self.congestion_max_duration = congestion_max_duration
        self.hours_max_duration = congestion_max_duration/60

        #Parámetros para crear velocidad
        self.event_probability = 1

        self.max_depth = 3
        self.total_earliness_clients = 0
        self.total_delay_clients = 0
        self.total_overtime_vehicles = 0
        #self.max_depth = 0
        

        #Diferencia de tiempo en que se actualiza el horizonte de tiempo
        self.tau_multiplicator_difference = 2

        self.action = [0 for _ in range(number_vehicles)]
        #self.action = np.zeros(number_vehicles, dtype=int)

        self.vehicles_shortest_path = [[0,0] for _ in range(number_vehicles)]
        #self.vehicles_shortest_path = {i: [0, 0] for i in range(number_vehicles)}


        #En que tiempo llega el vehículo al próximo nodo
        self.node_time_arrival = [horizon_start_time for _ in range(number_vehicles)]

        self.tau_multiplicator = horizon_start_time + self.tau_multiplicator_difference

        self.total_cost = 0

        self.transition_cost = 0

        self.work_time = horizon_end_time

        self.visited_clients = []

        #Creamos una lista que indica los tiempos en que un vehículo sale de un nodo
        self.tau_salida = [horizon_start_time for _ in range(number_vehicles)]

        #Lista que representa en que tiempo cambió de horizonte de tiempo el vehículo
        self.tau_vehicle_horizon_change = [horizon_start_time for _ in range(number_vehicles)]

        #Condición para terminar el ciclo while.
        self.end_transition_function = 1

        #Cantidad de nodos y clientes visitados
        self.node = 0
        self.client = 0

        self.total_time_riding = np.zeros(number_vehicles)

        self.distance_arc_distance_travelled = [0 for _ in range(number_vehicles)]

        #Costos totales para cada componente de costo.
        self.total_earliness_cost = 0
        self.total_delay_cost = 0
        self.total_distance_cost = 0
        self.total_overtime_cost = 0
        self.total_state_counter = 0

        #Listas para guardar los estados, rewards y acciones
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [0]

        self.velocities = []


        #Parametros para normalizar costos.
        self.state_distance_cost = 0
        self.state_delay_cost = 0
        self.state_earliness_cost = 0
        self.state_overtime_cost = 0

        self.number_of_unexpected_events_seen = {}

        self.route = [[0] for _ in range(self.number_vehicles)]


    def calculate_action_route (self, action):
        for vehicle in range(len(action)):
            #print("la accion es", action[vehicle])
            #Si esto ocurre significa que el vehículo está entregando un articulo y está cumpliendo el tiempo de servicio.
            if self.tau_salida[vehicle] > self.state.tau_episode:
                self.create_and_actualize_state_velocity(vehicle)

            elif action[vehicle] == self.random_depot and self.state.vehicle_position[vehicle] == self.random_depot:
                self.node_time_arrival[vehicle] = float('inf')
                self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

            #Si se escoge un nodo en el que ya llegamos dejamos ese vehiculo esperando en el nodo.
            #elif action[vehicle] == self.state.vehicle_position[vehicle]:
            #    self.tau_salida[vehicle] = self.state.tau_episode + 6
            #    self.create_and_actualize_state_velocity(vehicle)


            elif(self.action[vehicle] != action[vehicle] and self.node_time_arrival[vehicle] != float('inf')):
                vehicle_position = self.state.vehicle_position[vehicle]
                vehicle_destination = action[vehicle]
                #Primero revisamos si el vehículo se encuentra en un nodo o cliente
                if(self.tau_salida[vehicle] == self.state.tau_episode):
                    #Simplemente realizamos el shortest path desde su posición actual hasta su destino
                    shortest_path = copy.deepcopy(self.spm.shortest_paths[(vehicle_position, vehicle_destination)][0])
                    #shortest_path = self.spm.shortest_paths[(vehicle_position, vehicle_destination)][0][:]
                    #shortest_path = self.spm.shortest_paths[(vehicle_position, vehicle_destination)][0]
                    self.vehicles_shortest_path[vehicle] = shortest_path[:]
                    #Generamos velocidad, actualizamos el travel_time y actualizamos listas por consistencia
                    self.create_and_actualize_state_velocity(vehicle)


                #Si es distinto, esto significa que el vehículo va en la mitad de un arco y debe primero ir al nodo
                #que va en camino y luego cambiar la ruta
                else:
                    shortest_path = copy.deepcopy(self.spm.shortest_paths[(self.vehicles_shortest_path[vehicle][1], vehicle_destination)][0])
                    #shortest_path = self.spm.shortest_paths[(self.vehicles_shortest_path[vehicle][1], vehicle_destination)][0][:]
                    #shortest_path = self.spm.shortest_paths[(self.vehicles_shortest_path[vehicle][1], vehicle_destination)][0]
                    #La velocidad y el travel time quedan igual, dado que ya iba en camino y ya fue calculado.
                    shortest_path.insert(0, vehicle_position)
                    self.vehicles_shortest_path[vehicle] = shortest_path[:]

                    #self.vehicles_shortest_path[vehicle] = self.spm.shortest_paths[(vehicle_position, vehicle_destination)][0]

                #Actualizamos la acción
            
                self.action[vehicle] = action[vehicle]
                self.state.vehicles_direction[vehicle] = action[vehicle]
            
            #print("el shortest path es", self.vehicles_shortest_path[vehicle])
        #print("La poscion de los vehiculos es", self.state.vehicle_position)
        #print("La acción es", action)


    def create_and_actualize_state_velocity(self, vehicle):
        #Hacemos el caso para cuando el vehículo está entregando un paquete
        if self.state.tau_episode > 1198:
            self.terminate_state_passing_horizon()
           # return self.transition_cost

        elif self.tau_salida[vehicle] > self.state.tau_episode:
            self.state.observed_velocity[vehicle].pop(0)
            #Se le agrega la velocidad = 0
            self.state.observed_velocity[vehicle].append(0)
            self.node_time_arrival[vehicle] = self.tau_salida[vehicle]
            #Actualizamos esta lista por temas de consistencia
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

        else:
            velocity_and_travel_time = self.data_calc.create_random_velocity(self.vehicles_shortest_path[vehicle][0], self.vehicles_shortest_path[vehicle][1],
                                                                self.state.tau_episode, self.event_probability)
            
            self.velocities.append(velocity_and_travel_time[1]*60)

            #self.state.transitioning_arc[vehicle].pop(0)
            #self.state.transitioning_arc[vehicle].append([self.vehicles_shortest_path[vehicle][0], self.vehicles_shortest_path[vehicle][1]])

            #self.state.time_of_transitioning[vehicle].pop(0)
            #self.state.time_of_transitioning[vehicle].append(self.state.tau_episode)


            #Eliminamos la primera posición, es decir la última velocidad vista
            self.state.observed_velocity[vehicle].pop(0)



            #Agregamos esta nueva velocidad a la última posición
            self.state.observed_velocity[vehicle].append(velocity_and_travel_time[1])

            #Se guarda el tiempo en que llegará al cliente el vehículo
            self.node_time_arrival[vehicle] = self.state.tau_episode + velocity_and_travel_time[0]

            #Actualizamos el tiempo de salida del vehículo
            self.tau_salida[vehicle] = self.state.tau_episode

            #Actualizamos esta lista por temas de consistencia
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode


            self.total_time_riding[vehicle] += velocity_and_travel_time[0]



    def vehicle_reaches_client(self, min_travel_time_vehicle, min_travel_time):
         #Agregamos el cliente al que llegamos
        self.visited_clients.append(self.vehicles_shortest_path[min_travel_time_vehicle][1])
        #print("Se llego al cliente", self.vehicles_shortest_path[min_travel_time_vehicle][1])
        #service_time = self.service_times[self.vehicles_shortest_path[min_travel_time_vehicle][1]]

        

        #for i, sublist in enumerate(self.state.greedy_insertion_routes):
        #    if self.vehicles_shortest_path[min_travel_time_vehicle][1] in sublist:
        #        position = sublist.index(self.vehicles_shortest_path[min_travel_time_vehicle][1])
        #        sublist.pop(position)
        #        break
        
        #Eliminamos al cliente del estado
        self.state.clients_not_visited.remove(self.vehicles_shortest_path[min_travel_time_vehicle][1])

        #Guardamos la información del cliente
        earliness_time_window = self.cg.clients[self.vehicles_shortest_path[min_travel_time_vehicle][1]][0]
        lateness_time_window = self.cg.clients[self.vehicles_shortest_path[min_travel_time_vehicle][1]][1]

        self.clients_travel_time_dict[self.vehicles_shortest_path[min_travel_time_vehicle][1]] = [min_travel_time, min_travel_time_vehicle, earliness_time_window, lateness_time_window]

        self.route[min_travel_time_vehicle].append(self.vehicles_shortest_path[min_travel_time_vehicle][1])

        #Actualizamos la posición del vehículo a la posición del cliente
        self.state.vehicle_position[min_travel_time_vehicle] = self.vehicles_shortest_path[min_travel_time_vehicle][1]

        #Guardamos en el estado la hora a la que se llegó al cliente
        self.state.clients_arrival[self.vehicles_shortest_path[min_travel_time_vehicle][1]] = [min_travel_time, min_travel_time_vehicle]

        time_window_cost = 0
        #Si se llegó al cliente antes se incurre a un costo de earliness
        if min_travel_time < earliness_time_window:
            time_window_cost = (earliness_time_window-min_travel_time) * self.earliness_cost
            #print("El costo de earliness es", time_window_cost)
            self.transition_cost += time_window_cost
            self.total_earliness_cost += time_window_cost
            self.total_earliness_clients += 1
            if time_window_cost <0:
                print("El costo de earliness es", time_window_cost)
                
            self.state_earliness_cost += time_window_cost



        #Si se llegó al cliente después de la TW, se incurre un costo de lateness
        elif min_travel_time > lateness_time_window:
            time_window_cost = (min_travel_time-lateness_time_window) * self.delay_cost
            if time_window_cost <0:
                print("El costo de delay es", time_window_cost)
            #print("El costo de delay es", time_window_cost)
            self.transition_cost += time_window_cost
            self.total_delay_cost += time_window_cost
            self.state_delay_cost += time_window_cost
            self.total_delay_clients += 1


            self.delay_client+=1


        self.clients[self.vehicles_shortest_path[min_travel_time_vehicle][1]] = [min_travel_time, earliness_time_window, lateness_time_window, time_window_cost]

        self.vehicle_distance_transition_cost(min_travel_time)
        #print("En el tiempo", self.state.tau_episode)
        #self.vehicle_distance_transition_cost_2(min_travel_time, min_travel_time_vehicle)

        #self.state.tau_episode = min_travel_time

        #Actualizamos el tiempo de salida del vehículo
        #self.tau_salida[min_travel_time_vehicle] = self.state.tau_episode + service_time
        self.tau_salida[min_travel_time_vehicle] = self.state.tau_episode + self.service_time

        self.state.vehicle_completing_service[min_travel_time_vehicle] = 1

        #Actualizamos vehicle_horizon para calcular bien el tiempo los cambios de horizontes de tiempo
        self.tau_vehicle_horizon_change[min_travel_time_vehicle] = self.state.tau_episode

        self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0



    def vehicle_distance_transition_cost(self, min_travel_time):
        #Estos vehículos se movieron por una diferencia de tau a una velocidad dada.
        diff_tau = min_travel_time - self.state.tau_episode

        for vehicle in range(self.number_vehicles):
            #Se salta el vehiculo que ya llegó al depot
            if(self.node_time_arrival[vehicle] != float('inf')):
                #La última posición de la lista vehicle_velocity marca la última velocidad vista por el vehículo
                Vehicle_velocity = self.state.observed_velocity[vehicle][self.state.n_arcs-1]
                distance_travelled = Vehicle_velocity * diff_tau
                self.total_distance_travelled += distance_travelled
                self.state.total_vehicle_distance_travelled[vehicle] += distance_travelled
                self.transition_cost += (distance_travelled * self.distance_cost)
                self.total_distance_cost += distance_travelled * self.distance_cost
                self.state_distance_cost += distance_travelled * self.distance_cost


        #Actualizamos tiempo del episodio
        self.state.tau_episode = min_travel_time



    def vehicle_reaches_node(self, min_travel_time, min_travel_time_vehicle):
        #El vehículo llegó a un nodo que es un cliente, pero que ya fue visitado por otro vehículo
        if min_travel_time > 1198 or self.state.tau_episode > 1198:
            self.terminate_state_passing_horizon()

        elif(len(self.vehicles_shortest_path[min_travel_time_vehicle])==2
        and self.vehicles_shortest_path[min_travel_time_vehicle][1] in self.visited_clients):

            #Actualizamos costos de distancia por temas de consistencia (ineficiencia) y el tau del estado
            self.vehicle_distance_transition_cost(min_travel_time)

            #Se actualiza la posición del vehículo al nodo al que llegó
            self.state.vehicle_position[min_travel_time_vehicle] = self.vehicles_shortest_path[min_travel_time_vehicle][1]

            #Actualizamos la lista de salida y del cambio de ciclo
            self.tau_salida[min_travel_time_vehicle] = self.state.tau_episode
            self.tau_vehicle_horizon_change[min_travel_time_vehicle] = self.state.tau_episode

            self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0


            #Se debe terminar el ciclo para que se escoga una acción para este vehículo
            
            self.end_transition_function = 2

            return self.transition_cost

        else:
            #Se actualiza la posición del vehículo al nodo al que llegó
            self.state.vehicle_position[min_travel_time_vehicle] = self.vehicles_shortest_path[min_travel_time_vehicle][1]

            #Removemos el nodo en que nos encontrabamos del camino
            self.vehicles_shortest_path[min_travel_time_vehicle].pop(0)

            #Actualizamos costos de distancia por temas de consistencia (ineficiencia) y el tau del estado
            self.vehicle_distance_transition_cost(min_travel_time)

            #Generamos travel time, velocidad y actualizamos el tiempo de salida
            self.create_and_actualize_state_velocity(min_travel_time_vehicle)

            self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0


    def terminate_state_passing_horizon(self):
        self.state.terminal = True
        self.end_transition_function = 2
        #self.transition_cost += 5000

        delay_costs = 0
        for client in self.state.clients_not_visited:
            client_due_time = self.cg.clients[client][1]
            #print(client_due_time)
            if self.state.tau_episode > client_due_time:
                delay_costs += (self.state.tau_episode - client_due_time) * self.delay_cost

        self.total_delay_cost+= delay_costs
        self.transition_cost += delay_costs

        overtime_cost = 0
        for vehicle_position in self.state.vehicle_position:
            if vehicle_position != self.random_depot:
                overtime_cost += (self.state.tau_episode - self.work_time) * self.overtime_cost
                print("El tiempo es", self.state.tau_episode)

        self.total_overtime_cost+=overtime_cost
        self.transition_cost+=overtime_cost

        self.total_cost += self.transition_cost

    def terminate_state_if_all_vehicles_come_back(self):
        self.state.terminal = True
        self.end_transition_function = 2
        number_of_visited_clients = len(self.visited_clients)
        delay_costs = 0
        for client in self.state.clients_not_visited:
            client_due_time = self.cg.clients[client][1]
            if self.state.tau_episode > client_due_time:
                delay_costs += (1150 - client_due_time) * self.delay_cost

        self.total_delay_cost += delay_costs
        self.transition_cost += delay_costs
        print("volvieron antes y sobraron: ", len(self.state.clients_not_visited))
        self.premature_ending = 1
        #self.transition_cost += 10000

        self.total_cost += self.transition_cost


    def time_horizon_actualization(self, vehicle):
        #Se eliminan todos los eventos adversos que ya se deben eliminar
        #self.data_calc.eliminate_velocity_penalization(self.state.tau_episode)


        #Caso en que el vehículo sigue sin entregar un paquete
        if self.tau_salida[vehicle] > self.state.tau_episode:
            random_velocity = 0
            time_in_arc = self.state.tau_episode - self.tau_vehicle_horizon_change[vehicle]
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

            #Eliminamos la primera posición, es decir la última velocidad vista
            self.state.observed_velocity[vehicle].pop(0)

            #Agregamos esta nueva velocidad a la última posición
            self.state.observed_velocity[vehicle].append(random_velocity)

            #Se guarda el tiempo en que llegará al cliente el vehículo
            self.node_time_arrival[vehicle] = self.tau_salida[vehicle]

        #Los vehículos que ya llegaron al depot no se realiza esto.
        elif(self.node_time_arrival[vehicle] != float('inf')):

            #Generamos las velocidad y travel times
            velocity_and_travel_time = self.data_calc.create_random_velocity(self.vehicles_shortest_path[vehicle][0], self.vehicles_shortest_path[vehicle][1],
                                                                        self.state.tau_episode, self.event_probability)

            random_velocity = velocity_and_travel_time[1]

            #Vemos cuanta distancia fue recorrida con la velocidad pasada
            time_in_arc = self.state.tau_episode - self.tau_vehicle_horizon_change[vehicle]
            #print("Time_in_arc", time_in_arc)

            #Actualizamos el cambio de horizonte de tiempo del vehículo
            self.tau_vehicle_horizon_change[vehicle] = self.state.tau_episode

            distance_travelled = self.state.observed_velocity[vehicle][self.state.n_arcs-1] * time_in_arc


            #Vamos sumando la distancia que recorre en cada intervalo, dado que se puede actualizar el horizonte de tiempo varias veces
            #Antes de que el vehículo llegue a un nodo o cliente.
            self.distance_arc_distance_travelled[vehicle] += distance_travelled


            distance_left_to_travel = velocity_and_travel_time[2] - self.distance_arc_distance_travelled[vehicle]


            #Calculamos el travel time que queda por terminar el arco distancia/velocidad
            travel_time = distance_left_to_travel/random_velocity

            #Eliminamos la primera posición, es decir la última velocidad vista
            self.state.observed_velocity[vehicle].pop(0)

            #Agregamos esta nueva velocidad a la última posición
            self.state.observed_velocity[vehicle].append(random_velocity)

            #Se guarda el tiempo en que llegará al cliente el vehículo
            self.node_time_arrival[vehicle] = self.state.tau_episode + travel_time

    
    def _next_congestion_end(self):
        """
        Devuelve (t_event, veh) con el próximo tiempo en el que
        expira una congestión que afecta a un vehículo todavía en tránsito.
        Si no hay nada relevante: (float('inf'), None)
        """
        t_event  = float('inf')
        veh_event = None
        for v in range(self.number_vehicles):
            if self.node_time_arrival[v] == float('inf'):
                continue                              # vehículo estacionado en el depósito
            start = self.vehicles_shortest_path[v][0]
            dest  = self.vehicles_shortest_path[v][1]
            arc   = (start, dest)
            if arc in self.data_calc.congested_arcs:
                end = self.data_calc.congested_arcs[arc][1]
                # ¿El evento ocurre después de "ahora" y antes de llegar al nodo?
                if self.state.tau_episode < end < self.node_time_arrival[v]:
                    if end < t_event:
                        t_event, veh_event = end, v
        #print("End of congestion", t_event)
        return t_event, veh_event


    def transition_function(self, action):
        self.transition_cost = 0

        self.contador = 0

        #Planeamos rutas en base a las acciones
        self.calculate_action_route(action)
        self.end_transition_function = 1


        while(self.end_transition_function == 1):
            event_end, veh_event = self._next_congestion_end()
            min_travel_time_vehicle, min_travel_time = __builtins__.min(enumerate(self.node_time_arrival), key=lambda x: x[1])

            t_next = min(self.tau_multiplicator, min_travel_time, event_end)

            #FIFO si termina una congestión antes de min travel time o el contador.
            if t_next == event_end:
                # 1) coste de distancia hasta el minuto t_event
                self.vehicle_distance_transition_cost(t_next)
                
                for vehicle in range(self.number_vehicles):
                    self.time_horizon_actualization(vehicle)
                
                continue
                            



            #verificamos si el estado es terminal. Es decir, si todos los clientes han sido visitados y todos los vehículos han llegado al depot.
            if all(positions == self.random_depot for positions in self.state.vehicle_position) and len(self.state.clients_not_visited) == 0:
                #El estado es terminal
                self.state.terminal = True
                #Finaliza la funcion transición 
                self.end_transition_function = 2

                self.vehicle_distance_transition_cost(self.state.tau_episode)

                #Actualizamos costo total en los segementos del codigo que finalizará la transición.


                self.total_cost += self.transition_cost

            elif all(time == float('inf') for time in self.node_time_arrival):
                self.terminate_state_if_all_vehicles_come_back()

            #Si seguimos en el mismo horizonte de tiempo, entonces el vehículo llega a un nodo o un cliente
            elif(self.tau_multiplicator > min_travel_time):
                #Si un vehículo termina de entregar un paquete
                if self.tau_salida[min_travel_time_vehicle] == min_travel_time and len(self.vehicles_shortest_path[min_travel_time_vehicle]) <= 2:
                    self.state.vehicle_completing_service[min_travel_time_vehicle] = 0
                    #print("terminó el service time y la posición del vehiculo es", self.state.vehicle_position[min_travel_time_vehicle])    

                    self.vehicle_distance_transition_cost(min_travel_time)
                    self.tau_vehicle_horizon_change[min_travel_time_vehicle] = self.state.tau_episode
                    self.distance_arc_distance_travelled[min_travel_time_vehicle] = 0
                    self.state.tau_episode = min_travel_time
                    #Se debe terminar el ciclo para que se escoga una acción para este vehículo
                    self.end_transition_function = 2

                    self.total_cost += self.transition_cost

                    #return self.transition_cost

                else:
                    next_node = self.vehicles_shortest_path[min_travel_time_vehicle][1]

                    self.state.vehicle_next_node[min_travel_time_vehicle] = next_node

                    #verificamos si el estado es terminal. Es decir, si todos los clientes han sido visitados y todos los vehículos han llegado al depot.

                    #Verificamos si un vehículo llega al depot.
                    #condicion si no le faltan clientes por visitar
                    if next_node == self.random_depot and len(self.vehicles_shortest_path[min_travel_time_vehicle]) == 2:
                        #Le sumamos el costo de distancia
                        self.vehicle_distance_transition_cost(min_travel_time)
                        #Actualizamos la posición del vehículo
                        self.state.vehicle_position[min_travel_time_vehicle] = self.random_depot

                        #Sumamos el costo de overtime con la condición de que es la unica vez que ha llegado al depot.
                        if(self.state.tau_episode > self.work_time and self.node_time_arrival[min_travel_time_vehicle]!= float('inf')):

                            self.transition_cost += self.overtime_cost * (self.state.tau_episode - self.work_time)
                            self.total_overtime_cost += self.overtime_cost * (self.state.tau_episode - self.work_time)
                            self.total_overtime_vehicles += 1
                            if self.total_overtime_cost <0:
                                print("El tiempo es", self.state.tau_episode)
                            self.state_overtime_cost += self.overtime_cost * (self.state.tau_episode - self.work_time)

                        #Le sumamos un número grande a su self.node_time_arrival, para que no sea escogido nuevamente
                        #Cambiar a infinito
                        self.node_time_arrival[min_travel_time_vehicle]  = float('inf')
                        self.tau_salida[min_travel_time_vehicle] = 0


                        self.total_cost += self.transition_cost


                        self.end_transition_function = 2

                       # return self.transition_cost


                    #Si el vehículo llega a un cliente
                    elif next_node in self.state.clients_not_visited and action[min_travel_time_vehicle] == next_node:

                        #Actualizamos estado y costo de transición
                        self.vehicle_reaches_client(min_travel_time_vehicle, min_travel_time)

                        #La pasamos a 2 para que no vuelva iterar el ciclo y finalice la func de transición
                        self.end_transition_function = 2

                        #Actualizamos costo total en los segementos del codigo que finalizará la transición.
                        self.total_cost += self.transition_cost

                        #Retornamos el costo de transición.
                        self.client += 1

                        #return self.transition_cost

                    #El vehículo llega a un nodo que no es un cliente
                    else:
                        self.vehicle_reaches_node(min_travel_time, min_travel_time_vehicle)
                        self.node += 1

            #Sino cambiamos de horizonte de tiempo
            else:
                #print(self.state.tau_episode)
                #Actualizamos costo de transición y generamos nuevas velocidades para cada vehículo
                if self.tau_multiplicator >= 1198:
                    self.state.terminal = True
                    self.end_transition_function = 2
                    number_of_visited_clients = len(self.visited_clients)
                    self.transition_cost += 40000 - 200*number_of_visited_clients
                    self.total_cost += self.transition_cost
                    #return self.transition_cost


                else:
                    self.vehicle_distance_transition_cost(self.tau_multiplicator)

                    #Eliminamos todo evento inesperado que se debe eliminar
                    #self.data_calc.eliminate_velocity_penalization(self.state.tau_episode)

                    #Generamos evento inesperado aleatorio
                    time = (self.state.tau_episode + 180-2)/60
                    #if time%1 == 0:
                    if time%self.hours_max_duration == 0:
                    #    print(time)
                        #self.data_calc.create_random_unexpected_event(self.state.tau_episode, self.event_probability, self.max_depth)
                        self.data_calc.create_random_unexpected_event_with_probability_and_2_nodes(self.state.tau_episode, self.event_probability, self.max_depth, self.lower_congestion_bound, self.upper_congestion_bound)
                        #print(data_calculations.congested_arcs)
                        #print(self.state.tau_episode)

                    #Creamos nuevas velocidades en base al horizonte de tiempo y actualizamos el estado en base a esto
                    for vehicle in range(self.number_vehicles):
                        self.time_horizon_actualization(vehicle)
                    
                    
                    self.tau_multiplicator += self.tau_multiplicator_difference

                    if self.tau_multiplicator >=1150:
                        self.terminate_state_passing_horizon()
                        print("Se paso del horizonte de tiempo")
                        print("Min_travel_time es", min_travel_time)
                        #print("La velocidad es, ", self.state.vehicle_velocities)
                        print("El tiempo del estado es", self.state.tau_episode)
                    
                    time = (self.state.tau_episode + 180-2)
                    if time%6 == 0:
                        self.end_transition_function = 2


                        self.total_cost += self.transition_cost


        return self.transition_cost


    def save_episode_state_action(self, action):
        state = copy.deepcopy(self.state)
        self.episode_states.append(state)
        action = copy.deepcopy(action)
        self.episode_actions.append(action)     # Copia de la lista de acción

    def save_episode_reward(self, reward):
        reward = copy.deepcopy(reward)
        self.episode_rewards.append(reward)

    def create_static_episode(self):
        self.taus = []
        self.data_calc.congested_arcs = {}
        while not self.state.terminal:
            #print(self.data_calc.congested_arcs)
            action = self.policy.static_policy(self.state)
            self.save_episode_state_action(action)
            reward = self.transition_function(action)
            self.total_cost_2 += self.transition_cost
            self.taus.append(self.state.tau_episode)
            #print(self.state.tau_episode)
            self.save_episode_reward(reward)
            self.total_state_counter += 1
            #print("La acción es", action)
            #print("Tiempo es", self.state.tau_episode)
            #print("La posicion es", self.state.vehicle_position)
            #print("La velocidad del vehiculo 0 es", self.state.observed_velocity[0])

        #print(self.data_calc.congested_arcs)
        self.total_number_congestions = len(self.data_calc.congested_arcs)

        if self.total_delay_clients == 0:
            self.mean_delay_time = 0
        else:
            self.mean_delay_time = self.total_delay_cost/(self.total_delay_clients*self.delay_cost)
        
        if self.total_earliness_clients == 0:
            self.mean_earliness_time = 0
        else:
            self.mean_earliness_time = self.total_earliness_cost/(self.total_earliness_clients*self.earliness_cost)
        
        if self.total_overtime_vehicles == 0:
            self.mean_overtime = 0
        
        else:
            self.mean_overtime = self.total_overtime_cost/(self.total_overtime_vehicles*self.overtime_cost)

        self.data_calc.congested_arcs = {}
        self.data_calc.all_arc_velocity = {}


    def create_dynamic_episode(self):
        self.taus = []
        self.all_state_distance_cost = []
        self.all_delay_costs = []
        self.all_earliness_cost=[]
        self.all_overtime_cost = []
        self.episode_actions = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [0]
        while not self.state.terminal:
            self.state_distance_cost = 0
            self.state_delay_cost = 0
            self.state_earliness_cost = 0
            self.state_overtime_cost = 0

            self.data_calc.eliminate_velocity_penalization(self.state.tau_episode)
            action = self.policy.dynamic_policy(self.state)
            action_2 =copy.deepcopy(action)
            self.episode_actions.append(action_2)
            reward = self.transition_function(action_2)
            self.save_episode_reward(reward)
            #self.all_state_distance_cost.append(self.state_distance_cost)
            #self.all_delay_costs.append(self.state_delay_cost)
            #self.all_earliness_cost.append(self.state_earliness_cost)
            #self.all_overtime_cost.append(self.state_overtime_cost)

            #self.save_episode_information(action, reward)
            self.taus.append(self.state.tau_episode)

            self.total_state_counter += 1
        self.data_calc.congested_arcs = {}
        #self.data_calc.eliminate_velocity_penalization(2000)


    def create_monte_carlo_episode_train(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [0]
        #self.vehicles_route = [[self.random_depot] for _ in range(self.number_vehicles)]
        self.data_calc.congested_arcs = {}
        while not self.state.terminal:
            #self.data_calc.eliminate_velocity_penalization(self.state.tau_episode)
            action = self.policy.monte_carlo_policy_train(self.state)
            self.save_episode_state_action(action)
            
            reward = self.transition_function(action)
            #print("Costo de distancia", self.total_distance_cost)
            self.save_episode_reward(reward)
            #print("reward", reward)
            #print("rewards", self.episode_rewards)
            self.total_cost_2 += self.transition_cost
            cont = 0
            self.total_distance_cost = 0
            self.total_state_counter += 1

        
        self.policy.actualize_W(self.episode_states, self.episode_actions, self.episode_rewards)
        #y,x = self.policy.calculate_variable_significance(self.episode_states, self.episode_actions, self.episode_rewards)
        self.data_calc.congested_arcs = {}
        self.data_calc.all_arc_velocity = {}

        #return y,x
        
    def create_monte_carlo_episode_test(self):
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = [0]
        #self.vehicles_route = [[self.random_depot] for _ in range(self.number_vehicles)]
        self.data_calc.congested_arcs = {}
        while not self.state.terminal:
            #self.data_calc.eliminate_velocity_penalization(self.state.tau_episode)
            action = self.policy.monte_carlo_policy_test(self.state)
            reward = self.transition_function(action)
            #print("Numero de clientes", len(self.state.clients_not_visited))
            self.total_cost_2 += self.transition_cost
            cont = 0
            self.total_state_counter += 1


        if self.total_delay_clients == 0:
            self.mean_delay_time = 0
        else:
            self.mean_delay_time = self.total_delay_cost/(self.total_delay_clients*self.delay_cost)
        
        if self.total_earliness_clients == 0:
            self.mean_earliness_time = 0
        else:
            self.mean_earliness_time = self.total_earliness_cost/(self.total_earliness_clients*self.earliness_cost)
        
        if self.total_overtime_vehicles == 0:
            self.mean_overtime = 0
        
        else:
            self.mean_overtime = self.total_overtime_cost/(self.total_overtime_vehicles*self.overtime_cost)
        
        #y,x = self.policy.calculate_variable_significance(self.episode_states, self.episode_actions, self.episode_rewards)
        self.data_calc.congested_arcs = {}
        self.data_calc.all_arc_velocity = {}

    def create_Q_learning_episode(self):
        while not self.state.terminal:
            action = self.policy.monte_carlo_policy(self.state)
            state_t = self.state
            reward = self.transition_function(action)
            state_t_1 = self.state
            self.policy.actualize_W_Q_learning(reward, state_t, action, state_t_1)

        self.data_calc.congested_arcs = {}

class training_and_testing():
    def __init__(self, data_calculations, spm):
        self.data_calculations = data_calculations
        self.spm = spm
        self.cg = ClientGenerator(0)
        self.Best_W = []
        self.w = []

    def training_model(self, total_train_iterations, test_frequency, learning_rate, epsilon, congestion_lower_bound, congestion_upper_bound,max_congestion_duration,mean_number_clients, diff_TW):
        cont = 0
        Q_pred = 100000000000
        self.Best_W = []
        self.Newest_W = []
        random_seed = 1000
        All_W_test = []

        w = None


        #random_seed = 0    
        #Muchos clientes y muchos vehiculos en el depot.
        #El costo crece raiz de n
        lr = 0.000001
        while cont < total_train_iterations:
            #os.environ['OMP_NUM_THREADS'] = '1'

            rutas_multiarmed_150 = [[]]
            n_arcs = 3
            horizon_start_time = 300
            horizon_end_time = 780
            random_depot = 0

            cg = ClientGenerator(random_depot)
            cg.client_generator_function(random_seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time)
            np.random.seed(random_seed)
            clients = cg.client_list

            number_vehicles = int(cg.number_vehicles)
                    
            clients = cg.client_list
            number_clients = len(clients)
                    
            number_actions_train = number_vehicles + 2
            number_actions_test = number_vehicles + 2
            random_depot = int(random_depot)

            s = state(number_vehicles, clients, n_arcs, horizon_start_time, random_depot)
            p = policy(number_vehicles, rutas_multiarmed_150, self.spm, cg, self.data_calculations, s, number_clients , epsilon, random_depot, congestion_lower_bound, congestion_upper_bound, number_actions_train, number_actions_test, lr, w)
            m = model(s, p, self.data_calculations, self.spm, cg, number_vehicles, horizon_start_time, horizon_end_time, random_depot, congestion_lower_bound, congestion_upper_bound,max_congestion_duration)

            lr = learning_rate
            
            #random_seed = 0

            m.create_monte_carlo_episode_train()

            w = m.policy.W




            cont += 1

            
            random_seed+=1

            
            if cont%test_frequency == 0:
                total_cost_2 = 0
                cont_2 = 0
                self.Newest_W = copy.deepcopy(m.policy.W)
                #self.w = copy.deepcopy(self.m.policy.W)
                for seed in range(100000, 100050):

                    cg = ClientGenerator(random_depot)
                    cg.client_generator_function(seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time)
                    np.random.seed(seed)
                    clients = cg.client_list

                    number_vehicles = int(cg.number_vehicles)
                    
                    clients = cg.client_list
                    number_clients = len(clients)
                    
                    number_actions_train = number_vehicles + 2
                    number_actions_test = number_vehicles + 2
                    random_depot = 0

                    s = state(number_vehicles, clients, n_arcs, horizon_start_time, random_depot)
                    p = policy(number_vehicles, rutas_multiarmed_150, self.spm, cg, self.data_calculations, s, number_clients , epsilon, random_depot, congestion_lower_bound, congestion_upper_bound, number_actions_train, number_actions_test, learning_rate, self.Newest_W)
                            # 
                    m = model(s, p, self.data_calculations, self.spm, cg, number_vehicles, horizon_start_time, horizon_end_time, random_depot, congestion_lower_bound, congestion_upper_bound,max_congestion_duration)
                    m.create_monte_carlo_episode_test()

                    total_cost_2 += m.total_cost
                    cont_2 += 1
                All_W_test.append(total_cost_2/cont_2)
                print("costo promedio es", total_cost_2/cont_2)
                if total_cost_2/cont_2 < Q_pred:
                    Q_pred = total_cost_2/cont_2
                    self.Best_W = copy.deepcopy(self.Newest_W)
            
        
        # Supongamos que All_W_test ya está definido
        filename = f"Results__MeanClients{mean_number_clients}_DiffTw{diff_TW}_lr{learning_rate}_eps{epsilon}_lc{congestion_lower_bound}_uc{congestion_upper_bound}_duration{max_congestion_duration}_dos_test.png"

        x_test = [i * test_frequency for i in range(1, len(All_W_test) + 1)]

        if congestion_upper_bound == 0.4:
            if max_congestion_duration == 60:
                if mean_number_clients == 150:
                    mean_static_policy = 1094
                elif mean_number_clients == 250:
                    mean_static_policy = 1632
                    
            elif max_congestion_duration == 120:
                if mean_number_clients == 150:
                    mean_static_policy = 2490
                elif mean_number_clients == 250:
                    mean_static_policy = 2490
        
        if congestion_upper_bound == 0.3:
            if max_congestion_duration == 60:
                if mean_number_clients == 150:
                    mean_static_policy = 1250
                elif mean_number_clients == 250:
                    mean_static_policy = 1985

            elif max_congestion_duration == 120:
                if mean_number_clients == 150:
                    mean_static_policy = 2490
                elif mean_number_clients == 250:
                    mean_static_policy = 2490

        if congestion_upper_bound == 0.2:
            if max_congestion_duration == 60:
                if mean_number_clients == 150:
                    mean_static_policy = 1626
                elif mean_number_clients == 250:
                    mean_static_policy = 2591

            elif max_congestion_duration == 120:
                if mean_number_clients == 150:
                    mean_static_policy = 2490
                elif mean_number_clients == 250:
                    mean_static_policy = 2490

        if congestion_upper_bound == 0.1:
            if max_congestion_duration == 60:
                if mean_number_clients == 150:
                    mean_static_policy = 2559
                elif mean_number_clients == 250:
                    mean_static_policy = 4209

            elif max_congestion_duration == 120:
                if mean_number_clients == 150:
                    mean_static_policy = 2490
                elif mean_number_clients == 250:
                    mean_static_policy = 2490
                
        plt.figure(figsize=(20, 5))
        plt.plot(x_test, All_W_test, marker='o', linestyle='-', label='Cost')
        plt.axhline(y=mean_static_policy, color='red', linestyle=':', label='Mean Static Policy')
        plt.title('Objective Function under Greedy Policy during Training')
        plt.xlabel('Number of Episodes')
        plt.ylabel('Objective Function')
        plt.legend()

        ax = plt.gca()

        # Configuración del eje x para notación científica (10⁴)
        formatter_x = ticker.ScalarFormatter(useMathText=True)
        formatter_x.set_scientific(True)
        formatter_x.set_powerlimits((3, 3))  # Fuerza notación científica para 10⁴
        ax.xaxis.set_major_formatter(formatter_x)

        # Configuración del eje y para notación científica (10³)
        formatter_y = ticker.ScalarFormatter(useMathText=True)
        formatter_y.set_scientific(True)
        formatter_y.set_powerlimits((3, 3))  # Fuerza notación científica para 10³
        ax.yaxis.set_major_formatter(formatter_y)

        #Esto teniamos antes
        #plt.savefig(filename.replace('.png', '_graph.png'))
        # Crear la carpeta Training_Graph_Folder si no existe
        
        training_folder = "Training_Graph_Folder"
        if not os.path.exists(training_folder):
            os.makedirs(training_folder)

        # Generar el nombre del archivo y la ruta completa
        graph_filename = filename.replace('.png', '_graph.png')
        filepath = os.path.join(training_folder, graph_filename)
        plt.savefig(filepath)
        
    def test_model_2(self, congestion_lower_bound, congestion_upper_bound, max_congestion_duration, num_iteraciones_test, epsilon, learning_rate, mean_number_clients, diff_TW):
        #Celda de Testeo
        self.cost_list_1 = []
        self.total_distance_cost_list = []
        self.total_delay_cost_list = []
        self.total_earliness_cost_list = []
        self.total_overtime_cost_list = []
        self.total_tau_list = []
        self.mean_delay_time_list = []
        self.mean_earliness_time_list = []
        self.mean_overtime_list = []
        self.mean_states_list = []
        self.total_earliness_clients_list = []
        self.total_delay_clients_list = []

        filename = f"Results_MeanClients{mean_number_clients}_Tw{diff_TW}_lr{learning_rate}_eps{epsilon}_lc{congestion_lower_bound}_uc{congestion_upper_bound}_D{max_congestion_duration}.txt"

        folder = "Test_Metrics_Folder"
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)

        with open(filepath, "w") as file:
            file.write(f"Results for Learning Rate: {learning_rate}, Epsilon: {epsilon}, duration "
                    f"Lower Congestion: {congestion_lower_bound}, Upper Congestion: {congestion_upper_bound}, duration: {max_congestion_duration}\n")
            file.write("=" * 80 + "\n\n")

        if mean_number_clients == 150:
            seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,124,125,126,127,128,129,130,131,132,133,135,136,137,138,139,140,141,143,144,145,146,147,148,149,150,151,152]
            vehicle_list = [6, 5, 5, 7, 7, 6, 6, 5, 6, 5, 6, 5, 4, 7, 5, 6, 6, 5, 5, 6, 5, 6, 4, 7, 6, 5, 7, 8, 4, 4, 4, 4, 4, 5, 4, 5, 6, 8, 5, 5, 7, 5, 7, 6, 6, 4, 6, 7, 5, 6]
        
        elif mean_number_clients == 250:
            seeds = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141,142,143,144,145,146,148,149,150,151,152,153]
            vehicle_list = [9, 8, 9, 9, 9, 9, 8, 10, 8, 9, 9, 7, 10, 8, 9, 10, 8, 8, 8, 8, 10, 8, 9, 8, 9, 8, 9, 9, 6, 8, 7, 7, 8, 8, 8, 9, 8, 10, 8, 9, 11, 7, 10, 8, 7, 9, 9, 8, 9, 8]            

        number_actions_test = [2, 5, 10,15,20]
        for actions in number_actions_test:
            for i in range (len(seeds)):
                cont = 0
                cost = 0
                total_distance_cost = 0
                total_delay_cost = 0
                total_earliness_cost = 0
                total_overtime_cost = 0
                total_tau = 0
                cont = 0
                delay_clients = 0
                earliness_clients = 0
                overtime = 0
                total_states = 0
                total_earliness_clients = 0
                total_delay_clients = 0
                seed = seeds[i]
                while cont < num_iteraciones_test:
                    rutas_multiarmed_150 = [[]]
                    n_arcs = 3
                    horizon_start_time = 300
                    horizon_end_time = 780
                    #epsilon = 0
                    random_depot = 0
                    
                    cg = ClientGenerator(random_depot)
                    
                    cg.client_generator_function(seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time)
                    np.random.seed(seed)
                    number_vehicles = vehicle_list[i]
                    clients = cg.client_list
                    number_clients = len(clients)
                    
                    number_actions_train = number_vehicles + 2
                    number_actions_test = number_vehicles + actions

                    s = state(number_vehicles, clients, n_arcs, horizon_start_time, cg.random_depot)
                    p = policy(number_vehicles, rutas_multiarmed_150, self.spm, cg, self.data_calculations, s, number_clients , epsilon, cg.random_depot, congestion_lower_bound, congestion_upper_bound, number_actions_train, number_actions_test, learning_rate, self.Best_W)
                            # 
                    m = model(s, p, self.data_calculations, self.spm, cg, number_vehicles, horizon_start_time, horizon_end_time, cg.random_depot, congestion_lower_bound, congestion_upper_bound,max_congestion_duration)
                    m.create_monte_carlo_episode_test()

                    #if m.premature_ending == 1:
                    #    print("Se salta este episodio")
                    #    continue


                    cost += m.total_cost
                    total_distance_cost += m.total_distance_cost
                    total_delay_cost += m.total_delay_cost
                    total_earliness_cost += m.total_earliness_cost
                    total_overtime_cost += m.total_overtime_cost
                    total_tau += m.state.tau_episode
                    
                    delay_clients += m.mean_delay_time
                    earliness_clients += m.mean_earliness_time
                    overtime += m.mean_overtime
                    total_states += m.total_state_counter
                    total_earliness_clients += m.total_earliness_clients
                    total_delay_clients += m.total_delay_clients

                    cont+=1

                self.cost_list_1.append(cost/num_iteraciones_test)
                self.total_distance_cost_list.append(total_distance_cost/num_iteraciones_test)
                self.total_delay_cost_list.append(total_delay_cost/num_iteraciones_test)
                self.total_earliness_cost_list.append(total_earliness_cost/num_iteraciones_test)
                self.total_overtime_cost_list.append(total_overtime_cost/num_iteraciones_test)
                self.total_tau_list.append(total_tau/num_iteraciones_test)
                self.mean_delay_time_list.append(delay_clients/num_iteraciones_test)
                self.mean_earliness_time_list.append(earliness_clients/num_iteraciones_test)
                self.mean_overtime_list.append(overtime/num_iteraciones_test)
                self.mean_states_list.append(total_states/num_iteraciones_test)
                self.total_earliness_clients_list.append(total_earliness_clients/num_iteraciones_test)
                self.total_delay_clients_list.append(total_delay_clients/num_iteraciones_test)


                # Cálculo de medias y desviaciones estándar
            metrics = {
                "Mean Cost": (np.mean(self.cost_list_1), np.std(self.cost_list_1)),
                "Mean Total Distance Cost": (np.mean(self.total_distance_cost_list), np.std(self.total_distance_cost_list)),
                "Mean Total Delay Cost": (np.mean(self.total_delay_cost_list), np.std(self.total_delay_cost_list)),
                "Mean Total Earliness Cost": (np.mean(self.total_earliness_cost_list), np.std(self.total_earliness_cost_list)),
                "Mean Total Overtime Cost": (np.mean(self.total_overtime_cost_list), np.std(self.total_overtime_cost_list)),
                "Mean Total Tau": (np.mean(self.total_tau_list), np.std(self.total_tau_list)),
                "Mean Delay Time": (np.mean(self.mean_delay_time_list), np.std(self.mean_delay_time_list)),
                "Mean Earliness Time": (np.mean(self.mean_earliness_time_list), np.std(self.mean_earliness_time_list)),
                "Mean Overtime": (np.mean(self.mean_overtime_list), np.std(self.mean_overtime_list)),
                "Mean States": (np.mean(self.mean_states_list), np.std(self.mean_states_list)),
                "Mean Total Earliness Clients": (np.mean(self.total_earliness_clients_list), np.std(self.total_earliness_clients_list)),
                "Mean Total Delay Clients": (np.mean(self.total_delay_clients_list), np.std(self.total_delay_clients_list)),
            }

                    
            file.write(f"Results for Actions = {actions}:\n")
            file.write("-" * 80 + "\n")
            for metric, values in metrics.items():
                mean_val, std_val = values
                file.write(f"{metric}:\n")
                file.write(f"  Mean: {mean_val:.4f}\n")
                file.write(f"  Std: {std_val:.4f}\n\n")

            file.write("=" * 80 + "\n\n")

    def test_model(self, congestion_lower_bound, congestion_upper_bound, max_congestion_duration, num_iteraciones_test, epsilon, learning_rate, mean_number_clients, diff_TW):

        filename = f"Results_MeanClients{mean_number_clients}_Tw{diff_TW}_lr{learning_rate}_eps{epsilon}_lc{congestion_lower_bound}_uc{congestion_upper_bound}_D{max_congestion_duration}_dos_test.txt"

        folder = "Test_Metrics_Folder"
        if not os.path.exists(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, filename)

        with open(filepath, "w") as file:
            file.write(f"Results for Learning Rate: {learning_rate}, Epsilon: {epsilon}, duration "
                    f"Lower Congestion: {congestion_lower_bound}, Upper Congestion: {congestion_upper_bound}, duration: {max_congestion_duration}_\n")
            file.write("=" * 80 + "\n\n")

            if mean_number_clients == 150:
                seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,124,125,126,127,128,129,130,131,132,133,135,136,137,138,139,140,141,143,144,145,146,147,148,149,150,151,152,153]
                vehicle_list = [6, 5, 5, 7, 7, 6, 6, 5, 6, 5, 6, 4, 7, 5, 6, 6, 5, 5, 6, 5, 6, 4, 7, 6, 5, 7, 8, 4, 4, 4, 4, 4, 5, 4, 5, 6, 8, 5, 5, 7, 5, 7, 6, 6, 4, 6, 7, 5, 6,6]

            elif mean_number_clients == 250:
                seeds = [100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141,142,143,144,145,146,148,149,150,151,152,153]
                vehicle_list = [9, 8, 9, 9, 9, 9, 8, 10, 8, 9, 9, 7, 10, 8, 9, 10, 8, 8, 8, 8, 10, 8, 9, 8, 9, 8, 9, 9, 6, 8, 7, 7, 8, 8, 8, 9, 8, 10, 8, 9, 11, 7, 10, 8, 7, 9, 9, 8, 9, 8]            

            number_actions_test_list = [2,10,20,30,40,50]
            for actions in number_actions_test_list:
                self.cost_list_1 = []
                self.total_distance_cost_list = []
                self.total_delay_cost_list = []
                self.total_earliness_cost_list = []
                self.total_overtime_cost_list = []
                self.total_tau_list = []
                self.mean_delay_time_list = []
                self.mean_earliness_time_list = []
                self.mean_overtime_list = []
                self.mean_states_list = []
                self.total_earliness_clients_list = []
                self.total_delay_clients_list = []
                for i in range (len(seeds)):
                    cont = 0
                    cost = 0
                    total_distance_cost = 0
                    total_delay_cost = 0
                    total_earliness_cost = 0
                    total_overtime_cost = 0
                    total_tau = 0
                    cont = 0
                    delay_clients = 0
                    earliness_clients = 0
                    overtime = 0
                    total_states = 0
                    total_earliness_clients = 0
                    total_delay_clients = 0
                    seed = seeds[i]
                    while cont < num_iteraciones_test:
                        rutas_multiarmed_150 = [[]]
                        n_arcs = 3
                        horizon_start_time = 300
                        horizon_end_time = 780
                        #epsilon = 0
                        random_depot = 0
                   
                        cg = ClientGenerator(random_depot)
                   
                        cg.client_generator_function(seed, mean_number_clients, diff_TW, horizon_start_time, horizon_end_time)
                        np.random.seed(seed)
                        number_vehicles = vehicle_list[i]
                        clients = cg.client_list
                        number_clients = len(clients)
                   
                        number_actions_train = number_vehicles + 2
                        number_actions_test = number_vehicles + actions

                        s = state(number_vehicles, clients, n_arcs, horizon_start_time, cg.random_depot)
                        p = policy(number_vehicles, rutas_multiarmed_150, self.spm, cg, self.data_calculations, s, number_clients , epsilon, cg.random_depot, congestion_lower_bound, congestion_upper_bound, number_actions_train, number_actions_test, learning_rate, self.Best_W)
                                #
                        m = model(s, p, self.data_calculations, self.spm, cg, number_vehicles, horizon_start_time, horizon_end_time, cg.random_depot, congestion_lower_bound, congestion_upper_bound,max_congestion_duration)
                        m.create_monte_carlo_episode_test()

                        #if m.premature_ending == 1:
                        #    print("Se salta este episodio")
                        #    continue


                        cost += m.total_cost
                        total_distance_cost += m.total_distance_cost
                        total_delay_cost += m.total_delay_cost
                        total_earliness_cost += m.total_earliness_cost
                        total_overtime_cost += m.total_overtime_cost
                        total_tau += m.state.tau_episode
                   
                        delay_clients += m.mean_delay_time
                        earliness_clients += m.mean_earliness_time
                        overtime += m.mean_overtime
                        total_states += m.total_state_counter
                        total_earliness_clients += m.total_earliness_clients
                        total_delay_clients += m.total_delay_clients

                        cont+=1

                    self.cost_list_1.append(cost/num_iteraciones_test)
                    print(self.cost_list_1)
                    self.total_distance_cost_list.append(total_distance_cost/num_iteraciones_test)
                    self.total_delay_cost_list.append(total_delay_cost/num_iteraciones_test)
                    self.total_earliness_cost_list.append(total_earliness_cost/num_iteraciones_test)
                    self.total_overtime_cost_list.append(total_overtime_cost/num_iteraciones_test)
                    self.total_tau_list.append(total_tau/num_iteraciones_test)
                    self.mean_delay_time_list.append(delay_clients/num_iteraciones_test)
                    self.mean_earliness_time_list.append(earliness_clients/num_iteraciones_test)
                    self.mean_overtime_list.append(overtime/num_iteraciones_test)
                    self.mean_states_list.append(total_states/num_iteraciones_test)
                    self.total_earliness_clients_list.append(total_earliness_clients/num_iteraciones_test)
                    self.total_delay_clients_list.append(total_delay_clients/num_iteraciones_test)


                    # Cálculo de medias y desviaciones estándar
                metrics = {
                    "Mean Cost": (np.mean(self.cost_list_1), np.std(self.cost_list_1)),
                    "Mean Total Distance Cost": (np.mean(self.total_distance_cost_list), np.std(self.total_distance_cost_list)),
                    "Mean Total Delay Cost": (np.mean(self.total_delay_cost_list), np.std(self.total_delay_cost_list)),
                    "Mean Total Earliness Cost": (np.mean(self.total_earliness_cost_list), np.std(self.total_earliness_cost_list)),
                    "Mean Total Overtime Cost": (np.mean(self.total_overtime_cost_list), np.std(self.total_overtime_cost_list)),
                    "Mean Total Tau": (np.mean(self.total_tau_list), np.std(self.total_tau_list)),
                    "Mean Delay Time": (np.mean(self.mean_delay_time_list), np.std(self.mean_delay_time_list)),
                    "Mean Earliness Time": (np.mean(self.mean_earliness_time_list), np.std(self.mean_earliness_time_list)),
                    "Mean Overtime": (np.mean(self.mean_overtime_list), np.std(self.mean_overtime_list)),
                    "Mean States": (np.mean(self.mean_states_list), np.std(self.mean_states_list)),
                    "Mean Total Earliness Clients": (np.mean(self.total_earliness_clients_list), np.std(self.total_earliness_clients_list)),
                    "Mean Total Delay Clients": (np.mean(self.total_delay_clients_list), np.std(self.total_delay_clients_list)),
                }

                   
                file.write(f"Results for Actions = {actions}:\n")
                file.write("-" * 80 + "\n")
                for metric, values in metrics.items():
                    mean_val, std_val = values
                    file.write(f"{metric}:\n")
                    file.write(f"  Mean: {mean_val:.4f}\n")
                    file.write(f"  Std: {std_val:.4f}\n\n")

                # Agregar los arrays W y Best_W al archivo
                file.write("=" * 80 + "\n")
                file.write("Array Newest_W:\n")
                file.write(np.array2string(self.Newest_W, precision=4, separator=', ') + "\n\n")
                file.write("Array Best_W:\n")
                file.write(np.array2string(self.Best_W, precision=4, separator=', ') + "\n")

def main():
    #Parámetros = input("Ingrese parametros de modelo en el siguiente orden, separados por comas: total_train_iterations, test_frequency, learning_rate, epsilon, congestion_lower_bound, congestion_upper_bound, max_congestion_duration, mean_number_clients, diff_TW, num_iteraciones_test")
    parametros = sys.argv[1]
    #print(parametros)
    opcion = 1

    if opcion == 1:
        # Usando split para separar la entrada en componentes
        parametros = parametros.split(',')
        total_train_iterations = int(parametros[0])
        test_frequency = int(parametros[1])
        learning_rate = float(parametros[2])
        epsilon = float(parametros[3])
        congestion_lower_bound = float(parametros[4])
        congestion_upper_bound = float(parametros[5])
        max_congestion_duration = int(parametros[6])
        mean_number_clients = int(parametros[7])
        diff_TW = int(parametros[8])
        num_iteraciones_test = int(parametros[9])
        
        file_path = "link.csv"
        file_path_velocities_morning = "speed[601]_[0].csv"
        file_path_velocities_afternoon = "speed[601]_[1].csv"
        random.seed(0)
        clients = random.sample(range(1, 1900), 150)
        #clients = [i for i in range (0,16)]
        #Le agregamos el 0
        clients.insert(0, 0)
        start_time = 300
        end_time = 780
        g = environment(file_path, file_path_velocities_morning, file_path_velocities_afternoon, clients, start_time, end_time)
        g.preprocess_data_average()

        data_calculations = DataCalculations(g, max_congestion_duration)

        spm = shortest_path_memory(g)

        del g

        training_and_testing_class = training_and_testing(data_calculations, spm)
        training_and_testing_class.training_model(total_train_iterations, test_frequency, learning_rate, epsilon, congestion_lower_bound, congestion_upper_bound, max_congestion_duration, mean_number_clients, diff_TW)
        training_and_testing_class.test_model(congestion_lower_bound, congestion_upper_bound, max_congestion_duration, num_iteraciones_test, epsilon, learning_rate, mean_number_clients, diff_TW)





if __name__ == "__main__":
    main()


