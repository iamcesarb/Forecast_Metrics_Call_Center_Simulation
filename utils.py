
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import truncnorm

from distfit import distfit
import scipy.stats as stats

import matplotlib.pyplot as plt
import simpy

import os


class TimeDeal():
    def __init__(self, time_str:str):
        self.time = datetime.strptime(time_str, '%H:%M').time()

    @staticmethod
    def time_to_timestamp(time: datetime.time, timestamp_duration: int = 15) -> str:
        """
        Converts a time (datetime.time) to a timestamp, based on how long a timestamp is. For the exercise, the default
        timestamp duration is 15 minutes. The time hour is multiplied by how many timestamps fit in an hour (4 by
        default), and is added to how many minutes in the time fit in the timestamp duration

        Parameters
        ----------
        time: a HH:MM:SS formatted time
        timestamp_duration: the duration of a timestamp in minutes

        Returns
        -------
        The calculated timestmap as an integer

        """
        hour_timestamp = time.hour * (60 / timestamp_duration)
        minute_timestamp = time.minute / timestamp_duration
        return int(hour_timestamp + minute_timestamp)

    
    




class CallsGenerator:
    """
    Generates a synthetic DataFrame with calls for every day of 2022. 
    """
    def __init__(self, starting_time, closing_time, seed=None):
        self.open_time = datetime.strptime(starting_time, '%H:%M').time()
        self.closing_time = datetime.strptime(closing_time, '%H:%M').time()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.rng_np = np.random.default_rng(seed)  # Generador para NumPy
            self.rng_scipy = np.random.default_rng(seed)  # Generador para SciPy
        self.__call_duration_mean = 300  # Duración media de la llamada en segundos
        self.__call_duration_std = 250   # Desviación estándar de la duración en segundos
        self.duration_dist = self.get_truncated_normal(mean=self.__call_duration_mean, sd=self.__call_duration_std, low=30, upp=1e6)
        self.distributions = self.create_distributions()

    @staticmethod
    def create_directory():
                
        carpetas = ['distributions', 'historic_data', 'plots', 'salidas']

        
        directorio_base = os.getcwd()

        
        for carpeta in carpetas:
            ruta_carpeta = os.path.join(directorio_base, carpeta)
            
            # Verifica si la carpeta existe
            if not os.path.exists(ruta_carpeta):
                # Si no existe se crea
                os.makedirs(ruta_carpeta)
                print(f'Se ha creado la carpeta: {ruta_carpeta}')
            else:
                print(f'La carpeta ya existe: {ruta_carpeta}')


    def create_distributions(self):
        distributions = {}
        for day in range(7):  # 0 Monday, 6 Sunday
            for minute in range(0, 1440, 15):  # 1440 minutes day,  15 min increase
                hour = minute // 60
                if self.is_within_operating_hours(hour):
                    mean, std = self.define_distribution_parameters(day, hour)
                    distributions[(day, hour)] = (mean, std)
        return distributions

    def is_within_operating_hours(self, hour):
        return self.open_time <= datetime(1, 1, 1, hour, 0).time() < self.closing_time

    def define_distribution_parameters(self, day, hour):
        # Mean and std for the synthetic data
        base_mean = 20  # 
        base_std = 5    # 

        # According to the weekday
        if day < 5:  # Días laborales
            mean_adjustment = np.random.uniform(0.8,1.2)
        else:  # Fin de semana
            mean_adjustment = np.random.uniform(0.4,0.8)

        if 9 <= hour <= 17:  # Horas pico
            hour_adjustment = np.random.uniform(1.5 ,1.8)
        else:
            hour_adjustment = np.random.uniform(0.5 ,0.9)

        mean = base_mean * mean_adjustment * hour_adjustment
        std = base_std * hour_adjustment
        return mean, std

    


    def get_truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        """
        Generates a truncated normal distribution according to scipy 
        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        """
        return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)



    def generate_calls_for_year(self, year=2022):
        """
        Generates syntetic calls for all the year 2022 accordign to the distributions
        defined. 
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        delta = timedelta(days=1)
        calls = []

        while start_date <= end_date:
            day_of_week = start_date.weekday()
            current_time = datetime.combine(start_date, self.open_time)
            while current_time.time() < self.closing_time:
                hour = current_time.hour
                mean, std = self.distributions[(day_of_week, hour)]
                num_calls = self.rng_np.poisson(mean)  # Random Number of calls
                for _ in range(num_calls):
                    duration = self.duration_dist.rvs(random_state=self.rng_scipy)
                    calls.append({'Start_Time': current_time, 'AHT': int(duration)})
                current_time += timedelta(minutes=15)
            start_date += delta

        return pd.DataFrame(calls)




    



class CallsSimulator:
    """
    Takes a historic df of calls, iterates through every week day/timestamp to get the
    probability distribution of calls and AHT. 
    """
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.rng_np = np.random.default_rng(seed)  # Generador para NumPy
            self.rng_scipy = np.random.default_rng(seed)

        self.days_of_week = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    
    def get_forecast(self):
        data = self.simulated_calls

        index = pd.MultiIndex.from_product([self.days_of_week, self.timestamps], names=['Dia', 'Franja'])

        df = pd.DataFrame(data.reshape(len(self.days_of_week) * len(self.timestamps), self.num_simulations), index=index)

        media = df.mean(axis=1)
        percentil_10 = df.quantile(0.10, axis=1)
        percentil_90 = df.quantile(0.90, axis=1)


        media = media.astype(int)
        percentil_10 = percentil_10.astype(int)
        percentil_90 = percentil_90.astype(int)


        result_df = pd.DataFrame({'Media': media, 'Percentil 10': percentil_10, 'Percentil 90': percentil_90})

        result_df.reset_index(inplace=True)
        print('---------------------------------\nForecast')
        display(result_df)
        result_df.to_excel('salidas/Forecast.xlsx', index=False)
        self.forecast= result_df

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 10))
        fig.suptitle("Forecast - Percentil 10 y Percentil 90 por Franja para Cada Día de la Semana", fontsize=20)
        min_value = np.min(result_df['Percentil 10'])
        max_value = np.max(result_df['Percentil 90'])

        # Iterar sobre cada día de la semana y trazar los percentiles 10 y 90 en los subplots correspondientes
        for i, ax in enumerate(axes.flat):
            if i < 7:  # Para los primeros 7 subplots
                dia_actual = result_df[result_df['Dia'] == self.days_of_week[i]]
                ax.fill_between(dia_actual.index, dia_actual['Percentil 10'], dia_actual['Percentil 90'], alpha=0.5, label='Percentil 10-90', color='blue')
                ax.set_xticks(dia_actual.index)
                ax.set_xticklabels(dia_actual['Franja'], rotation=90)
                ax.set_xlabel("Franja Horaria")
                ax.set_ylabel("Cantidad de Llamadas")
                ax.set_title(f"Forecast - Percentil 10 y Percentil 90 para {self.days_of_week[i]}", fontsize=16)
                ax.set_ylim(min_value-2, max_value+2)
                ax.legend()
                ax.grid()
            elif i >= 7:  
                ax.axis('off')  

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        plt.savefig('plots/forecast_plot.png')
        plt.show()

    @staticmethod
    def reformat_demand_times(demands: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats the date time format that demands originally have, and separates that value into three columns:
            1. date
            2. time
            3. timestamp based on the time
            4. day of the week (as string)

        Parameters
        ----------
        demands: the demands table

        Returns
        -------
        The demands table with the new date columns
        """
        demands['Start_Time'] = pd.to_datetime(demands['Start_Time'])
        demands['date'] = demands['Start_Time'].dt.date
        demands['time'] = demands['Start_Time'].dt.time
        demands['timestamp'] = demands['time'].apply(TimeDeal.time_to_timestamp)
        demands['day'] = demands['Start_Time'].dt.weekday
        return demands


    @classmethod
    def initialize_demands(cls, file_path: str, sheet_name: str = None):
        """
        Initializes the demands table on the demands static variable of the class

        Parameters
        ----------
        file_path: the path of the file containing the demands
        sheet_name: the sheet name that has the demands. Only takes effect when the file extension is xlsx
        """
        if file_path.split('.')[1] == 'csv':
            cls.demands = cls.reformat_demand_times(pd.read_csv(file_path))

        elif file_path.split('.')[1] == 'xlsx':
            cls.demands = cls.reformat_demand_times(pd.read_excel(file_path, sheet_name=sheet_name))

        else:
            raise ValueError('File extension not recognized')

    
    def get_daily_demand(self):

        daily_demand = dict()
        demands = self.demands[['date','timestamp','day']]
        demands['Calls'] = 1
        demands = demands.groupby(by=['date','timestamp', 'day']).sum()
        demands.reset_index(inplace=True)
        timestamps = sorted(list(demands['timestamp'].unique()))
        

        
        for day in range(7):
            daily_demand[day] = dict()
            data = demands[demands['day'] == day]  
            for timestamp in timestamps:
                daily_demand[day][timestamp] = list(data[data['timestamp']==timestamp]['Calls'])
        
        self.daily_demand = daily_demand
        self.timestamps = timestamps
        

    def simulate_calls(self, num_simulations = 1000):
        self.num_simulations = num_simulations
        result_array = np.zeros((7, len(self.timestamps), num_simulations))
        # Crear un mapeo de franja a índice
        franja_to_index = {str(franja): index for index, franja in enumerate(sorted(self.daily_demand[0].keys()))}

        for dia, franja in self.calls_probability_distributions.items():
            for franja_, values in franja.items():
                franja_ = str(franja_)
                # Ajustar con distfit
                index = franja_to_index[franja_]
                best_dist_name = values['best_dist_name']
                params = values['params']

                # Generar nuevas variables aleatorias basadas en la mejor distribución
                if hasattr(stats, best_dist_name):
                    dist_func = getattr(stats, best_dist_name)
                    generated_values = dist_func.rvs(*params, size=num_simulations, random_state=self.rng_scipy)
                    generated_values[generated_values < 0] = 0
                    result_array[int(dia), index]  = generated_values

                else:
                    print(f"Distribución '{best_dist_name}' no soportada en scipy.stats")

        self.simulated_calls = result_array


    def get_calls_distributions(self, distribution = None):
        """
        This function iterates over evety day and timestamp to get the best fir probability distribution
        according do distfit
        """
        if distribution:
            self.calls_probability_distributions = distribution

        else:
            calls_probability_distributions=dict()

            for dia, franja in self.daily_demand.items():
                dia = str(dia)
                calls_probability_distributions[dia]=dict()
                for franja_, values in franja.items():
                    franja_ = str(franja_)
                    calls_probability_distributions[dia][franja_]=dict()

                    # Ajustar con distfit
                    dist = distfit()
                    dist.fit_transform(np.array(values))
                    best_dist_name = dist.model['name']
                    params = dist.model['params']

                    calls_probability_distributions[dia][franja_]['best_dist_name'] = best_dist_name
                    calls_probability_distributions[dia][franja_]['params'] = params

            self.calls_probability_distributions = calls_probability_distributions

    def get_abandoned_dist(self, mean, std):
        abandoned_probability_distribution = {'best_dist_name': 'norm',
                                                'params': (mean,std)
                                                        }

        self.abandoned_probability_distribution = abandoned_probability_distribution



    def get_aht_distributions(self, distribution = None):
        """
        This function gets the best distribution fit for AHT, according to the 
        historic data. We assume AHT does not deppend on time or day. 
        """
        if distribution:
            self.AHT_probability_distributions = distribution

        else: 
            AHT_probability_distribution = dict()
            AHT_historic = np.array(self.demands['AHT'])

            dist = distfit()
            dist.fit_transform(AHT_historic)
            best_dist_name = dist.model['name']
            params = dist.model['params']
            AHT_probability_distribution['best_dist_name'] = best_dist_name
            AHT_probability_distribution['params'] = params

            self.AHT_probability_distributions = AHT_probability_distribution





    def plot_simulated_calls(self):

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(25, 10))
        fig.suptitle("Simulación de Llamadas por Día de la Semana", fontsize=20)
        min_value = np.min(self.simulated_calls)
        max_value = np.max(self.simulated_calls)
        
        # Iterar sobre cada día de la semana
        for i, ax in enumerate(axes.flat):
            if i < 7:
                ax.plot(self.simulated_calls[i, :, :], lw=0.5)  # Gráfico para el día i
                ax.set_xticks(range(len(self.timestamps)))   # Configurar las marcas del eje x
                ax.set_xticklabels(self.timestamps, rotation=90)          # Etiquetas del eje x
                ax.set_xlabel("Franja Horaria")
                ax.set_ylabel("Llamadas")
                ax.set_title(f"Simulación de Llamadas para {self.days_of_week[i]}", fontsize =16)
                ax.set_ylim(min_value-2, max_value+2)
                ax.text(40, 80, 'n=1000', ha='left', va='bottom', fontsize=12)
            elif i >= 7:  
                ax.axis('off')

        # Mostrar el gráfico
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Ajustar el layout para acomodar el título y el texto común
        plt.savefig('plots/forecast_simulation_plot.png')
        plt.show()


class CallCenterSimulator:
    def __init__(self, simulator, agents_per_timestamp, day_of_week, n_simulaciones=1000, seed=None):
        
        # self.env = env
        
        self.simulator = simulator
        self.calls_prob_dist = simulator.calls_probability_distributions[day_of_week]
        self.aht_prob_dist =  simulator.AHT_probability_distributions
        self.agents_per_timestamp = agents_per_timestamp
        self.day_of_week = day_of_week
        
        self.rng_scipy = np.random.default_rng(seed)  # Generador para SciPy
        self.abandoned_prob_dist = simulator.abandoned_probability_distribution
        self.total_calls = 0
        self.calls_answered_within_30s = 0
        self.abandoned_calls = 0
        self.n_simulaciones = n_simulaciones
        self.call_times = []  # Lista para registrar el tiempo de cada llamada que entra
        self.wait_times = [] 
        self.abandoned_times = []
        self.call_AHT_times = []

        
        



    # Numero de llamadas por timestamp
    def generate_calls(self, timestamp):
        dist_info = self.calls_prob_dist[timestamp]
        if hasattr(stats, dist_info['best_dist_name']):
            dist_func = getattr(stats, dist_info['best_dist_name'])
            num_calls = dist_func.rvs(*dist_info['params'], random_state=self.rng_scipy)
            return max(int(num_calls), 0)  # Asegurar que sea al menos 0
        return 0

    #Tomar llamada
    def handle_call(self, timestamp):

        #Contador llamadas
        self.total_calls += 1

        #Hora llamada
        self.call_times.append(self.env.now)
        #Generamos el tiempo de abandono 
        abandoned_time = self.generate_abandoned_time()
        #Resgistramos el tiempo de abandono
        self.abandoned_times.append(abandoned_time)

        start_wait = self.env.now

        with self.agents.request() as request:

            # Se contesta la llamada O el cliente se cansa de esperar
            result = yield request | self.env.timeout(abandoned_time)


            #Si la llamada de contesta
            if request in result:

                #Registramos el tiempo de espera

                wait_time = self.env.now - start_wait
                self.wait_times.append(wait_time)
 
                # Registramos si está en métrica 
                if wait_time <= 30:
                    self.calls_answered_within_30s += 1

                # Simular la duración de la llamada
                aht_info = self.aht_prob_dist
                if hasattr(stats, aht_info['best_dist_name']):
                    aht_func = getattr(stats, aht_info['best_dist_name'])
                    call_duration = aht_func.rvs(*aht_info['params'], random_state=self.rng_scipy)
                    self.call_AHT_times.append(call_duration)
                    yield self.env.timeout(max(call_duration, 0))  # Duración de la llamada

            else:             
                self.abandoned_calls += 1



     

    def run_call_center(self, env):
        for timestamp in self.simulator.timestamps:  
            

            timestamp = str(timestamp)
            #Ajustamos el numero de agentes disponibles 
            self.env = env
            self.agents = simpy.Resource(env, capacity=self.agents_per_timestamp[timestamp]) 

            num_calls = self.generate_calls(timestamp)

            if num_calls > 0:
                time_between_calls = 900 / num_calls  ### Asumimos que estan repartidas igual las llamadas dentro de los 15 minutos 
            
                for _ in range(num_calls):
                    #tomar llamada
                    env.process(self.handle_call(timestamp))
                    #Genere la siguiente llamada
                    yield env.timeout(time_between_calls)  
            else:
                continue
            
            #Apendeamos conteos por franja
  
    


    def generate_abandoned_time(self):
        # Simular el tiempo de espera para abandonar una llamada
        abandoned_info = self.abandoned_prob_dist
        if hasattr(stats, abandoned_info['best_dist_name']):
            abandoned_func = getattr(stats, abandoned_info['best_dist_name'])
            abandoned_time = abandoned_func.rvs(*abandoned_info['params'], random_state=self.rng_scipy)
            return max(abandoned_time, 0)  # Asegurar que sea positivo
        return float('inf')  # Si no hay distribución, nunca se abandona




    def simulate(self):
        nivel_servicio = list()
        abandonadas =  list()
        espera = list()
        total_abandonadas_timestamp = {str(timestamp):[] for timestamp in self.simulator.timestamps}
        total_nivel_servicio_timestamp = {str(timestamp):[] for timestamp in self.simulator.timestamps}
        total_espera_timestamp = {str(timestamp):[] for timestamp in self.simulator.timestamps}
        total_n_llamadas_timestamp = {str(timestamp):[] for timestamp in self.simulator.timestamps}


        for i in range(self.n_simulaciones):
            # Crear un nuevo entorno de SimPy para cada simulación
            env = simpy.Environment()

            # Usar i como semilla para la generación de números aleatorios
            seed = i

            # Crear una nueva instancia del simulador para cada simulación
            
            call_simulator = CallCenterSimulator(self.simulator, self.agents_per_timestamp, self.day_of_week, seed=seed)

            # Ejecutar la simulación
            env.process(call_simulator.run_call_center(env))
            env.run(until=43200)  # Tiempo total de simulación en segundos

            # Recoger y almacenar los resultados de esta simulación
            nivel_servicio.append(call_simulator.calls_answered_within_30s / call_simulator.total_calls)
            abandonadas.append(call_simulator.abandoned_calls / call_simulator.total_calls)
            espera.append(np.mean(call_simulator.wait_times))

            print(f'Simulación....{i+1}')


        nivel_servicio
        percentil_10 =  np.percentile(nivel_servicio, 10)
        percentil_90 = np.percentile(nivel_servicio, 90)
        std = np.std(nivel_servicio)
        mean = np.std(nivel_servicio)
        print('-------------------------------------\nNivel de servicio')
        print(f'Número de simulaciones ={self.n_simulaciones}')
        print(f'Media = {mean}')
        print(f'Desviación estandar = {std}')
        print(f'Intervalo de confianza al 80% = [{percentil_10},{percentil_90}]')
        plt.hist(nivel_servicio, color='r', alpha=0.7, edgecolor='k')
        plt.title('Nivel de Servicio en las simulaciones')
        plt.xlabel('Nivel de Servicio')
        plt.ylabel('Freq')
        plt.savefig('plots/Nivel de servicio.png')
        plt.show()
        

        percentil_10 =  np.percentile(abandonadas, 10)
        percentil_90 = np.percentile(abandonadas, 90)
        std = np.std(abandonadas)
        mean = np.std(abandonadas)
        print('-------------------------------------\nTasa de abandono')
        print(f'Número de simulaciones ={self.n_simulaciones}')
        print(f'Media = {mean}')
        print(f'Desviación estandar = {std}')
        print(f'Intervalo de confianza al 80% = [{percentil_10},{percentil_90}]')
        plt.hist(abandonadas, color='r', alpha=0.7, edgecolor='k')
        plt.title('Tasa de llamadas abandonadas en las simulaciones')
        plt.xlabel('Tasa de llamadas abandonadas')
        plt.ylabel('Freq')
        plt.savefig('plots/Tasa de abandono.png')
        plt.show()



        percentil_10 =  np.percentile(espera, 10)
        percentil_90 = np.percentile(espera, 90)
        std = np.std(espera)
        mean = np.std(espera)
        print('-------------------------------------\nTiempo de espera')
        print(f'Número de simulaciones ={self.n_simulaciones}')
        print(f'Media = {mean}')
        print(f'Desviación estandar = {std}')
        print(f'Intervalo de confianza al 80% = [{percentil_10},{percentil_90}]')
        plt.hist(espera, color='r', alpha=0.7, edgecolor='k')
        plt.title('Tiempo de espera para ser atendido en las simulaciones')
        plt.xlabel('Tiempo de espera (sec)')
        plt.ylabel('Freq')
        plt.savefig('plots/Tiempo de espera.png')
        plt.show()




