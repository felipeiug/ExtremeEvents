import os
import torch
import rioxarray
import numpy as np
import pandas as pd

# from numpy._core._exceptions import _ArrayMemoryError
from threading import Thread, Lock
from datetime import timedelta
from time import sleep

class ReadRasters:
    step = 0
    lock = Lock()   # Lock para sincronização
    buffer = None

    def __init__(self, n_days, data_path, data_grid, n_bandas, uso_solo_path, elevation_file, declive_file,  batch_size, n_times, train_percent, randomize=True, null_value = -99):
        """
        Os dados do dia N são os dados do dia N-1 ao dia `N-1-n_times`
        Todos os dias sem dados serão substituidos por uma matriz equivalente com os valores null_value em todo os 
        dados indisponíveis
        """

        self.randomize = randomize
        self.null_value = null_value
        self.batch_size = batch_size
        self.n_times = n_times
        self.train_percent = train_percent
        self.uso_solo_path = uso_solo_path
        self.elevation_file = elevation_file
        self.declive_file = declive_file
        self.grid = data_grid
        self.n_bandas = n_bandas
        self.n_days = n_days

        # Valores maximos e mínimos de cada banda
        self.max_mins = [
            [0, 400],    # Chuva
            [0, 5000],   # Vazão
            [0, 5000],   # Cota
            [-10, 40],   # Temperatura
            [0, 100],    # Umidade (0 a 100%)
            [0, 360],    # Dir. Vento
            [-10, 30],   # Ponto Orvalho
            [700, 1200], # Pressão
            [-10, 5000], # Radiação
            [0, 20],     # Velocidade do vento
            [pd.to_datetime("2000-01-01").timestamp(), pd.to_datetime("2024-12-31").timestamp()], # Data Min e Max
            [0, 1],      # Uso do solo não mudará (0, 1), categórico
            [0, 9E6],    # Declividade
            [-2, 1500],  # Altitude
        ]

        # Dados da cota no ponto de estudo
        self.cotas = pd.read_csv(r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\dados\dados.csv", low_memory=False)
        self.cotas.codigo = self.cotas.codigo.astype(str)
        self.cotas = self.cotas[(self.cotas.codigo == '87450005') & (self.cotas.tipo == "cota")]
        self.cotas = self.cotas[['date', 'value']]
        self.cotas.value = self.cotas.value.copy()/500 # Normalizando de 0 a 500 para 0 a 1
        self.cotas.loc[self.cotas["value"] < 0, "value"] = self.null_value
        self.cotas.date = pd.to_datetime(self.cotas.date)

        # Média dos valores observados
        self.mean_cotas = torch.tensor(np.nanmean(self.cotas.value.values))
        
        # Para armazenar o uso do solo anterior, caso os dados estejam no mesmo ano
        self.last_uso_solo = None
        self.last_uso_solo_year = None

        #Altitude e declividade
        min = self.max_mins[12][0]
        max = self.max_mins[12][1]
        self.declividade = self._read_raster(declive_file)
        self.declividade[self.declividade == -9999] = self.null_value
        self.declividade[(self.declividade < min) | (self.declividade > max)] = self.null_value
        self.declividade = (self.declividade - min)/(max-min)
        self.declividade[(self.declividade < 0) | (self.declividade > 1)] = self.null_value

        min = self.max_mins[13][0]
        max = self.max_mins[13][1]
        self.altitude = self._read_raster(elevation_file)
        self.altitude = (self.altitude - min)/(max-min)
        self.altitude[(self.altitude < 0) | (self.altitude > 1)] = self.null_value

        self.data_path = data_path
        self.files = os.listdir(data_path)
        
        # Range de datas que será utilizado
        self.date_range = self._generate_dates()

        # Iniciando dados aleatórios
        self.reset()

        # Adicionando a primeira leitura
        Thread(target=self._read_next).start()
        sleep(0.5) # Tempo para setar o lock

    def total(self):
        return int(np.ceil(self.indexes.size/self.batch_size))
    
    def total_train(self):
        return int(np.ceil(self.n_train_data/self.batch_size))
    
    def total_test(self):
        return int(np.ceil(self.n_test_data/self.batch_size))

    def next(self)->torch.Tensor:
        with self.lock:
            Thread(target=self._read_next, daemon=True, name=f"{self.step}").start()
            sleep(0.5) # Tempo para setar o lock

            return self.buffer

    def reset(self):
        self.step = 0
        self.indexes = np.arange(len(self.date_range))

        if(self.randomize):
            np.random.shuffle(self.indexes)
        
        self.n_train_data = int(np.floor(self.train_percent*self.indexes.size))
        self.n_test_data = int(self.indexes.size - self.n_train_data)

        self.train_indexes = self.indexes[0:self.n_train_data]
        self.test_indexes  = self.indexes[self.n_train_data:]

    def _read_next(self)->torch.Tensor:
        # Sinaliza que a leitura está em andamento
        self.lock.acquire()
        
        dados = torch.full((self.batch_size, self.n_times, self.n_bandas, self.grid[0], self.grid[1]), self.null_value, device="cuda", dtype=torch.float32)
        saidas = torch.full((self.batch_size, self.n_days), self.null_value, device="cuda", dtype=torch.float32)

        for batch in range(self.batch_size):
            index = self.indexes[self.step]
            actual_date = self.date_range[index]
            dates = pd.date_range(actual_date-timedelta(days=self.n_times-1), actual_date)
            dates_saida = pd.date_range(actual_date+timedelta(days=1), actual_date+timedelta(days=self.n_days))

            if self.last_uso_solo is None or actual_date.year != self.last_uso_solo_year:
                self.last_uso_solo_year = actual_date.year

                self.last_uso_solo = self._read_raster(self.uso_solo_path + "/output_" + str(actual_date.year) + ".tif")
                self.last_uso_solo = self.last_uso_solo.to(torch.float32)

                self.last_uso_solo[self.last_uso_solo==0] = self.null_value
                self.last_uso_solo = torch.where(self.last_uso_solo >= 90, self.null_value, self.last_uso_solo)

            falha = False
            for n, date in enumerate(dates):
                filename = date.strftime("%Y-%m-%d") + ".tiff"
                if filename in self.files:
                    dados_day = self._read_raster(self.data_path + "/" + filename)

                    # Normalizando os dados
                    min_max = torch.tensor(self.max_mins[0:dados_day.shape[0]], device="cuda")
                    min = min_max[:, 0].view(-1, 1, 1)
                    max = min_max[:, 1].view(-1, 1, 1)
                    dados_day = (dados_day - min)/(max-min)
                    dados_day[(dados_day < 0) | (dados_day > 1)] = self.null_value

                    dados[batch, n, 0:self.n_bandas-4] = dados_day
                else:
                    falha = True

                
                dados[batch, n, -1] = self.altitude
                dados[batch, n, -2] = self.declividade
                dados[batch, n, -3] = self.last_uso_solo

                min = self.max_mins[-4][0]
                max = self.max_mins[-4][1]
                dados[batch, n, -4] = (date.timestamp() - min)/(max-min)

            if not falha:
                for n, date in enumerate(dates_saida):
                    cota_date = self.cotas[self.cotas.date == date]
                    if cota_date.empty:
                        raise ValueError("Valor vazio aqui")
                    
                    saidas[batch, n] = torch.tensor(cota_date.value.values[0], device="cuda")

            self.step += 1

        self.buffer = (dados, saidas)

        # Sinaliza que a leitura foi concluída
        self.lock.release()

    def _read_raster(self, raster):
        data_raster = rioxarray.open_rasterio(raster)
        dados = torch.from_numpy(data_raster.values)
        return dados.to("cuda")

    def _generate_dates(self):
        dates = pd.to_datetime([i.split(".")[0] for i in self.files])
        set_dates = set(self.cotas.date)
        dates_return = []
        for date_now in dates:
            date_range = pd.date_range(date_now + timedelta(days=1), date_now + timedelta(days=self.n_days))
            if set(date_range).issubset(set_dates):
                dates_return.append(date_now)
        
        dates_return = pd.to_datetime(dates_return)
        dates_return.sort_values()
        return dates_return

