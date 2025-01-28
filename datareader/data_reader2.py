import os
import torch
import psutil
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd

# from numpy._core._exceptions import _ArrayMemoryError
from scipy.optimize import curve_fit
from threading import Thread, Lock
from datetime import timedelta
from copy import deepcopy
from time import sleep
from tqdm import tqdm

uso_solo_str = """1. Floresta 1 #32a65e
1.1 Formação Florestal 3 #1f8d49
1.2. Formação Savânica 4 #7dc975
1.3. Mangue 5 #04381d
1.4. Floresta Alagável 6 #026975
1.5. Restinga Arbórea 49 #02d659
2. Formação Natural não Florestal 10 #ad975a
2.1. Campo Alagado e Área Pantanosa 11 #519799
2.2. Formação Campestre 12 #d6bc74
2.3. Apicum 32 #fc8114
2.4. Afloramento Rochoso 29 #ffaa5f
2.5. Restinga Herbácea 50 #ad5100
2.6. Outras Formações não Florestais 13 #d89f5c
3. Agropecuária 14 #FFFFB2
3.1. Pastagem 15 #edde8e
3.2. Agricultura 18 #E974ED
3.2.1. Lavoura Temporária 19 #C27BA0
3.2.1.1. Soja 39 #f5b3c8
3.2.1.2. Cana 20 #db7093
3.2.1.3. Arroz 40 #c71585
3.2.1.4. Algodão (beta) 62 #ff69b4
3.2.1.5. Outras Lavouras Temporárias 41 #f54ca9
3.2.2. Lavoura Perene 36 #d082de
3.2.2.1. Café 46 #d68fe2
3.2.2.2. Citrus 47 #9932cc
3.2.2.3. Dendê 35 #9065d0
3.2.2.4. Outras Lavouras Perenes 48 #e6ccff
3.3. Silvicultura 9 #7a5900
3.4. Mosaico de Usos 21 #ffefc3
4. Área não Vegetada 22 #d4271e
4.1. Praia, Duna e Areal 23 #ffa07a
4.2. Área Urbanizada 24 #d4271e
4.3. Mineração 30 #9c0027
4.4. Outras Áreas não Vegetadas 25 #db4d4f
5. Corpo D'água 26 #0000FF
5.1 Rio, Lago e Oceano 33 #2532e4
5.2 Aquicultura 31 #091077
6. Não observado 27 #ffffff"""

uso_solo_legenda = {}

for uso in uso_solo_str.split("\n"):
    partes = uso.split(" ")
    indice = partes[0]
    index = partes[-2]
    color = partes[-1]
    nome = " ".join(partes[1:-2])

    uso_solo_legenda[int(index)] = (indice, int(index), color, nome)

# Área de drenagem das estações
area_drenagem_estacoes = {
    "85350000": 664.7545, #km²
    "85395100": 13308.4828, #km²
    "85400000": 14036.4876, #km²
    "85438000": 1122.8741, #km²
    "85470000": 1027.9535, #km²
    "85480000": 2985.3806, #km²
    "85600000": 6847.0732, #km²
    "85610000": 109.0171, #km²
    "85623000": 666.4098, #km²
    "85642000": 27458.4351, #km²
    "85735000": 52.1176, #km²
    "85830000": 956.2804, #km²
    "85900000": 38829.3369, #km²
    "86099000": 35.5170, #km²
    "86100000": 35.5170, #km²
    "86100600": 151.8334, #km²
    "86160000": 1185.9869, #km²
    "86305000": 7768.3210, #km²
    "86401000": 1148.0476, #km²
    "86410000": 2860.5995, #km²
    "86420000": 541.6539, #km²
    "86440000": 3653.8083, #km²
    "86448000": 12134.3339, #km²
    "86470000": 12532.9414, #km²
    "86470800": 12788.7756, #km²
    "86472000": 12996.7139, #km²
    "86480000": 1278.9780, #km²
    "86490500": 1610.2594, #km²
    "86500000": 1825.3772, #km²
    "86505500": 2334.7234, #km²
    "86510000": 16047.2001, #km²
    "86560000": 2031.1593, #km²
    "86580000": 2488.6049, #km²
    "86700000": 441.2455, #km²
    "86720000": 19137.4844, #km²
    "86745000": 792.2009, #km²
    "86895000": 24660.0840, #km²
    "87150000": 1352.0183, #km²
    "87160000": 2023.6512, #km²
    "87168000": 268.0074, #km²
    "87170000": 3016.7668, #km²
    "87230000": 900.0837, #km²
    "87270000": 4365.5274, #km²
    "87374000": 1476.5764, #km²
    "87380000": 2957.7422, #km²
    "87382000": 3145.7686, #km²
    "87399000": 1319.3502, #km²
}

# Bandas Raster
# 01 - Chuva
# 02 - Vazão
# 03 - Cota
# 04 - Temperatura
# 05 - Umidade (0 a 100%)
# 06 - Dir. Vento
# 07 - Ponto Orvalho
# 08 - Pressão
# 09 - Radiação
# 10 - Velocidade do vento

class ReadRasters:
    step = 0
    lock = Lock()   # Lock para sincronização
    buffer = None
    rasters = {}
    n_usos_rasters = {}

    def __init__(self,
        n_days,
        data_path,
        data_grid,
        uso_solo_path,
        elevation_file,
        declive_file,
        batch_size,
        n_times,
        train_percent,
        min_date="2000-01-01",
        normalize=True,
        randomize=True,
        null_value = -99,
        ram_use_percentage = 0.7,
        threaded = True,
    ):
        """
        Os dados do dia N são os dados do dia N-1 ao dia `N-1-n_times`
        Todos os dias sem dados serão substituidos por uma matriz equivalente com os valores null_value em todo os 
        dados indisponíveis
        """

        read_vazoes = False

        self.ram_use_percentage = ram_use_percentage
        self.threaded = threaded
        self.randomize = randomize
        self.null_value = null_value
        self.batch_size = batch_size
        self.n_times = n_times
        self.train_percent = train_percent
        self.uso_solo_path = uso_solo_path
        self.elevation_file = elevation_file
        self.declive_file = declive_file
        self.grid = data_grid
        self.n_days = n_days
        self.min_date = pd.to_datetime(min_date)

        # Uso do solo
        self.uso_solo = {}
        usos_solo = os.listdir(self.uso_solo_path)
        for uso in tqdm(usos_solo, desc="Lendo usos do solo", total=len(usos_solo)):
            year = int(uso.split("_")[1].split(".")[0])

            uso_solo = self._read_raster(self.uso_solo_path + "/" + uso)
            uso_solo[(uso_solo==0)] = 27 # Valor sem observação
            uso_solo = torch.where(uso_solo >= 90, self.null_value, uso_solo)
            uso_solo = uso_solo.to(torch.float32)

            self.uso_solo[year] = uso_solo

        self.tipos_uso_solo = torch.tensor(list(uso_solo_legenda.keys()), device=uso_solo.device, dtype=uso_solo.dtype)

        # Valores maximos e mínimos de cada banda
        if normalize:
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
                [self.min_date.timestamp(), pd.to_datetime("2024-12-31").timestamp()], # Data Min e Max
                [0, 1],      # Uso do solo não mudará (0, 1), categórico
                [0, 9E6],    # Declividade
                [-2, 1500],  # Altitude
            ]
        else:
            self.max_mins = [
                [0, 1], # Chuva
                [0, 1], # Vazão
                [0, 1], # Cota
                [0, 1], # Temperatura
                [0, 1], # Umidade (0 a 100%)
                [0, 1], # Dir. Vento
                [0, 1], # Ponto Orvalho
                [0, 1], # Pressão
                [0, 1], # Radiação
                [0, 1], # Velocidade do vento
                [0, 1], # Data Min e Max
                [0, 1], # Uso do solo
                [0, 1], # Declividade
                [0, 1], # Altitude
            ]
            
        self.dados = pd.read_csv(r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\dados\dados.csv", low_memory=False)

        # Dados da cota no ponto de estudo
        self.cotas = self.dados.copy()
        self.cotas.codigo = self.cotas.codigo.astype(str)
        self.cotas = self.cotas[(self.cotas.codigo == '87450005') & (self.cotas.tipo == "cota")]
        self.cotas = self.cotas[['date', 'value']]
        self.cotas.loc[self.cotas["value"] < 0, "value"] = self.null_value
        self.cotas.date = pd.to_datetime(self.cotas.date)
        
        # Média dos valores observados
        self.mean_cotas = torch.tensor(np.nanmean(self.cotas.value.values))

        # Dados das vazões na bacia
        if read_vazoes:
            self.vazoes = self.dados.copy()
            self.vazoes = self.vazoes[(self.vazoes.tipo == "vazao")]
            self.vazoes["area"] = None
            for i, row in tqdm(self.vazoes.iterrows(), desc="Processando dados de vazão", total=len(self.vazoes)):
                code = row.codigo
                area = area_drenagem_estacoes[code]
                self.vazoes.at[i, "area"] = area
            self.vazoes = self.vazoes[['date', 'value', "area"]]
            self.vazoes.loc[self.vazoes["value"] < 0, "value"] = self.null_value
            self.vazoes.date = pd.to_datetime(self.vazoes.date)

        # Removendo os dados
        del self.dados

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
        self.altitude[(self.altitude < 0)] = self.null_value

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

    def reset(self):
        self.rasters = {}

        self.step = 0
        self.indexes = np.arange(len(self.date_range))

        self.indexes = self.indexes[self.date_range >= self.min_date]

        if(self.randomize):
            np.random.shuffle(self.indexes)
        
        self.n_train_data = int(np.floor(self.train_percent*self.indexes.size))
        self.n_test_data = int(self.indexes.size - self.n_train_data)

        self.train_indexes = self.indexes[0:self.n_train_data]
        self.test_indexes  = self.indexes[self.n_train_data:]

    def next(self)->torch.Tensor:
        if self.threaded:
            with self.lock:
                t = Thread(target=self._read_next, daemon=True, name=f"{self.step}")
                t.start()
                sleep(0.5) # Tempo para setar o lock

                if isinstance(self.buffer, BaseException):
                    raise self.buffer
                
                return self.buffer
        else:
            buff = deepcopy(self.buffer)
            self._read_next()
            return buff

    def _read_next(self)->torch.Tensor:
        """### Formato dos dados:
        Os dados serão retornados em dois tensores, o primeiro contendo os dados de entrada e o segundo a saída esperada do modelo
        #### Entrada:
        Conteúdo: Chuva x Vazão x Temperatura x Umidade x Radiação x Declividade x Altitude x Uso do Solo
        - Tensor (Batch Size, N Tempos, (8 + N Uso Solo), Grid X, Grid Y)
        `Descrição`: Tensor com uma matriz de dados para cada dia anterior a data da análise.
        
        Conteúdo: (Ano - 2000), Jan, Fev, ... , Nov, Dez
        - Tensor (Batch Size, N Tempos, 13)
        `Descrição`: Tensor com dados para cada data do dia.

        #### Returns:
        Conteúdo: Tensor
        - cotas
        `Descrição`: Tensor com os valores de cota para os dias de simulação.
        """

        # Sinaliza que a leitura está em andamento
        self.lock.acquire()

        try:    
            # Chuva x Vazão x Temperatura x Umidade x Radiação x Declividade x Altitude x Uso do Solo
            matrix_data = torch.full((self.batch_size, self.n_times, 7 + len(uso_solo_legenda), self.grid[0], self.grid[1]), self.null_value, device="cpu", dtype=torch.float32)
            
            # Data
            vetor_data  = torch.full((self.batch_size, 13), self.null_value, device="cpu", dtype=torch.float32)
            vetor_data[:, 0:13] = 0

            # Saídas
            cotas = torch.full((self.batch_size, self.n_days), self.null_value, device="cpu", dtype=torch.float32)

            for batch in range(self.batch_size):
                if self.train_indexes.shape[0] <= self.step:
                    break
                
                # Data de Processamento
                actual_date:pd.DatetimeIndex = self.date_range[self.train_indexes[self.step]]

                # Data
                vetor_data[batch, 0] = (actual_date.year-2000)
                vetor_data[batch, actual_date.month] = 1

                ### Uso do solo
                uso_solo = self.uso_solo[actual_date.year]

                # Para cada passo de tempo em cada posição da matriz e para cada tipo de uso do solo,
                # atribuo 1 para contém e 0 para não contém

                # Converte as chaves de uso_solo_legenda em um tensor
                mascaras = (uso_solo.unsqueeze(0) == self.tipos_uso_solo.view(-1, 1, 1)).to(torch.float32)
                matrix_data[batch, :, 7:7 + len(uso_solo_legenda)] = mascaras

                # Data dos dados de entrada e saída
                dates = pd.date_range(actual_date-timedelta(days=self.n_times-1), actual_date)
                dates_saida = pd.date_range(actual_date+timedelta(days=1), actual_date+timedelta(days=self.n_days))

                del actual_date

                # Obtendo os valores dos rasteres
                falha = False
                dados_day = None
                
                for n, date in enumerate(dates):
                    date = date.strftime("%Y-%m-%d")

                    # Verificando a ocorrência de dados
                    if self.rasters[date] is None or falha:

                        # Removendo dados não utilizandos
                        self.n_usos_rasters[date] -= 1
                        if self.n_usos_rasters[date] <= 0:
                            self.n_usos_rasters.pop(date)
                            self.rasters.pop(date)

                        falha = True
                        continue

                    # Lendo arquivo de dados
                    dados_day = self.rasters[date]
                    
                    # Dados da Matriz
                    # Chuva (B01) | Vazão (B02) | Temp. Méd. (B04) | Umidade Med. (B05) | Pressão Med. (B08) | Radiação Med. (B09)
                    matrix_data[batch, n, 0:5] = dados_day[[0, 1, 3, 4, 8]]

                    # Declividade e altitude
                    matrix_data[batch, n, 5] = self.declividade
                    matrix_data[batch, n, 6] = self.altitude
                del dados_day, n, date, dates
                

                # Dados de Saída
                if not falha:
                    for n, date in enumerate(dates_saida):
                        cota_date = self.cotas[self.cotas.date == date]
                        if cota_date.empty:
                            raise ValueError("Valor vazio aqui")
                        cotas[batch, n] = torch.tensor(cota_date.value.values[0], device="cpu")
                    del n, date, cota_date

                self.step += 1

            self.buffer = ((matrix_data, vetor_data), cotas)
        except Exception as e:
            self.buffer = e

        # Sinaliza que a leitura foi concluída
        self.lock.release()

    def _buffer_rasters(self):
        # Limpando o terminal
        os.system("cls")

        # Informação do que esta carregando o buffer
        print("Adicionando dados ao Buffer", end="")

        while psutil.virtual_memory().percent >= self.ram_use_percentage*100 and len(self.rasters) > 0:
            self.rasters.pop(list(self.rasters.keys())[0])

        i = self.step

        while True:
            if psutil.virtual_memory().percent >= self.ram_use_percentage*100:
                break

            actual_date = self.date_range[self.train_indexes[i]]
            dates = pd.date_range(actual_date-timedelta(days=self.n_times-1), actual_date)

            for date in dates:
                # Caso o dado já esteja nos dados
                if date in self.rasters and date in self.n_usos_rasters:
                    self.n_usos_rasters += 1
                    continue
                
                # Arquivo desta data
                filename = date + ".tiff"

                # Caso o dado não exista na base de dados
                if filename not in self.files:
                    self.rasters[date] = None
                    self.n_usos_rasters[date] = 1
                    continue

                raster = self._read_raster(self.data_path + "/" + filename)
                self.rasters[date] = raster
                self.n_usos_rasters[date] = 1

            print(".", end="")

            i+= 1
        print()

    def _read_raster(self, raster):
        data_raster = rioxarray.open_rasterio(raster)
        dados = torch.from_numpy(data_raster.values).to("cpu")
        return dados

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

