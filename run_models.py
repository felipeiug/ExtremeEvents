import os
import io
import torch.version
import pandas as pd

from tqdm import tqdm
from neuronios.NN import Rede3, Rede, Rede2
from datareader.data_reader import ReadRasters
from erros.metricas import CustomLoss, CustomLoss2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

os.system("cls")

device = "cpu"
torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_device("cuda")
else:
    print("CUDA não está disponível.")

modelo = 2     # Modelo utilizado no teste
batch_size = 3 # Processamento em paralelo
n_dias = 5     # Número de dias no futuro da previsão
n_temp = 10    # Número de tempos no passado LSTM (dias)
n_data = 14    # Número de bandas no raster <- Bandas dos rasters de dados, data, uso do solo, declividade e altitude
div_out = 4    # Os tamanhos X e Y serão dividos por div_out na convolução
size_x = 694   # Dimensão X do raster
size_y = 1198  # Dimensão Y do raster
bandas_time=[  # Bandas com dados temporais
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
]


print("Criando Leitor de Rasters")
reader = ReadRasters(
    data_path=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\rasters",
    data_grid=(size_x, size_y),
    n_bandas = n_data,
    uso_solo_path=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\uso_solo_reshape",
    elevation_file=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\elevacao\elevacao.tif",
    declive_file=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\elevacao\declividade.tif",
    batch_size=batch_size,
    n_times=n_temp,
    n_days=n_dias,
    train_percent=0.7,
    randomize=False,
)

print(f"Criando RNA com o modelo {modelo}")
if modelo == 1:
    cnn_lstm = Rede(
        batch_size  = batch_size,
        n_dias      = n_dias,
        n_tempos    = n_temp,
        n_bandas    = n_data,
        div_out     = div_out,
        size_x      = size_x,
        size_y      = size_y,
        bandas_time = bandas_time,
    )
elif modelo == 2:
    cnn_lstm = Rede2(
        batch_size  = batch_size,
        n_dias      = n_dias,
        n_tempos    = n_temp,
        n_bandas    = n_data,
        div_out     = div_out,
        size_x      = size_x,
        size_y      = size_y,
        bandas_time = bandas_time,
    )
elif modelo == 3:
    cnn_lstm = Rede3(
        batch_size  = batch_size,
        n_dias      = n_dias,
        n_tempos    = n_temp,
        n_bandas    = n_data,
        div_out     = div_out,
        size_x      = size_x,
        size_y      = size_y,
        bandas_time = bandas_time,
        null_val    = -99
    )

print("Importando modelo treinado")
modelos = os.listdir(f"modelos_{modelo}")

metricas = {
    "epoch":[],
    "LOSS":[],
    "REAL":[],
    "PRED":[],
    "NSE":[],
    "MSE":[],
}
if len(modelos) > 0:
    max_epoch = None
    last_loss = None
    for _modelo in tqdm(modelos, total=len(modelos), desc="Lendo Modelos"):
        file = f"modelos_{modelo}/{_modelo}"
        checkpoint = torch.load(file, weights_only=True)

        for key in metricas.keys():
            metricas[key].append(float(checkpoint[key]))

        if last_loss == None or (1-checkpoint['LOSS']) < last_loss:
            last_loss = checkpoint['LOSS']
            max_epoch = checkpoint['epoch']

    file = f"modelos_{modelo}/modelo_{max_epoch}.pth"

    checkpoint = torch.load(file, weights_only=True)
    cnn_lstm.load_state_dict(checkpoint['model_state_dict'])

pd.DataFrame(metricas).to_excel(f"epocas_modelo_{modelo}.xlsx")

# Treinamento
print("Iniciando Valores")

saidas = {
    "data":[],
}
for i in range(n_dias):
    saidas[f"R_t{i}"] = []
    saidas[f"P_t{i}"] = []

for ciclo in tqdm(range(reader.total()), total=reader.total(), desc=f"Processando dados treindos"):
    X, y = reader.next()

    if device != "cuda":
        X = X.to(device="cpu")
        y = y.to(device="cpu")

    outputs = cnn_lstm(X)

    for batch in range(reader.batch_size):
        index = (ciclo*reader.batch_size) + batch
        data = reader.date_range[index]

        saidas["data"].append(data)
        for i in range(n_dias):
            saidas[f"R_t{i}"].append(float(y[batch][i]))
            saidas[f"P_t{i}"].append(float(outputs[batch][i]))

pd.DataFrame(saidas).to_excel(f"valores_modelo_{modelo}.xlsx")

