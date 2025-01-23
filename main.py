# from run_models import *
# exit()

import os
import io
import torch.version
from tqdm import tqdm
from neuronios.NN import *
from datareader.data_reader2 import ReadRasters
from erros.metricas import CustomLoss3

import torch
import torch.optim as optim

# Apenas para gerar os resultados
# from run_models import *

os.system("cls")

load = False # Se irei utilizar o modelo salvo
device = "cpu"
torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_device("cuda")
else:
    print("CUDA não está disponível.")

modelo = 4     # Modelo utilizado no teste
min_date = "2000-01-01" # Data mínima para treino
batch_size = 3 # Processamento em paralelo
n_dias = 7     # Número de dias no futuro da previsão
n_temp = 10    # Número de tempos no passado LSTM (dias)
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
    uso_solo_path=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\uso_solo_reshape",
    elevation_file=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\elevacao\elevacao.tif",
    declive_file=r"D:\Mestrado\2º Semestre\Extreme Events\Codigos2\elevacao\declividade.tif",
    batch_size=batch_size,
    n_times=n_temp,
    n_days=n_dias,
    train_percent=0.8,
    randomize=False,
    min_date=min_date,
    normalize=False,
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
elif modelo == 4:
    cnn_lstm = Rede4(
        n_future      = n_dias,
        n_past    = n_temp,
        size_x      = size_x,
        size_y      = size_y,
    )

print("Criando Otimizador")
optimizer = optim.AdamW(cnn_lstm.parameters(), lr=0.001)

loss = 0
start_epoch = 0
last_losses = []

os.makedirs(f"modelos_{modelo}", exist_ok=True)
if load:
    print("Importando modelos treinados")
    modelos = os.listdir(f"modelos_{modelo}")
    if len(modelos) > 0:
        for _modelo in modelos:
            file = f"modelos_{modelo}/{_modelo}"
            checkpoint = torch.load(file, weights_only=True)
            last_losses.append(checkpoint['LOSS'])

        epochs = [int(i.split("_")[-1].split(".")[0]) for i in modelos]
        max_epoch = max(epochs)

        file = f"modelos_{modelo}/modelo_{max_epoch}.pth"

        checkpoint = torch.load(file, weights_only=True)

        cnn_lstm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        loss = checkpoint['LOSS']

# Métrica de Erro
print("Criando Métricas de Erro")
criterion = CustomLoss3(
    area_bacia=82448,
    media=reader.mean_cotas,
    last_losses=last_losses,
    start_epoch=start_epoch
).to(torch.float32)

# Treinamento
epoch = start_epoch
print("Iniciando Treino")

while True:
    reader.reset()
    cnn_lstm.train()

    criterion.epoch = epoch
    criterion.losses_epoch = []

    buffer = io.StringIO()
    progress_bar = tqdm(range(reader.total_train()), file=buffer, total=reader.total_train(), desc=f"LOSS: {loss:.4g} | Epoch: {epoch}")
    for step in progress_bar:
        X, y = reader.next()

        if (y[0] == -99).any().item():
            continue

        # Forward pass
        outputs = cnn_lstm(X)
        loss = criterion(outputs, y)
        
        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #description = " | ".join([f" {name.upper()}: {f'{erro:.4g}' if erro is not None else None}" for name, erro in criterion.erros.items()])
        description = f"LOSS: {criterion.erros["LOSS"]:.4g}"
        description += f" | Epoch: {epoch+1}"

        progress_bar.set_description(description)
        criterion.print(buffer.getvalue())

    criterion.last_losses.append(criterion.erros["LOSS"])

    dados_save = {
        'epoch': epoch,
        'model_state_dict': cnn_lstm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    for name, erro in criterion.erros.items():
        dados_save[name] = erro

    torch.save(dados_save, f"modelos_{modelo}/modelo_{epoch}.pth")

    epoch += 1



# Gerar visualização
# from torchviz import make_dot

# X, y = reader.next()

# X = X.to(dtype=torch.float32)
# y = y.to(dtype=torch.float32)

# if device != "cuda":
#     X = X.to(device="cpu")
#     y = y.to(device="cpu")
    
# output = cnn_lstm(X)

# # Gerar o grafo
# graph = make_dot(output, params=dict(cnn_lstm.named_parameters()))

# # Salvar como imagem
# graph.render("rna_config", format="pdf")
