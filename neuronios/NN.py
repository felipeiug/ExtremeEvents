import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .LSTM2D import LSTM2D

# Definir a CNN
class Rede(nn.Module):
    def __init__(self, batch_size = 1, n_dias=3, n_tempos = 10, n_bandas = 10, size_x = 1200, size_y = 1200, bandas_time:list[int]|None = None):
        super(Rede, self).__init__()

        self.n_dias = n_dias
        self.bandas_time = bandas_time if bandas_time is not None else [i for i in range(n_bandas)]
        self.batch_size = batch_size
        self.n_tempos = n_tempos
        self.n_bandas = n_bandas
        self.n_bandas_dynamic = len(bandas_time)
        self.n_bandas_static = n_bandas - len(bandas_time)
        self.size_x = size_x
        self.size_y = size_y

        self.dtype = torch.float32

        div_out = 4
        self.out_x = int(np.ceil(size_x/div_out)+2)
        self.out_y = int(np.ceil(size_y/div_out)+2)

        # Configurações das camadas CNN
        camadas_static = [
            nn.Conv2d(in_channels=self.n_bandas_static, out_channels=self.n_bandas_static*2, kernel_size=1, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=div_out, stride=div_out),
            nn.Conv2d(in_channels=self.n_bandas_static*2, out_channels=self.n_bandas_static, kernel_size=1, stride=1, padding=1),
        ]
        self.cnn_static = nn.Sequential(*camadas_static).to(dtype=self.dtype)

        camadas_dynamic = [
            nn.Conv2d(in_channels=self.n_bandas_dynamic, out_channels=self.n_bandas_dynamic*2, kernel_size=1, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=div_out, stride=div_out),
            nn.Conv2d(in_channels=self.n_bandas_dynamic*2, out_channels=self.n_bandas_dynamic, kernel_size=1, stride=1, padding=1),
        ]
        self.cnn_dynamic = nn.Sequential(*camadas_dynamic).to(dtype=self.dtype)

        # Configuração das camadas LSTM2D
        self.lstms2d = nn.ModuleList([
            LSTM2D(input_size=(self.out_x, self.out_y), timesteps=n_tempos, hidden_size=1)
            for _ in range(self.n_bandas_dynamic)
        ]).to(dtype=self.dtype)

        # Camadas lineares
        self.flatten_size = n_bandas * self.out_x * self.out_y
        self.fc1   = nn.Linear(in_features=self.flatten_size, out_features=64).to(dtype=self.dtype)
        self.fc2   = nn.Linear(in_features=64, out_features=64).to(dtype=self.dtype)
        self.relu1 = F.relu()
        self.fc3   = nn.Linear(in_features=64, out_features=16).to(dtype=self.dtype)
        self.fc4   = nn.Linear(in_features=16, out_features=n_dias).to(dtype=self.dtype)

    def forward(self, input:torch.Tensor):
        mask = torch.ones(input.shape[2], dtype=torch.bool, device=input.device)
        mask[self.bandas_time] = False

        input_dynamic = input[:, :, self.bandas_time]
        input_static = input[:, -1, mask]

        # Dados dinâmicos
        x_dynamic = self._forward_dynamic(input_dynamic)
        x_static = self._forward_static(input_static)

        x = torch.cat([x_dynamic, x_static], dim=1).to(dtype=self.dtype)

        # Rede neural totalmente conectada com um vetor de entrada
        x = x.view(self.batch_size, self.n_bandas * self.out_x * self.out_y).to(dtype=self.dtype)
        x = self.fc1(x).to(self.dtype)
        x = self.fc2(x).to(self.dtype)
        x = self.relu1(x)
        x = self.fc3(x).to(self.dtype)
        x = self.fc4(x).to(self.dtype)

        return x
    
    def _forward_dynamic(self, input: torch.Tensor):
        x = input.view(self.batch_size * self.n_tempos, len(self.bandas_time), self.size_x, self.size_y).to(self.dtype)
        x = self.cnn_dynamic(x)
        x = x.view(self.batch_size, self.n_tempos, len(self.bandas_time), self.out_x, self.out_y).to(self.dtype)
        
        # Para cada item na matriz de dados
        out_tensor = torch.zeros(self.batch_size, self.n_bandas_dynamic, self.out_x, self.out_y).to(self.dtype)
        for i in range(self.n_bandas_dynamic):
            input_data = x[:, :, i]
            output = self.lstms2d[i](input_data)
            output = output.mean(dim=-1)
            out_tensor[:, i] = output[:, -1]
        
        return out_tensor

    def _forward_static(self, input: torch.Tensor):
        input = input.to(self.dtype)
        x = self.cnn_static(input)
        return x


# Definir a CNN
class Rede2(nn.Module):
    def __init__(self, batch_size = 1, n_dias=3, n_tempos = 10, n_bandas = 10, div_out=4, size_x = 1200, size_y = 1200, bandas_time:list[int]|None = None):
        super(Rede2, self).__init__()

        self.n_dias = n_dias
        self.bandas_time = bandas_time if bandas_time is not None else [i for i in range(n_bandas)]
        self.batch_size = batch_size
        self.n_tempos = n_tempos
        self.n_bandas = n_bandas
        self.n_bandas_dynamic = len(bandas_time)
        self.n_bandas_static = n_bandas - len(bandas_time)
        self.size_x = size_x
        self.size_y = size_y

        self.dtype = torch.float32

        self.div_out = div_out
        self.out_x = int(np.ceil(size_x/div_out)+2)
        self.out_y = int(np.ceil(size_y/div_out)+2)

        # Configurações das camadas CNN
        camadas_cnn = [
            nn.Conv2d(in_channels=self.n_bandas, out_channels=self.n_bandas*2, kernel_size=1, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=div_out, stride=div_out),
            nn.Conv2d(in_channels=self.n_bandas*2, out_channels=self.n_bandas, kernel_size=1, stride=1, padding=1),
        ]
        self.cnn = nn.Sequential(*camadas_cnn).to(dtype=self.dtype)

        # LSTM's das camadas dinâmicas (1 LSTM para cada banda dinâmica)
        self.lstm_dynamic = nn.ModuleList([
            nn.LSTM(
                input_size=1,
                batch_first=True,
                hidden_size=1,
            ) for _ in range(self.n_bandas_dynamic)
        ])

        # Camadas lineares
        self.flatten_size = n_bandas * self.out_x * self.out_y

        self.fc1   = nn.Linear(in_features=self.flatten_size, out_features=64).to(dtype=self.dtype)
        self.fc2   = nn.Linear(in_features=64, out_features=64).to(dtype=self.dtype)
        self.relu1 = nn.ReLU()
        self.fc3   = nn.Linear(in_features=64, out_features=16).to(dtype=self.dtype)
        self.fc4   = nn.Linear(in_features=16, out_features=n_dias).to(dtype=self.dtype)

    def forward(self, input:torch.Tensor):
        x = input.view(self.batch_size*self.n_tempos, self.n_bandas, self.size_x, self.size_y)
        x = self.cnn(x)
        x = x.view(self.batch_size, self.n_tempos, self.n_bandas, self.out_x, self.out_y)

        mask = torch.ones(self.n_bandas, dtype=torch.bool, device=input.device)
        mask[self.bandas_time] = False

        input_dynamic = x[:, :, self.bandas_time]
        input_static  = x[:, -1, mask]

        # Dados dinâmicos
        x_dynamic = self._forward_dynamic(input_dynamic)
        x_static = self._forward_static(input_static)

        x = torch.cat([x_dynamic, x_static], dim=1).to(dtype=self.dtype)

        # Rede neural totalmente conectada com um vetor de entrada
        x = x.view(self.batch_size, self.n_bandas * self.out_x * self.out_y).to(dtype=self.dtype)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x
    
    def _forward_dynamic(self, input: torch.Tensor):
        x = input.view(self.batch_size*self.out_x*self.out_y, self.n_tempos, self.n_bandas_dynamic).unsqueeze(-1)

        saidas = torch.zeros(self.batch_size, self.n_bandas_dynamic, self.out_x, self.out_y, dtype=self.dtype)
        for band in range(self.n_bandas_dynamic):
            out, _ = self.lstm_dynamic[band](x[:, :, band])
            out = out.squeeze(-1)[:, -1].view(self.batch_size, self.out_x, self.out_y)
            saidas[:, band] = out

        return saidas

    def _forward_static(self, input: torch.Tensor):
        # Caso precise fazer algo no forward static
        return input

# Definir a CNN
class Rede3(nn.Module):
    def __init__(self, batch_size = 1, n_dias=3, n_tempos = 10, n_bandas = 10, div_out=4, size_x = 1200, size_y = 1200, bandas_time:list[int]|None = None, null_val = -99):
        super(Rede3, self).__init__()

        self.n_dias = n_dias
        self.bandas_time = bandas_time if bandas_time is not None else [i for i in range(n_bandas)]
        self.batch_size = batch_size
        self.n_tempos = n_tempos
        self.n_bandas = n_bandas
        self.n_bandas_dynamic = len(bandas_time)
        self.n_bandas_static = n_bandas - len(bandas_time)
        self.size_x = size_x
        self.size_y = size_y
        self.null_value = null_val

        self.dtype = torch.float32

        self.div_out = div_out
        self.out_x = int(np.ceil(size_x/div_out)+2)
        self.out_y = int(np.ceil(size_y/div_out)+2)

        # Configurações das camadas CNN
        camadas_cnn = [
            nn.Conv2d(in_channels=self.n_bandas, out_channels=self.n_bandas*2, kernel_size=1, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=div_out, stride=div_out),
            nn.Conv2d(in_channels=self.n_bandas*2, out_channels=self.n_bandas, kernel_size=1, stride=1, padding=1),
        ]
        self.cnn = nn.Sequential(*camadas_cnn).to(dtype=self.dtype)

        # LSTM's das camadas dinâmicas (1 LSTM para cada banda dinâmica)
        self.lstm_dynamic = nn.ModuleList([
            nn.LSTM(
                input_size=1,
                batch_first=True,
                hidden_size=50,
            ) for _ in range(self.n_bandas_dynamic)
        ])

        # Camadas lineares
        self.flatten_size = n_bandas * self.out_x * self.out_y + self.n_bandas_dynamic

        self.fc1   = nn.Linear(in_features=self.flatten_size, out_features=128).to(dtype=self.dtype)
        self.fc2   = nn.Linear(in_features=128, out_features=64).to(dtype=self.dtype)
        self.relu1 = nn.ReLU()
        self.fc3   = nn.Linear(in_features=64, out_features=32).to(dtype=self.dtype)
        self.fc4   = nn.Linear(in_features=32, out_features=16).to(dtype=self.dtype)
        self.fc5   = nn.Linear(in_features=16, out_features=n_dias).to(dtype=self.dtype)

    def forward(self, input:torch.Tensor):
        input_dynamic = input[:, :, self.bandas_time]
        x_dynamic = self._forward_dynamic(input_dynamic)

        x = input[:, -1]
        x = self.cnn(x)

        import time
        time.sleep(10)

        # Rede neural totalmente conectada com um vetor de entrada
        x = x.view(self.batch_size, self.n_bandas * self.out_x * self.out_y).to(dtype=self.dtype)
        x = torch.cat([x, x_dynamic], dim=1).to(dtype=self.dtype)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)

        return x
    
    def _forward_dynamic(self, input: torch.Tensor):
        saidas = torch.zeros(self.batch_size, self.n_bandas_dynamic, dtype=self.dtype)

        input[input == self.null_value] = torch.nan
        mean = input.nanmean(dim=(3, 4)).unsqueeze(-1)
        for band in range(self.n_bandas_dynamic):
            out, _ = self.lstm_dynamic[band](mean[:, :, band])
            out = out[:, -1, -1]
            saidas[:, band] = out

        return saidas

    def _forward_static(self, input: torch.Tensor):
        # Caso precise fazer algo no forward static
        return input

