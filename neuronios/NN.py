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

# Definir a CNN
class Rede4(nn.Module):
    def __init__(
            self,
            batch_first = True,
            n_future=7, n_past = 10,
            size_x = 1200,
            size_y = 1200,
            null_val = -99,
            cnn_config = None
        ):
        super(Rede4, self).__init__()

        self.n_future = n_future
        self.batch_first = batch_first
        self.n_past = n_past
        self.size_x = size_x
        self.size_y = size_y
        self.null_value = null_val

        self.dtype = torch.float32

        if cnn_config is None:
            cnn_config = [
                {
                    "type":"Conv2d",
                    "out_channels": 15,
                    "kernel_size": 2,
                    "stride": 2,
                    "padding": 0,
                    "dilation": 1,
                },
                {
                    "type":"Sigmoid"
                },
                {
                    "type":"MaxPool2d",
                    "kernel_size":2,
                    "stride": 2,
                    "padding": 0,
                    "dilation": 1,
                },
                {
                    "type":"Conv2d",
                    "out_channels": 8,
                    "kernel_size": 2,
                    "stride": 2,
                    "padding": 0,
                    "dilation": 1,
                },
                {
                    "type":"Sigmoid",
                },
                {
                    "type":"Conv2d",
                    "out_channels": 1,
                    "kernel_size": 2,
                    "stride": 2,
                    "padding": 0,
                    "dilation": 1,
                },
                {
                    "type":"Sigmoid",
                },
            ]

        # Configurações das camadas CNN
        camadas_cnn = []
        out_x = size_x
        out_y = size_y
        out_channels = 42
        for value in cnn_config:
            tipo = value.pop("type")
            camada = getattr(nn, tipo)

            if 'out_channels' in value:
                value["in_channels"] = out_channels

            try:
                out_x, out_y = self._calc_cnn_out(out_x, out_y, **value)
            except TypeError as e:
                pass

            camadas_cnn.append(camada(**value))

            if 'out_channels' in value:
                out_channels = value['out_channels']
        self.cnn = nn.Sequential(*camadas_cnn).to(dtype=self.dtype)
        
        # Tamanho do vetor de saída
        self.out_x = out_x
        self.out_y = out_y
        self.out_channels = out_channels

        # Camada que obtém a partir da convolução um valor estimado de vazão média
        self.linear_vazao = nn.Sequential(*[
            nn.Linear(in_features=out_x*out_y, out_features=32).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1).to(dtype=self.dtype),
        ])

        ##### LSTM para os 5 dados, Vazao (estimada na CNN), Temp. Méd., Umidade Med., Pressão Med., Radiação Med.  #####
        self.lstm = nn.LSTM(
            input_size=5,
            batch_first=True,
            hidden_size=30,
        )
        
        # Camadas lineares
        # 17 bandas de dados mais a vazão estimada pela convolução e o resultado da LSTM
        self.flatten_size = 19
        
        self.linear = nn.Sequential(*[
            nn.Linear(in_features=self.flatten_size, out_features=64).to(dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=64).to(dtype=self.dtype),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=32).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16).to(dtype=self.dtype),
            nn.Linear(in_features=16, out_features=n_future).to(dtype=self.dtype)
        ])
    
    def _calc_cnn_out(self, h_in, w_in, kernel_size, stride, padding, dilation, **kwargs):
        h = ((h_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        w = ((w_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        return (int(h), int(w))


    def forward(self, input:tuple[torch.Tensor]):
        """## Saída do modelo
        Tupla contendo:
        - Cota no ponto de estudo para cada um dos dias de simulação
        - Vazão média na bacia toda (para a área de 82.000 km²) para cada um dos dias de simulação
        """

        matrix = input[0]
        linear = input[1]

        batch_size = input[0].shape[0]

        # Obtendo a vazão média com o a convolução
        x_vazao = torch.zeros(batch_size, self.n_past, dtype=self.dtype)
        for n in range(self.n_past):
            x = matrix[:, n]
            x = self.cnn(x)
            x = x.squeeze(dim=1)        # Shape: (batch, X, Y)
            x = x.view(batch_size, -1)  # Shape: (batch, X * Y)
            x = self.linear_vazao(x)    # MLP das vazões
            x = x.squeeze()
            x_vazao[:, n] = x
        del x

        x_lstm = torch.zeros(batch_size, self.n_past, 5)
        for n in range(4):
            x_lstm[:, :, n] = linear[:, :, n]
        x_lstm[:, :, 4] = x_vazao

        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = x_lstm[:, -1, -1]

        # Rede neural totalmente conectada com um vetor de entrada
        # (Saída LSTM, Vazão, Temp. Méd., Umidade Med., Pressão Med., Radiação Med., (Ano - 2000), Jan, Fev, ... , Nov, Dez)
        # Porém apenas para o último dia
        x_linear = torch.zeros(batch_size, 19, dtype=self.dtype)

        for n in range(linear.shape[2]):
            x_linear[:, n] = linear[:, -1, n]  # Dados do para cada input da entrada
        
        x_linear[:, -2] = x_vazao[:, -1] # Dados do LSTM
        x_linear[:, -1] = x_lstm   # Dados da vazãoo estimada

        x = self.linear(x_linear) # MLP das cotas
        
        return (x, x_vazao)

# Rede que transforma chuva, vazao e temperatura na vazão.
class Rede5(nn.Module):
    def __init__(
            self,
            batch_first = True,
            n_future=7, n_past = 10,
            size_x = 1200,
            size_y = 1200,
            null_val = -99,
        ):
        super(Rede5, self).__init__()

        self.n_future = n_future
        self.batch_first = batch_first
        self.n_past = n_past
        self.size_x = size_x
        self.size_y = size_y
        self.null_value = null_val

        self.dtype = torch.float32

        config_cnn = [
            {
                "type":"Conv2d",
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
            },
            {
                "type":"ReLU"
            },
            {
                "type":"MaxPool2d",
                "kernel_size":3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
            },
            {
                "type":"Conv2d",
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
            },
            {
                "type":"ReLU",
            },
            {
                "type":"MaxPool2d",
                "kernel_size":2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
            },
        ]

        # CNN Chuva
        camadas_cnn = []
        out_x = size_x
        out_y = size_y
        out_channels = 45
        for value in config_cnn:
            tipo = value.pop("type")
            camada = getattr(nn, tipo)

            if 'out_channels' in value:
                value["in_channels"] = out_channels

            try:
                out_x, out_y = self._calc_cnn_out(out_x, out_y, **value)
            except TypeError as e:
                pass

            camadas_cnn.append(camada(**value))

            if 'out_channels' in value:
                out_channels = value['out_channels']
        
        self.out_x = out_x
        self.out_y = out_y
        self.out_channels = out_channels
        self.cnn = nn.Sequential(*camadas_cnn).to(dtype=self.dtype)
        
        ##### LSTM para cada pixel da saída e para cada um dos dois valores, Chuva e Vazão.  #####
        self.lstm = nn.LSTM(
            input_size=self.out_x*self.out_y*self.out_channels,
            batch_first=True,
            hidden_size=64, # <- Entra no linear
        )

        # Camada que obtém a partir da convolução um valor estimado de vazão média
        # Entrada igual a saida do LSTM mais 13, 13 da data atual, ano e mês.
        self.fc = nn.Sequential(*[
            nn.Linear(in_features=64+13, out_features=32).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=7).to(dtype=self.dtype),
        ])
    
    def _calc_cnn_out(self, h_in, w_in, kernel_size, stride, padding, dilation, **kwargs):
        h = ((h_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        w = ((w_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        return (int(h), int(w))


    def forward(self, input: tuple[torch.Tensor]):
        """
        chuva: Tensor (batch_size, n_past, height, width)
        vazao: Tensor (batch_size, n_past, height, width)

        ### Returns
        vazao: Tensor (batch_size, n_past)
        """

        x = input[0]
        dates = input[1]
        batch_size = input[0].shape[0]

        # Processar com CNN
        x = x.view(batch_size * self.n_past, x.shape[2], x.shape[3], x.shape[4])
        x:torch.Tensor = self.cnn(x)
        x = x.view(batch_size, self.n_past, -1)  # (batch_size, n_past, cnn_out_features)

        # Processar sequências de chuvas com LSTM
        x, (_, _) = self.lstm(x)  # h_n_chuva: (1, batch_size, 64)

        x = x[:, -1, :] #(batch_size, LSTM size)
        x = torch.cat([x, dates], dim=1) # (batch_size, LSTM size + 13)

        # Passar pelas camadas totalmente conectadas
        x = self.fc(x)  # (batch_size, n_future)

        return x

# Rede que transforma chuva, vazao e temperatura na vazão.
class Rede6(nn.Module):
    def __init__(
            self,
            batch_first = True,
            n_future=7, n_past = 10,
            size_x = 1200,
            size_y = 1200,
            null_val = -99,
        ):
        super(Rede6, self).__init__()

        self.n_future = n_future
        self.batch_first = batch_first
        self.n_past = n_past
        self.size_x = size_x
        self.size_y = size_y
        self.null_value = null_val

        self.dtype = torch.float32

        config_cnn = [
            {
                "type":"Conv2d",
                "out_channels": 32,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
            },
            {
                "type":"ReLU"
            },
            {
                "type":"MaxPool2d",
                "kernel_size":3,
                "stride": 2,
                "padding": 0,
                "dilation": 1,
            },
            {
                "type":"Conv2d",
                "out_channels": 16,
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
            },
            {
                "type":"ReLU",
            },
            {
                "type":"MaxPool2d",
                "kernel_size":2,
                "stride": 1,
                "padding": 0,
                "dilation": 1,
            },
        ]

        # CNN Chuva
        camadas_cnn = []
        out_x = size_x
        out_y = size_y
        out_channels = 45
        for value in config_cnn:
            tipo = value.pop("type")
            camada = getattr(nn, tipo)

            if 'out_channels' in value:
                value["in_channels"] = out_channels

            try:
                out_x, out_y = self._calc_cnn_out(out_x, out_y, **value)
            except TypeError as e:
                pass

            camadas_cnn.append(camada(**value))

            if 'out_channels' in value:
                out_channels = value['out_channels']
        
        self.out_x = out_x
        self.out_y = out_y
        self.out_channels = out_channels
        self.cnn = nn.Sequential(*camadas_cnn).to(dtype=self.dtype)
        
        ##### LSTM para cada pixel da saída e para cada um dos dois valores, Chuva e Vazão.  #####
        self.lstm = nn.LSTM(
            input_size=self.out_x*self.out_y*self.out_channels,
            batch_first=True,
            hidden_size=64, # <- Entra no linear
        )

        # Camada que obtém a partir da convolução um valor estimado de vazão média
        # Entrada igual a saida do LSTM mais 13, 13 da data atual, ano e mês.
        self.fc = nn.Sequential(*[
            nn.Linear(in_features=64+13, out_features=32).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16).to(dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.n_future).to(dtype=self.dtype),
        ])
    
    def _calc_cnn_out(self, h_in, w_in, kernel_size, stride, padding, dilation, **kwargs):
        h = ((h_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        w = ((w_in + (2*padding) - (dilation * (kernel_size-1)) - 1)/stride) + 1
        return (int(h), int(w))


    def forward(self, input: tuple[torch.Tensor]):
        """
        chuva: Tensor (batch_size, n_past, height, width)
        vazao: Tensor (batch_size, n_past, height, width)

        ### Returns
        vazao: Tensor (batch_size, n_past)
        """

        x = input[0]
        dates = input[1]
        batch_size = input[0].shape[0]

        # Processar com CNN
        x = x.view(batch_size * self.n_past, x.shape[2], x.shape[3], x.shape[4])
        x:torch.Tensor = self.cnn(x)
        x = x.view(batch_size, self.n_past, -1)  # (batch_size, n_past, cnn_out_features)

        # Processar sequências de chuvas com LSTM
        x, (_, _) = self.lstm(x)  # h_n_chuva: (1, batch_size, 64)

        x = x[:, -1, :] #(batch_size, LSTM size)
        x = torch.cat([x, dates], dim=1) # (batch_size, LSTM size + 13)

        # Passar pelas camadas totalmente conectadas
        x = self.fc(x)  # (batch_size, n_future)

        return x
