import os
import torch
import torch.nn as nn
import asciichartpy

from tabulate import tabulate

# Utilizad até a Epoch 30, após isto estabilizou
class CustomLoss(nn.Module):
    def __init__(self, media:torch.Tensor, start_epoch:int=0, last_losses:list[float]=[], null_value:torch.Tensor = -99):
        """
        Inicializa a função de perda personalizada.
        Args:
            power (float): Exponente aplicado à diferença entre as previsões e os rótulos.
        """
        super(CustomLoss, self).__init__()

        self.media = media
        self.null_value = null_value
        self.erros = {
            "LOSS": None,
            "REAL": None,
            "PRED": None,
        }
        self.last_losses = last_losses
        self.epoch = start_epoch
        self.last_pred = torch.tensor([[1]])
        self.last_obs  = torch.tensor([[1]])

    def print(self, progress):

        # x = range(0, self.epoch)
        y = torch.log10(torch.tensor(self.last_losses)).tolist()

        chart = asciichartpy.plot(y, {
            'height': 10,
            'min':0,
            'format':'{:.4g}'
        })

        data = [[], []]
        headers = []
        for i in range(self.last_obs.shape[1]+2):
            if i == 0:
                data[0].append("Obs.")
                data[1].append("Sim.")
                headers.append("Tipo")
                continue
            if i == self.last_obs.shape[1] + 1:
                data[0].append(self.erros["REAL"])
                data[1].append(self.erros["PRED"])
                headers.append("Mean")
                continue

            headers.append(f"Dia {i}")
            i = i-1
            data[0].append(f"{self.last_obs[0][i]:.4g}")
            data[1].append(f"{self.last_pred[0][i]:.4g}")

        table = tabulate(
            data,
            headers=headers,
            tablefmt="rounded_grid",
            numalign="center",
            stralign="center"
        )

        os.system("cls")
        print(chart)
        print()
        print(progress)
        print()
        print(table)


    def forward(self, y_pred:torch.Tensor, y_obs:torch.Tensor):
        """
        Calcula a perda.
        Args:
            y_pred (Tensor): Previsões do modelo.
            y_true (Tensor): Rótulos verdadeiros.
        Returns:
            Tensor: Escalar representando a perda.
        """

        self.last_obs = y_obs.clone()
        self.last_pred = y_pred.clone()

        self.erros["REAL"] = torch.nanmean(y_obs)
        self.erros["PRED"] = torch.nanmean(y_pred)

        # Valores estimados menores que 0 e que não deveriam ser (!= -99)
        mask_null = (y_obs == self.null_value)
        sum_erro = 0
        if y_pred.min() < 0:
            mask = ((~mask_null) & (y_pred < 0)) | (y_obs > 1)
            sum_erro += torch.tensor(5) ** torch.sum(mask)

        # Cálculo do NSE
        numerator = torch.sum(((y_pred - y_obs) ** 2), dim=0)
        denominator = torch.sum((y_obs - self.media) ** 2, dim=0)
        f1 = 1 - (numerator / denominator)
        nse = torch.nanmean(f1)
        self.erros["NSE"] = nse

        # Cálculo do RMSE
        # rmse = torch.nanmean(torch.sqrt(torch.mean((y_pred - y_obs) ** 2, dim=0)))
        # self.erros["RMSE"] = rmse

        # MSE
        mse = torch.mean(torch.abs(y_pred - y_obs) ** 2)
        self.erros["MSE"] = mse

        final_error = mse + ((nse-1)**2) + sum_erro
        self.erros["LOSS"] = final_error

        return final_error


# Utilizad a partir da Epoch 31
class CustomLoss2(nn.Module):
    def __init__(self, media:torch.Tensor, start_epoch:int=0, last_losses:list[float]=[], null_value:torch.Tensor = -99):
        """
        Inicializa a função de perda personalizada.
        Args:
            power (float): Exponente aplicado à diferença entre as previsões e os rótulos.
        """
        super(CustomLoss2, self).__init__()

        self.media = media
        self.null_value = null_value
        self.erros = {
            "LOSS": None,
            "REAL": None,
            "PRED": None,
        }
        self.last_losses = last_losses

        self.epoch = start_epoch
        self.losses_epoch = []
        
        self.last_pred = torch.tensor([[1]])
        self.last_obs  = torch.tensor([[1]])

    def print(self, progress):

        # x = range(0, self.epoch)
        list_data = self.last_losses.copy()
        if self.erros["LOSS"]:
            list_data.append(self.erros["LOSS"])

        y = torch.log10(torch.tensor(list_data)).tolist()

        chart = asciichartpy.plot(y, {
            'height': 10,
            'min':0,
            'format':'{:.4g}'
        })

        data = [[], []]
        headers = []
        for i in range(self.last_obs.shape[1]+2):
            if i == 0:
                data[0].append("Obs.")
                data[1].append("Sim.")
                headers.append("Tipo")
                continue
            if i == self.last_obs.shape[1] + 1:
                data[0].append(self.erros["REAL"])
                data[1].append(self.erros["PRED"])
                headers.append("Mean")
                continue

            headers.append(f"Dia {i}")
            i = i-1
            data[0].append(f"{self.last_obs[0][i]:.4g}")
            data[1].append(f"{self.last_pred[0][i]:.4g}")

        table = tabulate(
            data,
            headers=headers,
            tablefmt="rounded_grid",
            numalign="center",
            stralign="center"
        )

        os.system("cls")
        print(chart)
        print()
        print(progress)
        print()
        print(table)


    def forward(self, y_pred:torch.Tensor, y_obs:torch.Tensor):
        """
        Calcula a perda.
        Args:
            y_pred (Tensor): Previsões do modelo.
            y_true (Tensor): Rótulos verdadeiros.
        Returns:
            Tensor: Escalar representando a perda.
        """

        self.last_obs = y_obs.clone()
        self.last_pred = y_pred.clone()

        self.erros["REAL"] = torch.nanmean(y_obs)
        self.erros["PRED"] = torch.nanmean(y_pred)

        # Valores estimados menores que 0 e que não deveriam ser (!= -99)
        mask_null = (y_obs == self.null_value)
        sum_erro = 0
        if y_pred.min() < 0:
            mask = ((~mask_null) & (y_pred < 0)) | (y_obs > 1)
            sum_erro += torch.tensor(5) ** torch.sum(mask)

        # Cálculo do NSE
        numerator = torch.sum(((y_pred - y_obs) ** 2), dim=0)
        denominator = torch.sum((y_obs - self.media) ** 2, dim=0)
        f1 = 1 - (numerator / denominator)
        nse = torch.nanmean(f1)
        self.erros["NSE"] = nse

        # MSE
        mse = torch.mean(torch.abs(y_pred - y_obs) ** 2)
        self.erros["MSE"] = mse

        # Majorando o ERRO quando ele for menor que 1 e maior que 0
        if mse > 0.000005 and mse < 1 and nse < 0.3:
            sum_erro += 100

        final_error = mse + ((nse-1)**2) + sum_erro

        self.losses_epoch.append(final_error)
        mean_loss = sum(self.losses_epoch)/max(1, len(self.losses_epoch))
        self.erros["LOSS"] = mean_loss

        return final_error


# NASH ERROR:
def NSE(y_pred, y_true):
    """
    Calcula o índice de eficiência de Nash-Sutcliffe (NSE).
    
    Args:
        y_true (torch.Tensor): Valores observados (reais).
        y_pred (torch.Tensor): Valores previstos (predições).
    
    Returns:
        torch.Tensor: Métrica NSE.
    """
    # Calcula o termo do numerador (erros ao quadrado entre predições e valores reais)
    numerator = torch.sum((y_true - y_pred) ** 2)
    
    # Calcula o termo do denominador (variância dos valores reais em relação à média)
    denominator = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    # Evitar divisão por zero no denominador
    if denominator == 0:
        return -99
    
    # Calcula o NSE
    nse = 1 - (numerator / denominator)
    return nse

#MAE ERROR:
def MAE(y_pred, y_true):
    """
    Compute Mean Absolute Error (MAE).
    y_true: torch.Tensor of actual/observed values
    y_pred: torch.Tensor of predicted values
    """
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae.item()  # Convert tensor to scalar value

# R2
def R2(y_pred, y_true):
    """
    Calcula o coeficiente de determinação (R²).
    
    Args:
        y_true (torch.Tensor): Valores reais.
        y_pred (torch.Tensor): Valores previstos.
    
    Returns:
        torch.Tensor: R².
    """
    # Calcula o erro residual (numerador)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    
    # Calcula o total de variação (denominador)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    
    # Evitar divisão por zero no denominador
    if ss_tot == 0:
        return -99
    
    # Calcula o R²
    r2 = 1 - (ss_res / ss_tot)
    return r2

# PBIAS
def PBIAS(y_pred, y_true):
    """
    Calcula o Percent Bias (PBIAS).
    
    Args:
        y_true (torch.Tensor): Valores reais.
        y_pred (torch.Tensor): Valores previstos.
    
    Returns:
        float: Valor do PBIAS em porcentagem.
    """
    # Numerador: Soma das diferenças entre valores reais e previstos
    numerator = torch.sum(y_true - y_pred)
    
    # Denominador: Soma dos valores reais
    denominator = torch.sum(y_true)
    
    # Evitar divisão por zero
    if denominator == 0:
        return -99
    
    # Calcular PBIAS
    pbias_value = 100 * numerator / denominator
    return pbias_value.item()






