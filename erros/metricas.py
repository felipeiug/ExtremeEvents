import os
import numpy as np
import torch
import torch.nn as nn
import asciichartpy
from scipy.optimize import curve_fit
from scipy import stats

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


# Métrica de erro personalizada para verificar tanto a vazão quanto a cota.
class CustomLoss3(nn.Module):
    def __init__(self, area_bacia:float, media:torch.Tensor, start_epoch:int=0, last_losses:list[float]=[], null_value:torch.Tensor = -99):
        """
        Inicializa a função de perda personalizada.
        """
        super(CustomLoss3, self).__init__()

        self.area_bacia = area_bacia
        self.media = media
        self.null_value = null_value
        self.erros = {
            "LOSS": None,
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

        y = torch.log10(torch.tensor(list_data, requires_grad=False)).tolist()

        chart = asciichartpy.plot(y, {
            'height': 10,
            'min':0,
            'format':'{:.4g}'
        })

        #       Cota Obs. Cota Sim. Vaz Obs., Vaz Sim
        data = [[], [], [], []]
        headers = []
        for i in range(self.last_obs[0].shape[1]+2):
            if i == 0:
                data[0].append("Cota Obs.")
                data[1].append("Cota Sim.")

                data[2].append("Vazão Obs.")
                data[3].append("Vazão Sim.")

                headers.append("Tipo")
                continue
            if i == self.last_obs[0].shape[1] + 1:
                data[0].append(self.erros["COTA REAL"])
                data[1].append(self.erros["COTA PRED"])

                data[2].append(self.erros["VAZAO REAL"])
                data[3].append(self.erros["VAZAO PRED"])

                headers.append("Mean")
                continue

            headers.append(f"Dia {i}")
            i = i-1
            data[0].append(f"{self.last_obs[0][0][i]:.4g}")
            data[1].append(f"{self.last_pred[0][0][i]:.4g}")

            data[2].append(f"{self.last_obs[3][i]:.4g}")
            data[3].append(f"{self.last_pred[1][0][i]:.4g}")

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


    def forward(self, y_pred:tuple[torch.Tensor], y_obs:tuple[torch.Tensor]):
        """
        Calcula a perda.
        Args:
            y_pred (Tensor, Tensor): Previsões do modelo e Previsões de Vazão.
            y_true (Tensor, List):   Rótulos verdadeiros e Vazões Verdadeiras.
        Returns:
            Tensor: Escalar representando a perda.
        """

        # Pesos
        mse_cota  = 2.0
        mse_vazao = 2.0

        min_0_cota  = 1.0
        min_0_vazao = 1.0

        major_cota = 1.0
        major_vazao = 1.0

        # Pesos para cada dia
        pesos_dia = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0] # 1 e 7 dia com mais importância

        self.last_obs  = y_obs
        self.last_pred = y_pred

        # Cota
        self.erros["COTA REAL"] = torch.nanmean(y_obs[0])
        self.erros["COTA PRED"] = torch.nanmean(y_pred[0])

        # Vazão
        vazao = torch.tensor(self.ajuste_vazao(y_obs[1], y_obs[2]))
        self.erros["VAZAO REAL"] = torch.nanmean(vazao)
        self.erros["VAZAO PRED"] = torch.nanmean(y_pred[1])

        # Adicionando as vazões aos dados observados
        self.last_obs = (
            self.last_obs[0],
            self.last_obs[1],
            self.last_obs[2],
            vazao
        )

        erros = [] # Erro para cada dia na medição
        for dia in range(y_obs[0].shape[1]):
            # Valores simulados
            cota_sim  = y_pred[0][:, dia]
            vazao_sim = y_pred[1][:, dia]

            # Valores reais
            cota_obs = y_obs[0][:, dia]
            vazao_obs = vazao[dia]

            # Valor do erro
            sum_erro = torch.tensor(0.0, requires_grad=True)

            # Valores de cota estimados menores que 0
            if cota_sim.min() < 0:
                mask = (cota_sim < 0)
                sum_erro = sum_erro + torch.pow(5, torch.sum(mask)) * min_0_cota

            # Valores de vazão estimados menores que 0
            if vazao_sim.min() < 0:
                mask = (vazao_sim < 0)
                sum_erro = sum_erro + torch.pow(5, torch.sum(mask)) * min_0_vazao

            # Erros Cota
            mse = MSE(cota_sim, cota_obs)
            nse = NSE(cota_sim, cota_obs, self.media)
            self.erros[f"NSE Cota T+{dia+1}"] = nse
            self.erros[f"RMSE Cota T+{dia+1}"] = torch.sqrt(mse)
            sum_erro = sum_erro + (mse * mse_cota)       # Aplicando pesos
            if mse > 0.000005 and mse < 1 and nse < 0.3: # Majorando o ERRO da Cota quando ele for menor que 1 e maior que 0
                sum_erro = sum_erro + (100 * major_cota)
                
            # Erros Vazão
            mse = MSE(vazao_sim, vazao_obs)
            nse = NSE(vazao_sim, vazao_obs, self.media)
            self.erros[f"NSE Vazão T+{dia+1}"] = nse
            self.erros[f"RMSE Vazão T+{dia+1}"] = torch.sqrt(mse)
            sum_erro = sum_erro + (mse * mse_vazao)      # Aplicando pesos
            if mse > 0.000005 and mse < 1 and nse < 0.3: # Majorando o ERRO da Vazão quando ele for menor que 1 e maior que 0
                sum_erro = sum_erro + (100 * major_vazao)

            erros.append(sum_erro/(mse_cota + mse_vazao + min_0_cota + min_0_vazao + major_cota + major_vazao))

        # Erro final
        erros = torch.tensor(erros, requires_grad=True)
        pesos_dia = torch.tensor(pesos_dia, requires_grad=True)
        final_error = torch.sum(erros*pesos_dia)/torch.sum(pesos_dia)

        self.losses_epoch.append(final_error)
        mean_loss = sum(self.losses_epoch)/max(1, len(self.losses_epoch))
        self.erros[f"LOSS"] = mean_loss

        return final_error
    
    def _linear(self, x, a, b):
        return a*x + b

    def _potencial(self, x, a, b):
        return a*np.pow(x, b)

    def _polinomio(self, x, a, b, c, d, e):
        return a + b*x + c*np.pow(x, 2) + d*np.pow(x, 3) + e*np.pow(x, 4)

    def ajuste_vazao(self, vazoes, areas):

        ajustes_pot = []
        for batch in range(len(areas)):

            ajustes_pot.append([])
            for day in range(len(areas[batch])):
                A = np.array(areas[batch][day])
                vaz = np.array(vazoes[batch][day])

                vaz_especifica = vaz/A

                Q1 = np.quantile(vaz_especifica, 0.25)
                Q3 = np.quantile(vaz_especifica, 0.75)
                IQR = Q3 - Q1

                inf = Q1 - 1.5*IQR
                sup = Q3 + 1.5*IQR
                mask_outliers = (vaz_especifica <= sup) & (vaz_especifica >= inf)

                params, _ = curve_fit(self._potencial, A[mask_outliers], vaz_especifica[mask_outliers], maxfev = 2000)
                ajustes_pot[batch].append(params)

        ajustes_pot = np.array(ajustes_pot)
        ajustes_pot = np.mean(ajustes_pot, axis=0)
        
        vazao_esp_final = self._potencial(self.area_bacia, ajustes_pot[:, 0], ajustes_pot[:, 1])
        vazao_final = vazao_esp_final * self.area_bacia
        return vazao_final


# MSE Error
def MSE(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true) ** 2)

# NASH ERROR:
def NSE(y_pred, y_true, mean=None):
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
    if mean is None:
        mean = torch.mean(y_true)
    denominator = torch.sum((y_true - mean) ** 2)
    
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
def R2(y_pred, y_true, mean=None):
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
    if mean is None:
        mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - mean) ** 2)
    
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






