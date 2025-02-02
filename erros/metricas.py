import os
import time
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import asciichartpy

from datetime import timedelta
from scipy.optimize import curve_fit

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

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.start_time = None

    def print(self, step, total, description, date:pd.Timestamp):
        if self.start_time is None:
            print("É necessário iniciar o processo!")
            return
        
        # Obter a largura do terminal
        terminal_width = shutil.get_terminal_size().columns

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

        # Alterando a primeira coluna do chart:
        lines = chart.split("\n")
        new_chart = ""
        max_size = 0
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            if len(start) > max_size:
                max_size = len(start)
        
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            diff = (max_size - len(start))
            new_chart += " " * diff
            new_chart += line[0:len(line)]
            new_chart += "\n"
        chart1 = new_chart

        # Gráfico 2
        losses = self.losses_epoch
        N = max(20, int(terminal_width * 0.7))
        if len(losses) > N:
            losses = torch.tensor(losses, requires_grad=False)
            chunks = torch.chunk(losses, N+1)
            losses = torch.tensor([torch.nanmean(chunk) for chunk in chunks])
        
        chart = asciichartpy.plot(torch.tensor(losses, requires_grad=False).tolist(), {
            'height': 10,
            'min':0,
            'format':'{:.4g}'
        })

        # Alterando a primeira coluna do chart:
        lines = chart.split("\n")
        new_chart = ""
        max_size = 0
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            if len(start) > max_size:
                max_size = len(start)
        
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            diff = (max_size - len(start))
            new_chart += " " * diff
            new_chart += line[0:len(line)]
            new_chart += "\n"
        chart2 = new_chart
        chart2 += "Losses: " + str(len(losses))

        ### Barra de progresso ###

        progress = step / total
        percentage = int(progress * 100)

        description += f" {percentage}% ["

        # Tempo de processamento
        elapsed_time = time.time() - self.start_time
        time_per_item = elapsed_time / step if step > 0 else 0
        total_time = time_per_item * total

        text_after = f"] {step}/{total} [{self._time_to_str(elapsed_time)}<{self._time_to_str(total_time)}, {time_per_item:.2f}s/item]"

        # Exibir a barra

        bar_width = max(terminal_width - len(text_after) - len(description), 10)  # Garantir no mínimo 10 caracteres para a barra
        filled_length = int(bar_width * progress)

        # Criar a barra de progresso
        bar = "█" * filled_length + "-" * (bar_width - filled_length)

        progress_bar = description + bar + text_after

        # Cota Obs. Cota Sim.
        data = [[], []]
        headers = []
        for i in range(self.last_obs.shape[1]+2):
            if i == 0:
                data[0].append("Cota Obs.")
                data[1].append("Cota Sim.")

                headers.append("Tipo")
                continue
            if i == self.last_obs.shape[1] + 1:
                data[0].append(self.erros["REAL"])
                data[1].append(self.erros["PRED"])

                headers.append("Mean")
                continue
            
            date_str = (date + pd.Timedelta(days=i)).strftime("%d/%m/%y")
            headers.append(date_str)

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
        print("MEAN LOSS")
        print(chart1)
        print()
        print("EPOCH LOSS")
        print(chart2)
        print()
        print(progress_bar)
        print()
        print(table)

    def _time_to_str(self, time):
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def forward(self, y_pred:torch.Tensor, y_obs:torch.Tensor):
        """
        Calcula a perda.
        Args:
            y_pred (Tensor): Previsões do modelo.
            y_true (Tensor): Rótulos verdadeiros.
        Returns:
            Tensor: Escalar representando a perda.
        """

        self.last_obs = y_obs.clone().to("cpu")
        self.last_pred = y_pred.clone().to("cpu")

        self.erros["REAL"] = float(torch.nanmean(y_obs).to("cpu"))
        self.erros["PRED"] = float(torch.nanmean(y_pred).to("cpu"))

        # Valores estimados menores que 0 e que não deveriam ser (!= -99)

        # Cálculo do NSE
        numerator = torch.sum(((y_pred - y_obs) ** 2), dim=0)
        denominator = torch.sum((y_obs - self.media) ** 2, dim=0) + 1E-8 # somando uma constante pequena para nunca ser 0
        f1 = 1 - (numerator / denominator)
        nse = torch.nanmean(f1)
        self.erros["NSE"] = float(nse)

        # MSE
        mse = torch.mean(torch.abs(y_pred - y_obs) ** 2)

        rmse = torch.sqrt(mse)
        self.erros["RMSE"] = float(rmse)

        self.losses_epoch.append(mse)
        mean_loss = sum(self.losses_epoch)/max(1, len(self.losses_epoch))
        self.erros["LOSS"] = float(mean_loss)

        return mse.to("cuda")


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

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.start_time = None

    def print(self, step, total, description, date:pd.Timestamp):
        if self.start_time is None:
            print("É necessário iniciar o processo!")
            return

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

        # Alterando a primeira coluna do chart:
        lines = chart.split("\n")
        new_chart = ""
        max_size = 0
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            if len(start) > max_size:
                max_size = len(start)
        
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            diff = (max_size - len(start))
            new_chart += " " * diff
            new_chart += line[0:len(line)]
            new_chart += "\n"
        chart = new_chart

        ### Barra de progresso ###

        # Obter a largura do terminal
        terminal_width = shutil.get_terminal_size().columns
        progress = step / total
        percentage = int(progress * 100)

        description += f" {percentage}% ["

        # Tempo de processamento
        elapsed_time = time.time() - self.start_time
        time_per_item = elapsed_time / step if step > 0 else 0
        total_time = time_per_item * total

        text_after = f"] {step}/{total} [{self._time_to_str(elapsed_time)}<{self._time_to_str(total_time)}, {time_per_item:.2f}s/item]"

        # Exibir a barra

        bar_width = max(terminal_width - len(text_after) - len(description), 10)  # Garantir no mínimo 10 caracteres para a barra
        filled_length = int(bar_width * progress)

        # Criar a barra de progresso
        bar = "█" * filled_length + "-" * (bar_width - filled_length)

        progress_bar = description + bar + text_after

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
            
            date_str = (date + pd.Timedelta(days=i)).strftime("%d/%m/%y")
            headers.append(date_str)

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
        print(progress_bar)
        print()
        print(table)

    def _time_to_str(self, time):
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def forward(self, y_pred:tuple[torch.Tensor], y_obs:tuple[torch.Tensor])->torch.Tensor:
        """
        Calcula a perda.
        Args:
            y_pred (Tensor, Tensor): Previsões do modelo e Previsões de Vazão.
            y_true (Tensor, List):   Rótulos verdadeiros e Vazões Verdadeiras.
        Returns:
            Tensor: Escalar representando a perda.
        """

        # Erro final
        sum_erro = torch.tensor(0.0, device=y_pred[0].device)

        # Pesos
        pesos_cota = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], device=y_pred[0].device)
        pesos_vazao = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=y_pred[0].device)

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

        ##### Cálculo do erro

        # Menores que 0
        mask = (y_pred[0] < 0)
        count = torch.sum(mask)

        mask = (y_pred[1] < 0)
        count = count + torch.sum(mask)

        sum_erro = sum_erro + torch.pow(5, count)

        # RMSE Cota
        rmse_cota = torch.sqrt(torch.mean(torch.abs(y_pred[0] - y_obs[0]) ** 2, dim=0))

        # RMSE Vazao
        rmse_vazao = torch.sqrt(torch.mean(torch.abs(y_pred[1] - vazao) ** 2, dim=0))

        # NSE Cota
        numerator = torch.sum((y_obs[0] - y_pred[0]) ** 2, dim=0)
        denominator = torch.sum((y_obs[0] - self.media) ** 2, dim=0) + 1e-8 # Soma da constante para que o denominador não seja 0.
        nse_cota = 1 - numerator / denominator

        # Aplicando os pesos do erro e realizando uma média ponderada
        rmse_cota = (rmse_cota * pesos_cota)
        rmse_cota = torch.sum(rmse_cota)/torch.sum(pesos_cota)
        
        rmse_vazao = (rmse_vazao * pesos_vazao)
        rmse_vazao = torch.sum(rmse_vazao)/torch.sum(pesos_vazao)

        # Somando os valores de mse
        sum_erro = sum_erro + rmse_cota + (rmse_vazao/25)

        if len(self.losses_epoch) > 1000:
            self.losses_epoch.pop(0)
        self.losses_epoch.append(sum_erro)
        
        mean_loss = sum(self.losses_epoch)/max(1, len(self.losses_epoch))
        self.erros[f"LOSS"] = mean_loss

        self.erros["NSE COTA"] = torch.mean(nse_cota)
        self.erros["RMSE COTA"] = rmse_cota
        self.erros["RMSE VAZAO"] = rmse_vazao

        return sum_erro
    
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


# Utilizad a partir da Epoch 31
class CustomLoss4(nn.Module):
    def __init__(self, media:torch.Tensor, start_epoch:int=0, last_losses:list[dict[int,float]]=[], null_value:torch.Tensor = -99, dias_previsao:list[int] = []):
        """
        Inicializa a função de perda personalizada.
        Args:
            power (float): Exponente aplicado à diferença entre as previsões e os rótulos.
        """
        super(CustomLoss4, self).__init__()

        self.dias_previsao = dias_previsao
        self.media = media
        self.null_value = null_value
        self.erros = {
            "LOSS": None,
            "REAL": None,
            "PRED": None,
            "NSE": None,
            "RMSE": None,
        }
        self.last_losses = last_losses

        self.epoch = start_epoch
        self.losses_epoch:dict[int, list[float]] = {dia:[] for dia in self.dias_previsao}
        
        self.last_pred:dict[int, torch.Tensor] = {}
        self.last_obs:dict[int, torch.Tensor]  = {}

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.start_time = None

    def print(self, step, total, description:str, date:pd.Timestamp):
        if self.start_time is None:
            print("É necessário iniciar o processo!")
            return
        
        #%% =============================== Gráfico ===============================

        # Obter a largura do terminal
        terminal_width = shutil.get_terminal_size().columns

        # x = range(0, self.epoch)
        list_data = self.last_losses.copy()
        if self.erros["LOSS"]:
            list_data.append(self.erros["LOSS"])

        charts = ""
        for dia in self.dias_previsao:
            charts += f"\nDia {dia}\n"

            y = torch.log10(torch.tensor([i[dia] for i in list_data], requires_grad=False)).tolist()

            chart = asciichartpy.plot(y, {
                'height': 5,
                'min':0,
                'format':'{:.4g}'
            })

            charts += chart

        # Gráfico 2
        for dia in self.dias_previsao:
            charts += f"\nLOSS dia {dia}\n"
            losses = [i for i in self.losses_epoch[dia]]

            N = max(20, int(terminal_width * 0.8))
            if len(losses) > N:
                losses = torch.tensor(losses, requires_grad=False)
                chunks = torch.chunk(losses, N+1)
                losses = torch.tensor([torch.nanmean(chunk) for chunk in chunks])
        
            chart = asciichartpy.plot(torch.tensor(losses, requires_grad=False).tolist(), {
                'height': 5,
                'min':0,
                'format':'{:.4g}'
            })

            charts += chart

        # Alterando a primeira coluna do chart:
        lines = charts.split("\n")
        new_chart = ""
        max_size = 0
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            if len(start) > max_size:
                max_size = len(start)
        
        for line in lines:
            start = line.split("┤")[0].split("┼")[0]
            diff = (max_size - len(start))
            new_chart += " " * diff
            new_chart += line[0:len(line)]
            new_chart += "\n"
        charts = new_chart
        charts += "Losses: " + str(len(losses))

        #%% =============================== Barra de progresso ===============================

        progress = step / total
        percentage = int(progress * 100)

        description += f" {percentage}% ["

        # Tempo de processamento
        elapsed_time = time.time() - self.start_time
        time_per_item = elapsed_time / step if step > 0 else 0
        total_time = time_per_item * total

        text_after = f"] {step}/{total} [{self._time_to_str(elapsed_time)}<{self._time_to_str(total_time)}, {time_per_item:.2f}s/item]"

        # Exibir a barra

        len_desc = len(description.split("\n")[-1])
        bar_width = max(terminal_width - len(text_after) - len_desc, 10)  # Garantir no mínimo 10 caracteres para a barra
        filled_length = int(bar_width * progress)

        # Criar a barra de progresso
        bar = "█" * filled_length + "-" * (bar_width - filled_length)

        progress_bar = description + bar + text_after

        #%% =============================== Tabela dos Dados ===============================

        # Cota Obs. Cota Sim.
        data = [[], []]
        headers = []

        headers.append("Tipo")
        data[0].append("Cota Obs.")
        data[1].append("Cota Sim.")

        for dia in self.dias_previsao:
            date_str = (date + pd.Timedelta(days=dia)).strftime("%d/%m/%y")
            headers.append(date_str)

            data[0].append(f"{torch.nanmean(self.last_obs[dia]):.4g}")
            data[1].append(f"{torch.nanmean(self.last_pred[dia]):.4g}")

        table = tabulate(
            data,
            headers=headers,
            tablefmt="rounded_grid",
            numalign="center",
            stralign="center"
        )

        # %% =============================== Print ===============================
        os.system("cls")
        print(charts)
        print()
        print(progress_bar)
        print()
        print(table)

    def _time_to_str(self, time):
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def forward(self, y_pred:torch.Tensor, y_obs:torch.Tensor, dia: int):
        """
        Calcula a perda.
        Args:
            y_pred (Tensor): Previsões do modelo.
            y_true (Tensor): Rótulos verdadeiros.
        Returns:
            Tensor: Escalar representando a perda.
        """

        self.last_obs[dia] = y_obs.clone().to("cpu")
        self.last_pred[dia] = y_pred.clone().to("cpu")
        
        if self.erros["REAL"] is None or self.erros["PRED"] is None:
            self.erros["REAL"] = {}
            self.erros["PRED"] = {}

        self.erros["REAL"][dia] = float(torch.nanmean(y_obs).to("cpu"))
        self.erros["PRED"][dia] = float(torch.nanmean(y_pred).to("cpu"))

        # Cálculo do NSE
        numerator = torch.sum(((y_pred - y_obs) ** 2), dim=0)
        denominator = torch.sum((y_obs - self.media) ** 2, dim=0) + 1E-8 # somando uma constante pequena para nunca ser 0
        f1 = 1 - (numerator / denominator)
        nse = torch.nanmean(f1)

        if self.erros["NSE"] is None:
            self.erros["NSE"] = {}
        self.erros["NSE"][dia] = float(nse)

        # MSE
        mse = torch.mean(torch.abs(y_pred - y_obs) ** 2)

        rmse = torch.sqrt(mse)
        if self.erros["RMSE"] is None:
            self.erros["RMSE"] = {}
        self.erros["RMSE"][dia] = float(rmse)

        self.losses_epoch[dia].append(mse)
        mean_loss = sum(self.losses_epoch[dia])/max(1, len(self.losses_epoch[dia]))
        if self.erros["LOSS"] is None:
            self.erros["LOSS"] = {}
        self.erros["LOSS"][dia] = float(mean_loss)

        return mse.to("cuda")


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






