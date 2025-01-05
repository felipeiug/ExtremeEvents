import torch
import torch.nn as nn

class LSTM2D(nn.Module):
    def __init__(self, input_size: tuple[int, int], timesteps: int, hidden_size: int):
        """
        input_size: Dimensões (altura, largura) do tensor 2D.
        timesteps: Número de passos de tempo.
        hidden_size: Tamanho do estado oculto e da célula.
        """
        super().__init__()

        self.input_size = input_size
        self.timesteps = timesteps
        self.hidden_size = hidden_size

        # Camada LSTM única para toda a matriz 2D
        # Esta LSTM processa as entradas concatenadas das posições 2D em paralelo
        self.lstm = nn.LSTM(
            input_size=1,  # Cada entrada individual é escalar
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, input:torch.Tensor):
        """
        input: Tensor com formato (timesteps, altura, largura).
        Retorna um tensor processado por LSTMs individuais em cada posição 2D.
        """

        # Verificar formato da entrada
        batch_size, timesteps, height, width = input.shape
        assert (height, width) == self.input_size, "Dimensões espaciais não correspondem ao input_size"
        assert timesteps == self.timesteps, "A dimensão do tempo não corresponde ao timesteps"

        # Reorganizar o tensor para (batch_size * altura * largura, timesteps, 1)
        reshaped_input = input.permute(0, 2, 3, 1).reshape(-1, timesteps, 1)

        # Passar pela camada LSTM
        output, _ = self.lstm(reshaped_input)

        # Reorganizar a saída para (batch_size, altura, largura, timesteps, hidden_size)
        output = output.view(batch_size, height, width, timesteps, self.hidden_size).permute(0, 3, 1, 2, 4)

        return output
