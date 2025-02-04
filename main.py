# Apenas para gerar os resultados
# from run_models import *
# exit()

import os
import traceback

try:
    import torch.version
    from neuronios.NN import *
    from datareader.data_reader2 import ReadRasters
    from erros.metricas import *

    import torch
    import torch.optim as optim


    os.system("cls")

    load = True # Se irei utilizar o modelo salvo
    device = "cpu"
    torch.set_default_dtype(torch.float32)

    if torch.cuda.is_available():
        device = "cuda"
        torch.set_default_device("cuda")
    else:
        print("CUDA não está disponível.")

    modelo = 6     # Modelo utilizado no teste
    min_date = "2000-01-01" # Data mínima para treino
    batch_size = 1 # Processamento em paralelo
    n_data = 14    # Quantidade de bandas nos dados salvos
    n_dias = 7     # Número de dias no futuro da previsão
    dias_previsao = [1, 7] # Quais tempos devo prever Substitui n_dias!!!
    n_temp = 10    # Número de tempos no passado LSTM (dias)
    div_out = 4    # Os tamanhos X e Y serão dividos por div_out na convolução
    size_x = 694   # Dimensão X do raster
    size_y = 1198  # Dimensão Y do raster
    bandas_time=[  # Bandas com dados temporais
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ]

    # Obtendo máximo de dias
    n_dias = n_dias if dias_previsao is None else max(dias_previsao)


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
        randomize=True,
        min_date=min_date,
        normalize=False,
        threaded=False, # Não é recomendade usar as threads quando o buffer está em mais de 10%
        ram_use_percentage = 0.8,
    )

    print(f"Criando RNA's com o modelo {modelo}")
    nns:dict[int, nn.Module] = {}
    if modelo == 6:
        for dia in dias_previsao:
            nns[dia] = Rede6(
                n_future = 1,
                n_past   = n_temp,
                size_x   = size_x,
                size_y   = size_y,
            )

    total_params = sum(p.numel() for p in nns[list(nns.keys())[0]].parameters())
    print(f"Modelo {modelo} com {total_params} parâmetros")

    print("Criando Otimizadores")
    opts:dict[int, optim.Optimizer] = {}
    for dia in dias_previsao:
        opts[dia] = optim.AdamW(nns[dia].parameters(), lr=0.00002, weight_decay=0.001) # lr=0.001, weight_decay=0.05

    loss = 0
    start_epoch = 0
    last_losses = []

    os.makedirs(f"modelos_{modelo}", exist_ok=True)
    if load:
        print("Importando modelos treinados")
        modelos = os.listdir(f"D:/Mestrado/2º Semestre/Extreme Events/RNA/modelos_{modelo}")

        if len(modelos) > 0:
            for _modelo in modelos:
                file = f"D:/Mestrado/2º Semestre/Extreme Events/RNA/modelos_{modelo}/{_modelo}"
                checkpoint = torch.load(file, weights_only=True)
                last_losses.append(checkpoint['LOSS'])

            epochs = [int(i.split("_")[-1].split(".")[0]) for i in modelos]
            max_epoch = max(epochs)

            file = f"D:/Mestrado/2º Semestre/Extreme Events/RNA/modelos_{modelo}/modelo_{max_epoch}.pth"
            checkpoint = torch.load(file, weights_only=True)

            for dia in dias_previsao:
                if f'model_state_dict_{dia}' in checkpoint:
                    nns[dia].load_state_dict(checkpoint[f'model_state_dict_{dia}'])

                if f'optimizer_state_dict_{dia}' in checkpoint:
                    lr_ans = opts[dia].defaults["lr"]
                    w_ans = opts[dia].defaults["weight_decay"]

                    opts[dia].load_state_dict(checkpoint[f'optimizer_state_dict_{dia}'])

                    opts[dia].defaults["lr"] = lr_ans
                    opts[dia].defaults["weight_decay"] = w_ans
                    for param_group in opts[dia].param_groups:
                        param_group["lr"] = lr_ans
                        param_group["weight_decay"] = w_ans

            start_epoch = checkpoint['epoch']+1
            loss = checkpoint['LOSS']

    # Métrica de Erro
    print("Criando Métricas de Erro")
    criterion = CustomLoss4(
        media=reader.mean_cotas,
        last_losses=last_losses,
        start_epoch=start_epoch,
        dias_previsao=dias_previsao,
    ).to(torch.float32)

    # Treinamento
    epoch = start_epoch
    print("Iniciando Treino")

    while True:
        reader.reset()

        for dia in dias_previsao:
            nns[dia].train()

        criterion.epoch = epoch

        # Resetando os losses por epoca
        criterion.losses_epoch = {dia:[] for dia in dias_previsao}

        progress_bar = range(reader.total_train())
        criterion.start()
        
        for step in progress_bar:
            try:
                X, y = reader.next()

                if X is None and y is None:
                    continue
                elif (y == -99).any().item():
                    continue

                for dia in dias_previsao:
                    # Forward pass
                    outputs = nns[dia]((X[0].clone().to("cuda"), X[1].clone().to("cuda")))

                    # Métrica de erro #### Não esta ideal, mas funciona.
                    criterion(outputs, y[:, dia-1].to("cuda"), dia)
                    loss = torch.mean(torch.abs(outputs - y[:, dia-1].to("cuda")) ** 2)

                    # Backward pass e otimização
                    opts[dia].zero_grad()
                    loss.backward()
                    opts[dia].step()

                    del loss, outputs         # Limpa variáveis da CPU/RAM
                    torch.cuda.empty_cache()  # Limpa a cache da GPU

                # Legenda
                description = ""
                for dia in dias_previsao:
                    description += f"Dia {dia}: " + " | ".join([f" {name.upper()}: {f'{erro[dia]:.4g}' if erro[dia] is not None else None}" for name, erro in criterion.erros.items()])
                    description += "\n"

                description += f"Epoch: {epoch+1}"

                date = reader.date_range[reader.train_indexes[(reader.step - reader.batch_size)]]
                criterion.print(step, reader.total_train(), description, date)

                print("RASTERES:", len(reader.raster_buffer.rasters))
                del X, y
            except Exception as e:
                traceback.print_exc()
                print(e)
                input("Pressione enter para continuar")

        criterion.stop()
        criterion.last_losses.append(criterion.erros["LOSS"])

        dados_save = {
            'epoch': epoch,
        }

        for dia in dias_previsao:
            dados_save[f'model_state_dict_{dia}']     = nns[dia].state_dict()
            dados_save[f'optimizer_state_dict_{dia}'] = opts[dia].state_dict()

        for name, erro in criterion.erros.items():
            dados_save[name] = erro

        torch.save(dados_save, f"D:/Mestrado/2º Semestre/Extreme Events/RNA/modelos_{modelo}/modelo_{epoch}.pth")

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

except Exception as e:
    traceback.print_exc()
    print(e)