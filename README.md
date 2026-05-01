# U-Net

## Executando o treinamento

Builda o container
- `podman build -t unet -f ContainerFile .`

Cria um volume e le as imagebs em `support_images/`
- `podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run <CLI comamnds>`

## Estrutura de arquivos

As imagens de treinamento devem ser armazenadas em [support_images/dataset/raw](./support_images/dataset/raw/). Caso queira utilizar um diretório diferente é necessário atualizar [main.py#L229-L230](./main.py) com os caminho correto.

## Output

A imagem gerada por essa U-Net sempre será uma imagem binária, no entando a entrada pode variar. Essa rede esta preparada para receber uma imagem com uma(imagem em tom de cinza), ou duas(imagem em tom de cinza mais informação da borda). Para selecionar qual do tipos utilizar é necessario atualizar o arquivo [main.py#L223](./main.py#L223)

## Comandos da CLI

A CLI é executada através do script `unet-run.py`. O comando principal é:

```bash
python unet-run.py <ação> [opções]
```

### Ações disponíveis

| Ação | Descrição |
|------|-----------|
| `run` | Executa o treinamento da U-Net |
| `config-dataset` | Configura o dataset (em desenvolvimento) |

### Opções

| Opção | Tipo | Padrão | Descrição |
|-------|------|--------|-----------|
| `--seed` | int | 42 | Semente aleatória para reprodutibilidade |
| `--dims` | int | 1 | Número de dimensões de entrada (canais). Use 1 para imagem em tom de cinza ou 2 para imagem com informação de borda |
| `--classes` | int | 1 | Número de classes de saída |
| `--epochs` | int | 200 | Número de épocas de treinamento |
| `--folds` | int | 10 | Número de folds para validação cruzada (K-Fold) |
| `--dataset-path` | str | `./support_images/dataset/raw` | Caminho para o diretório do dataset |
| `--mask-path` | str | `./support_images/dataset/raw` | Caminho para o diretório das máscaras |
| `--simple` | flag | False | Utiliza a versão simplificada da U-Net |

### Exemplos de uso

Treinamento básico com parâmetros padrão:

```bash
python unet-run.py run
```

Treinamento com 200 épocas e 10 folds:

```bash
python unet-run.py run --epochs 100 --folds 5
```

Treinamento com imagem de 2 canais (tom de cinza + borda):

```bash
python unet-run.py run --dims 2
```

Treinamento especificando caminhos personalizados:

```bash
python unet-run.py run --dataset-path /caminho/para/dataset --mask-path /caminho/para/mascaras
```

Utilizando com container:

```bash
podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --epochs 100 --folds 5
```

## License


MIT License

Copyright (c) 2025 Joao Pedro Poloni Ponce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
