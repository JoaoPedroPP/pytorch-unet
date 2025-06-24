# U-Net

## Executando o treinamento

Builda o container
- `podman build -t unet -f ContainerFile .`

Cria um volume e le as imagebs em `support_images/`
- `podman run -it -v ./support_images:/usr/app/support_images:z unet`

## Esttrutura de arquivos

As imagens de treinamento devem ser armazenadas em [support_images/dataset/raw](./support_images/dataset/raw/). Caso queira utilizar um diretório diferente é necessário atualizar [main.py#L229-L230](./main.py) com os caminho correto.

## Output

A imagem gerada por essa U-Net sempre será uma imagem binária, no entando a entrada pode variar. Essa rede esta preparada para receber uma imagem com uma(imagem em tom de cinza), ou duas(imagem em tom de cinza mais informação da borda). Para selecionar qual do tipos utilizar é necessario atualizar o arquivo [main.py#L223]

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
