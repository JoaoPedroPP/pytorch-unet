# U-Net

Builda o container
- `podman build -t unet -f ContainerFile .`

Cria um volume e le as imagebs em `support_images/`
- `podman run -it -v ./support_images:/usr/app/support_images:z unet`
