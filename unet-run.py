#!/usr/bin/env python3

import argparse

import main as unet

def main():
    parser = argparse.ArgumentParser(description="Roda U-Net de acordo com os parâmetros")
    parser.add_argument("acao", nargs=1, choices=["run", "config-dataset"], help="Ação a ser executada pela CLI")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    parser.add_argument("--dims", type=int, default=1, help="Número de dimensões de entrada (canais)")
    parser.add_argument("--classes", type=int, default=1, help="Número de classes de saída")
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas de treinamento")
    parser.add_argument("--folds", type=int, default=10, help="Número de folds para validação cruzada (K-Fold)")
    parser.add_argument("--dataset-path", type=str, default="./support_images/dataset/raw", help="Caminho para o diretório do dataset")
    parser.add_argument("--mask-path", type=str, default="./support_images/dataset/raw", help="Caminho para o diretório das máscaras")
    parser.add_argument("--simple", action="store_true", help="Utiliza a versão simplificada da U-Net")

    args = parser.parse_args()
    if "run" in args.acao:
        unet.main(
            seed=args.seed,
            input_dimensions=args.dims,
            num_classes=args.classes,
            epochs=args.epochs,
            folds=args.folds,
            dataset_path=args.dataset_path,
            mask_path=args.mask_path,
            simple=args.simple
        )

if __name__ == "__main__":
    main()
