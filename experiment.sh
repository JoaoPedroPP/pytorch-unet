lidc_regular_dataset_path="./support_images/dataset/raw"
lidc_regular_mask_path="./support_images/dataset/raw"
lidc_edge_dataset_path="./support_images/dataset/raw2"
lidc_edge_mask_path="./support_images/dataset/raw"
epochs=2
folds=2
seed=42

python_cmd() {
    # experimento 1: loss dice
    echo "Experimento 1: loss dice LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_dice/
    mkdir ./support_images/preds
    # experimento 2: loss bce
    echo "Experimento 2: loss bce LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type bce
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_bce/
    mkdir ./support_images/preds
    # experimento 3: loss dice_bce e swap 0.5
    echo "Experimento 3: loss dice_bce e swap 0.5 LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice_bce --swap 0.5
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_dice_bce_swap_0.5/
    mkdir ./support_images/preds
    # experimento 4: loss dice e informacao de borda
    echo "Experimento 4: loss dice e informacao de borda LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_edge_dataset_path --mask-path $lidc_edge_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --dims 2
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_edge_200_epocas_2_folds_dice/
    mkdir ./support_images/preds
    # experimento 5: loss dice unet simples
    echo "Experimento 5: loss dice unet simples LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --simple
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_dice_unet_simples/
    mkdir ./support_images/preds
    # experimento 6: loss dice informacao de borda unet simples
    echo "Experimento 6: loss dice informacao de borda unet simples LIDC"
    touch logs.csv
    python unet-run.py run --dataset-path $lidc_edge_dataset_path --mask-path $lidc_edge_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --simple --dims 2
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_edge_200_epocas_2_folds_dice_unet_simples_informacao_de_borda/
    mkdir ./support_images/preds
}

podman_cmd() {
    podman build -t unet -f ContainerFile .
    #  Experimento 1:
    echo "Experimento 1: loss dice LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_dice/
    mkdir ./support_images/preds
    # Experimento 2:
    echo "Experimento 2: loss bce LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type bce
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_bce/
    mkdir ./support_images/preds
    # Experimento 3:
    echo "Experimento 3: loss dice_bce e swap 0.5 LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_regular_dataset_path --mask-path $lidc_regular_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice_bce --swap 0.5
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_regular_200_epocas_2_folds_dice_bce_swap_0.5/
    mkdir ./support_images/preds
    # Experimento 4:
    echo "Experimento 4: loss dice e informacao de borda LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_edge_dataset_path --mask-path $lidc_edge_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --dims 2
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_edge_200_epocas_2_folds_dice/
    mkdir ./support_images/preds
    # Experimento 5:
    echo "Experimento 5: loss dice unet simples LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_edge_dataset_path --mask-path $lidc_edge_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --simple
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_edge_200_epocas_2_folds_dice_unet_simples/
    mkdir ./support_images/preds
    # Experimento 6:
    echo "Experimento 6: loss dice informacao de borda unet simples LIDC"
    touch logs.csv
    podman run -it -v ./support_images:/usr/app/support_images:z -v ./logs.csv:/usr/app/logs.csv:z unet run --dataset-path $lidc_edge_dataset_path --mask-path $lidc_edge_mask_path --epochs $epochs --folds $folds --seed $seed --loss-type dice --simple --dims 2
    mv logs.csv ./support_images/preds/logs.csv
    mv ./support_images/preds/ ./support_images/preds_imagem_edge_200_epocas_2_folds_dice_unet_simples_informacao_de_borda/
    mkdir ./support_images/preds
}

if [[ $1 == "python" ]]; then
    python_cmd
fi

if [[ $1 == "podman" ]]; then
    podman_cmd
fi