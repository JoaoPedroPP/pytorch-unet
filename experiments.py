import os
import yaml

from datetime import datetime
from main import main as main_unet

def main():
    with open('Experimentos.yaml', 'r') as file:
        experiments = yaml.load(file, Loader=yaml.FullLoader)

    for experiment in experiments['experimentos']:
        if experiment['execute']:
            os.system(f"touch logs.csv")
            print(experiment['name'])
            main_unet(**experiment['parameters'])
            print(f"Experimento {experiment['name']} finalizado")
            print("Guardando resultados...")
            os.system(f"mv logs.csv ./support_images/preds/logs.csv")
            os.system(f"mv ./support_images/preds ./support_images/{datetime.now().strftime('%Y%m%d_%H%M%S')}_preds_{experiment['filename']}")
            os.system("mkdir ./support_images/preds")
    print("Experimentos finalizados")

if __name__ == "__main__":
    main()