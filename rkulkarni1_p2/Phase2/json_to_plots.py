import json
import numpy as np
import matplotlib.pyplot as plt

path_lego = "./rkulkarni1_p2/Phase2/outputs/training_loss_data/nerf_experiment_lego_3_epochs_final.json"
path_ship = "./rkulkarni1_p2/Phase2/outputs/training_loss_data/nerf_experiment_ship_4_epochs_final.json"
path_lego_without_encoding = "./rkulkarni1_p2/Phase2/outputs/training_loss_data/nerf_experiment_lego_3_epochs_without_encoding_final.json"

with open(path_lego_without_encoding) as f:
    data = json.load(f)

steps = [entry[1] for entry in data]
values = [entry[2] for entry in data]
plt.plot(steps, values, linestyle='-', color='b')
plt.xlabel("Number of Iterations")
plt.ylabel("SSE Loss")
plt.title("Sum Squared Error(SSE) Loss vs. Number of Iterations")
plt.grid(True)
# plt.show()
plt.savefig(f'./rkulkarni1_p2/Phase2/outputs/training_loss_data/loss_vs_iteration_lego_3_epochs_without_encoding.png')
plt.close()

