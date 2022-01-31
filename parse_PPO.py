# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from operator import add, sub
from scipy.signal import savgol_filter


if __name__ == "__main__":
    # python parse_PPO.py -f dataset_Topologies -g Netrail -d ./Logs/expPPO_agentLogs.txt
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    parser.add_argument('-f', help='dataset folder name', type=str, required=True, nargs='+')
    parser.add_argument('-g', help='graph topology name', type=str, required=True, nargs='+')
    args = parser.parse_args()

    dataset_folder_name = args.f[0]
    graph_topology_name = args.g[0]
    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    actor_loss = []
    critic_loss = []
    avg_rewards = []
    learning_rate = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    if not os.path.exists("./Images/TRAINING/"+differentiation_str):
        os.makedirs("./Images/TRAINING/"+differentiation_str)
    
    path_to_dir = "./Images/TRAINING/"+differentiation_str+"/"

    model_id = 0
    # Load best model
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    
    print("Model with maximum reward: ", model_id)

    with open(args.d[0]) as fp:
        for line in fp:
            arrayLine = line.split(",")
            if arrayLine[0]=="a":
                actor_loss.append(float(arrayLine[1]))
            elif arrayLine[0]=="lr":
                learning_rate.append(float(arrayLine[1]))
            elif arrayLine[0]=="REW":
                avg_rewards.append(float(arrayLine[1]))
            elif arrayLine[0]=="c":
                critic_loss.append(float(arrayLine[1]))

    plt.plot(actor_loss)
    plt.xlabel("Training Episode")
    plt.ylabel("ACTOR Loss")
    plt.savefig(path_to_dir+"ACTORLoss" + differentiation_str)
    plt.close()

    plt.plot(critic_loss)
    plt.xlabel("Training Episode")
    plt.ylabel("CRITIC Loss (MSE)")
    plt.yscale("log")
    plt.savefig(path_to_dir+"CRITICLoss" + differentiation_str)
    plt.close()

    print("DRL MAX reward: ", np.amax(avg_rewards))

    plt.plot(avg_rewards)
    plt.xlabel("Episodes")
    plt.title("GNN+DQN Testing score")
    plt.ylabel("Average reward")
    plt.savefig(path_to_dir+"AvgReward" + differentiation_str)
    plt.close()

    plt.plot(learning_rate)
    plt.xlabel("Episodes")
    plt.title("GNN+DQN Testing score")
    plt.ylabel("Learning rate")
    plt.savefig(path_to_dir+"Lr_" + differentiation_str)
    plt.close()
