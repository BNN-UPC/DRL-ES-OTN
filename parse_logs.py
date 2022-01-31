# Copyright (c) 2022, Carlos Güemes [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: carlos.guemes@estudiantat.upc.edu

import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # python parse_PPO.py -f dataset_Topologies -g Netrail -d ./Logs/expPPO_agentLogs.txt
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='Log files', type=str, required=True, nargs='+')
    parser.add_argument('-s', help='Optional file path to save the results', type=str, required=True, nargs='+')
    args = parser.parse_args()



    aux = args.d[0].split(".")
    aux = aux[0].split("exp")
    print(aux)
    differentiation_str = str(aux[1])

    all_values = list()
    for ii in range(len(args.d)):
        meanREW = []
        maxREW = []
        stdREW = []
        trTime = []
        with open(args.d[ii]) as fp:
            for line in fp:
                arrayLine = line.split(",")
                if arrayLine[0]=="MeanREW":
                    meanREW.append(float(arrayLine[1]))
                elif arrayLine[0]=="MaxREW":
                    maxREW.append(float(arrayLine[1]))
                elif arrayLine[0]=="STDREW":
                    stdREW.append(float(arrayLine[1]))
                elif arrayLine[0]=="TrTime":
                    trTime.append(float(arrayLine[1]))
                    # trTime.append(7000.0 * len(trTime))
        all_values.append({
            "meanREW": meanREW,
            "maxREW": maxREW,
            "stdREW": stdREW,
            "trTime": trTime
        })

    # Plot evolutions
    for entry, name in zip(all_values, args.d):
        plt.plot(entry["trTime"], entry["meanREW"], label=name if len(name) < 10 else name[-10:])
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward Over Time")
    plt.xlabel("Time (s)")
    plt.savefig(args.s)
    plt.legend()
    plt.show()
    plt.close()

