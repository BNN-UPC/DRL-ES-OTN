# ES_OTN_Public
**Carlos Güemes Palau, Paul Almasan, Pere Barlet Ros, Albert Cabellos Aparicio**

Contact us: *[carlos.guemes@estudiantat.upc.edu](mailto:carlos.guemes@estudiantat.upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract
This repository is the code of the paper [Accelerating Deep Reinforcement Learning for Digital Twin Network Optimization with Evolutionary Strategies](https://arxiv.org/abs/2202.00360)

The recent growth of emergent network applications (e.g., satellite networks, vehicular networks) is increasing the complexity of managing modern communication networks. As a result, the community proposed the Digital Twin Networks (DTN) as a key enabler of efficient network management. Network operators can leverage the DTN to perform different optimization tasks (e.g., Traffic Engineering, Network Planning). Deep Reinforcement Learning (DRL) showed a high performance when applied to solve network optimization problems. In the context of DTN, DRL can be leveraged to solve optimization problems without directly impacting the real-world network behavior. However, DRL scales poorly with the problem size and complexity. In this paper, we explore the use of Evolutionary Strategies (ES) to train DRL agents for solving a routing optimization problem. The experimental results show that ES achieved a training time speed-up of 128 and 6 for the NSFNET and GEANT2 topologies respectively.

## Instructions to execute

### Setting up the enviroment

1. First, make sure your OS has a functioning implementation of MPI. We recommend using [OpenMPI](https://www.open-mpi.org/).
2. Create the virtual environment and activate the environment.
```
virtualenv -p python3 myenv
source myenv/bin/activate
```
3. Then we install the required packages
```
pip install -r Prerequisites/requirements.txt
```
or
```
pip install absl-py==0.13.0 astunparse==1.6.3 cachetools==4.2.2 certifi==2021.5.30 charset-normalizer==2.0.2 cloudpickle==1.6.0 cycler==0.10.0 dataclasses==0.8 decorator==4.4.2 flatbuffers==1.12 gast==0.3.3 google-auth==1.33.0 google-auth-oauthlib==0.4.4 google-pasta==0.2.0 grpcio==1.32.0 gym==0.18.3 h5py==2.10.0 idna==3.2 importlib-metadata==4.6.1 Keras==2.4.3 Keras-Preprocessing==1.1.2 kiwisolver==1.3.1 Markdown==3.3.4 matplotlib==3.3.4 mpi4py==3.0.3 networkx==2.5.1 numpy==1.19.5 oauthlib==3.1.1 opt-einsum==3.3.0 pandas==1.1.5 Pillow==8.2.0 pkg_resources==0.0.0 protobuf==3.17.3 pyasn1==0.4.8 pyasn1-modules==0.2.8 pyglet==1.5.15 pyparsing==2.4.7 python-dateutil==2.8.2 pytz==2021.1 PyYAML==5.4.1 requests==2.26.0 requests-oauthlib==1.3.0 rsa==4.7.2 scipy==1.5.4 six==1.15.0 tensorboard==2.5.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.0 tensorflow==2.4.0 tensorflow-estimator==2.4.0 termcolor==1.1.0 typing-extensions==3.7.4.3 urllib3==1.26.6 Werkzeug==2.0.1 wrapt==1.12.1 zipp==3.5.0 kspath
```
NOTE: as an alternative to steps 1-3 you can try to set up a docker image to install both MPI and python. We offer an incomplete dockerfile with the steps needed to cover all the code dependencies at "Prerequisites/sample_dockerfile.dockerfile". The file must be completed so the image also clones the repository and runs the code.

4. Register custom gym environment
````
pip install -e gym-environments/
````

### Running ES
1. Now we can train an ES agent. To do so we execute the following command, choosing an adequate configuration file (*.config).
```
python train_ES_agent_multipleEnvs.py -c path/to/configuration/file.config
```

2. While training occurs the resulting file will be generated in Logs. to visualize the results, we can then plot the results using the following command:
```
python parse_logs.py -d Logs/log1.txt Logs/log2.txt
```
We can add one more log files for the "-d" option. We can also add the "-s" option to store the generated graph in a file:
```
python parse_logs.py -d Logs/log1.txt Logs/log2.txt -s graph_file.png
```

### Running PPO
We also added the code necessary to run the solution using PPO, as to compare its result to ES

1. To train the PPO agent we must execute the following command.
   - The "-e" option controls the number of iterations in the algorithm
   - The "-f" option is used to indicate the folder in which the topologies are stored ("*.graph") files
   - The "-g" option is used to indicate the name of the topology
   - Notice that inside train_PPO_agent.py there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc.
```
python train_PPO_agent.py -e 1500 -f dataset_Topologies -g nsfnet
```
2. Now that the training process is executing, we can see the PPO agent performance evolution by parsing the log files.
   - The "-f" and "-g" options are the same as before
   - The "-d" option is used to indicate the path to the log file
```
python parse_PPO.py -d ./Logs/expsample_PPO_agentLogs.txt -f dataset_Topologies -g nsfnet
```

## Repository contents

- configs: folder containing the configuration files for the code.
  - As it is right now, the different configuration files should be grouped in subfolders (e.g., BatchFinal) as for the correct generation of the log files.
- dataset_Topologies: contains the graph and paths files. The graphs must be represented as ".graph" files
- gym_environments: pip package to be installed, which includes the gym environment to be used to train model
- Logs: contains the logs generated by training of the models
- models: contains the parameters of the network at the different stages of its training. The parameters are stored every time the network is updated. The different models will be divided in subfolders.
- Prerequisites: a folder containing some files that may prove useful to set up the python environment
  - packages.txt: pip freeze of all the python packages needed to run the enviroment.
  - sample_dockerfile.dockerfile: (incomplete) dockerfile to launch an image with all the code requirements fulfilled in order to launch the code.
- saved_params: folder containing the files containing the initial parameters of the network. These can be used to ensure that different executions start from the same initial set of weights.
- tmpPPO: folder needed to store temporal files created by the PPO algorithm

- actorPPO: python file that contains the definition of the actor neural network for PPO.
- criticPPO: python file that contains the definition of the critic neural network for PPO
- parsePPO: python file used to parse PPO's logs
- train_PPO_agent.py: python file that contains the implementation of the PPO algorithm.


- actorES32.py: python file that contains the definition of the neural network for ES
- optimizers.py: python file that contains the implementation of the gradient descent algorithm
- parse_logs.py: python file used to parse ES's logs
- train_ES_agent_multipleEnvs.py: python file that contains the implementation of the ES algorithm.

## Configuration file options

The configuration file is a JSON file that contains the hyperparameters and other variable fields of the algorithm. These fields are as follows:

- gnn-params: dict containing the hyperparameters of the GNN. These are:
  - link_state_dim: dimension of the hidden layer of the message and update functions
  - readout_units: dimension of the hidden layers of the readout function
  - T: number of iterations done for the message passing phase
  - num:demands: number of the number of possible demands sizes done by the environment
- list_of_demands: list containing the possible demand sizes done by the environment
- params_file: (Optional) name of the file that contains the initial weights of the network. If it doesn't exist, it will create one.
- tr_graph_topologies: list of names of the topologies to be used for training the network
- tr_dataset_folder_name: list of paths where the topologies specified in "tr_graph_topologies" can be found
- eval_graph_topologies: list of names of the topologies to be used for evaluating the network. A log file will be generated for every topology listed here, as well as a log file that contains the average result across all the specified topologies.
- eval_dataset_folder_name: list of paths where the topologies specified in "eval_graph_topologies" can be found
- evaluation_episodes: hoy many episodes must be run to evaluate each topology in "eval_graph_topologies".
- evaluation_period: indicates how often the current model has to be evaluated
- number_mutations: Number of perturbations generated (this does NOT include those generated by mirrored sampling, true number will be double)
- l2_coeff: Coefficient to be used for L2 regularization.
- param_noise_std: Standard deviation used to generate the mutations
- action_noise_std: Standard deviation of noise to be added to the action probabilities distributions (0 means no noise is added)
- episode_iterations: number of iterations to run
- optimizer: Type of optimizer to run. Name must match one of the optimizers in "optimizers.py"
- lr: Initial rate of the optimizer

## License
See [LICENSE](LICENSE) for full of the license text.

```
Copyright Copyright 2022 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


