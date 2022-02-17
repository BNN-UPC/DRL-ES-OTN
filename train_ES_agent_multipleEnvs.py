# Copyright (c) 2022, Carlos Güemes [^1], Paul Almasan [^2]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: carlos.guemes@upc.edu
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import os

from time import perf_counter

import numpy as np
import gym
import gc
import gym_environments
import actorES32 as actor
import tensorflow as tf
import argparse
import ctypes
import optimizers
import json
import pickle
from itertools import takewhile, count
from mpi4py import MPI

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ENV_NAME = 'GraphEnv-v1'
NUM_ACTIONS = 4 # We limit the actions to the K=4 shortest paths

# MPI
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()  # Number of processes in the comm
mpi_rank = mpi_comm.Get_rank()  # ID of process within the comm

# RNG
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)

differentiation_str = "ES_agent"

def mpi_print(*args, **kwargs):
    if mpi_rank == 0:
        print(*args, **kwargs)

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class PerturbationsGenerator:
    """
    Class that obtains the perturbations for this process and all other processes in the comm
    """

    def __init__(self, number_params):
        self.number_params = number_params
        # Initialize RNG
        if mpi_rank:
            self.RNG = np.random.RandomState(mpi_rank)
        else:
            self.RNG = np.array([np.random.RandomState(ii) for ii in range(mpi_size)], dtype=object)

    def obtain_perturbations(self, perturbations):
        # Generate  perturbations
        return self.RNG.randn(perturbations, self.number_params)

    def global_obtain_perturbations(self, episodes_per_worker, episodes_indices, total_episodes):
        # print(episodes_per_worker, episodes_indices, total_episodes, list(episodes_per_worker))
        global_res = np.empty((total_episodes, self.number_params), dtype=np.float32)
        local_res = None
        # print(episodes_per_worker, episodes_indices)
        for ii, episodes in enumerate(list(episodes_per_worker)):
            if episodes:
                temp_res = self.RNG[ii].randn(episodes, self.number_params)
                if not ii:
                    local_res = temp_res
                # print(episodes_indices[ii], episodes_indices[ii+1], episodes)
                global_res[episodes_indices[ii]: episodes_indices[ii+1]] = temp_res
        return local_res, global_res



class ESPolicy:
    def __init__(self,
                 hparams,
                 listofDemands,
                 params_file,
                 tr_dataset_folder_name,
                 tr_graph_topology_names,
                 eval_dataset_folder_name,
                 eval_graph_topology_names,
                 param_noise,
                 action_noise,
                 number_mutations):

        # PPO trick from https://costa.sh/blog-the-32-implementation-details-of-ppo.html
        hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
        kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)

        # Model to build
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()
        self.actor_hparams = hparams
        # Current set of weights
        self.current_weights = self.actor.get_weights()
        # Random number stream
        self.policy_rng = np.random.RandomState(1)
        # Enviroments
        self.tr_envs = list()
        for dataset_folder_name, graph_topology_name in zip(tr_dataset_folder_name, tr_graph_topology_names):
            self.tr_envs.append(gym.make(ENV_NAME))
            self.tr_envs[-1].seed(SEED)
            self.tr_envs[-1].generate_environment(listofDemands, dataset_folder_name, graph_topology_name, NUM_ACTIONS)
        self.current_tr_environment = None
        self.eval_envs = list()
        for dataset_folder_name, graph_topology_name in zip(eval_dataset_folder_name, eval_graph_topology_names):
            self.eval_envs.append(gym.make(ENV_NAME))
            self.eval_envs[-1].seed(SEED)
            self.eval_envs[-1].generate_environment(listofDemands, dataset_folder_name, graph_topology_name, NUM_ACTIONS)
        # Parameters for policy rollout
        self.ACTION_NOISE_STD = action_noise
        self.PARAM_NOISE_STD = param_noise

        if params_file:
            try:
                with open("saved_params/{}.pkcl".format(params_file), 'rb') as ff:
                    self.current_weights = pickle.load(ff)
                self.actor.set_weights(self.current_weights)
            except FileNotFoundError as err:
                # The parameters haven't been saved until now: we store them for future reference
                with open("saved_params/{}.pkcl".format(params_file), 'wb') as ff:
                    pickle.dump(self.current_weights, ff)
        else:
            mpi_print("Warning: No hyperparams file detected!")

        # Total Number of mutations
        self.total_number_mutations = number_mutations


    def tr_rollout(self):

        # Accumulator of rewards
        total_rewards = 0.0
        # Enviroment variables (State)
        demand, source, destination = self.tr_envs[self.current_tr_environment].reset()
        done = False
        # Iterate through the enviroment
        while not done:
            action_dist = self.pred_action_node_distrib(self.tr_envs[self.current_tr_environment], source, destination,
                                                        demand)
            # If needed add noise to the action distribution
            if self.ACTION_NOISE_STD > 0:
                action_dist += self.policy_rng.randn(action_dist.size) * self.ACTION_NOISE_STD
                action_dist[action_dist < 0] = 0  # We remove negative terms possibly caused by noise
                action_dist /= action_dist.sum()
            action = action_dist.argmax()
            # Allocate the traffic of the demand to the shortest path
            reward, done, demand, source, destination = self.tr_envs[self.current_tr_environment].step(action, demand,
                                                                                                       source,
                                                                                                       destination)
            total_rewards += reward
        # Return aggregated results
        return total_rewards

    def perform_tr_rollouts(self, perturbations, iteration):
        # If we have no perturbations, perform no rollouts
        if perturbations.shape[0] == 0:
            return NULL_BUFF, NULL_BUFF

        # Select current environment
        self.current_tr_environment = iteration % len(self.tr_envs)

        # Define rewards buffer
        positive_rewards = np.empty(perturbations.shape[0], dtype=np.float32)
        negative_rewards = np.empty_like(positive_rewards)

        for ii in range(positive_rewards.size):
            noise = perturbations[ii] * self.PARAM_NOISE_STD
            # Positive noise
            self.actor.set_weights(self.current_weights + noise)
            positive_rewards[ii] = self.tr_rollout()
            # Negative reward
            self.actor.set_weights(self.current_weights - noise)
            negative_rewards[ii] = self.tr_rollout()

        # Reset weights
        self.actor.set_weights(self.current_weights)
        # Obtain both rewards per rank
        return positive_rewards, negative_rewards

    def eval_rollout(self, env):
        # List of rewards
        total_rewards = 0.0
        # Enviroment variables (State)
        demand, source, destination = env.reset()
        done = False
        # Iterate through the enviroment
        while not done:
            action_dist = self.pred_action_node_distrib(env, source, destination, demand)
            # If needed add noise to the action distribution
            action = action_dist.argmax()
            # Allocate the traffic of the demand to the shortest path
            reward, done, demand, source, destination = env.step(action, demand, source, destination)
            total_rewards += reward
        # Return aggregated results
        return total_rewards

    def perform_eval_rollouts(self, evaluation_episodes, file_name):
        if evaluation_episodes == 0:
            return NULL_BUFF

        local_eval_rewards = np.empty((evaluation_episodes, len(self.eval_envs)), dtype=np.float32)

        for ii, env in enumerate(self.eval_envs):
            # Perform evaluations in the current environment
            for eval_iter in range(evaluation_episodes):
                local_eval_rewards[eval_iter, ii] = self.eval_rollout(env)

        # Return the local rewards
        return local_eval_rewards

    def pred_action_node_distrib(self, env, source, destination, demand):
        """
        Method to obtain the action distribution
        """
        # List of graph features that are used in the cummax() call
        list_k_features = list()

        k_path = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while k_path < len(env.allPaths[str(source) +':'+ str(destination)]):
            env.mark_action_k_path(k_path, source, destination, demand)

            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            env.edge_state[:,1] = 0
            k_path = k_path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict action probabilities (i.e., one per graph/action)
        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'], training=False)
        listQValues = tf.reshape(r, (1, len(r)))
        softMaxQValues = tf.nn.softmax(listQValues)

        # Return action distribution
        return softMaxQValues.numpy()[0]
    
    def get_graph_features(self, env, source, destination):
        """
        Obtain graph features for model
        """
        # We normalize the capacities
        capacity_feature = env.edge_state[:,0] / env.maxCapacity

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': tf.convert_to_tensor(value=capacity_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=env.bw_allocated_feature, dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        # The hidden states of the links are composed of the link capacity and bw_allocated padded with zeros
        # Notice that the bw_allocated is stored as one-hot vector encoding to make it easier to learn for the GNN
        hiddenStates = tf.concat([sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, self.actor_hparams['link_state_dim'] - 1 - self.actor_hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def update_actor_params(self, gradients):
        self.current_weights += gradients
        self.actor.set_weights(self.current_weights)

    def get_num_params(self):
        return self.actor.get_number_weights()[1]


def obtain_ranking(vec):
    """
    Obtain a ranking centered at 0 ([-0.5, 0.5])
    """
    ranks = np.empty(vec.size, dtype=np.float32)
    ranks[vec.argsort()] = np.arange(vec.size, dtype=np.float32)
    return (ranks / (vec.size - 1)) - 0.5


def distribute_episodes(number_episodes, number_processes, heuristic_handicap=1):
    # Split the number of episodes between processes
    if number_episodes < number_processes:
        distributed_episodes = np.zeros(number_processes, dtype=int)
        distributed_episodes[-number_episodes:] = 1
        mpi_print("WARNING: less episodes than processes")
    else:
        distributed_episodes = np.empty(number_processes, dtype=int)
        min_episodes_per_episode = number_episodes // number_processes
        distributed_episodes[:] = min_episodes_per_episode

        remainder_episodes = number_episodes - (min_episodes_per_episode * number_processes)
        if 0 < remainder_episodes:
            distributed_episodes[:remainder_episodes] += 1

        # Heuristic: remove episodes from the main process to compensate for its additional responsibilities
        if heuristic_handicap and number_processes > 1:
            if distributed_episodes[0] < heuristic_handicap:
                heuristic_handicap = distributed_episodes[0]
                distributed_episodes[0] = 0
            else:
                distributed_episodes[0] -= heuristic_handicap
            # Start assigning them to the back first, as before we where handing it to the front
            min_episodes_per_episode = heuristic_handicap // (number_processes - 1)
            distributed_episodes[1:] += min_episodes_per_episode
            remainder_episodes = heuristic_handicap - (min_episodes_per_episode * (number_processes - 1))
            if 0 < remainder_episodes:
                distributed_episodes[-remainder_episodes:] += 1

    # Obtain the positions (accumulated indices)
    accumulated_episodes = np.zeros_like(distributed_episodes)
    for ii in range(1, number_processes):
        accumulated_episodes[ii] = accumulated_episodes[ii-1] + distributed_episodes[ii-1]

    # Obtain the positions to load the episodes
    loaded_episodes = np.zeros(number_processes + 1, dtype=int)
    loaded_episodes[:number_processes] = accumulated_episodes
    loaded_episodes[number_processes] = number_episodes

    return tuple(distributed_episodes), tuple(accumulated_episodes), loaded_episodes

def main_process():
    # Prepare optimizer
    optimizer = getattr(optimizers, HYPERPARAMS["optimizer"])(alpha=HYPERPARAMS["lr"])
    # Obtain the normalization factor for the gradient
    GRADIENT_FACTOR = 1 / (2 * HYPERPARAMS["number_mutations"] * HYPERPARAMS["param_noise_std"])

    # Name of training evaluations
    training_graph_names = "+".join(HYPERPARAMS["tr_graph_topologies"])
    checkpoint_dir = "./models/{}_numProc_{}".format(training_graph_names, mpi_size)

    # Prepare logs
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")
    config_name_file = args.c.split("/")[2]
    file_logs = list()
    for graph_name in HYPERPARAMS["eval_graph_topologies"]:
        file_logs.append(open("./Logs/exp{}_{}_numProc_{}_{}_{}_Logs.txt".format(training_graph_names, graph_name,
                                                                                 mpi_size, differentiation_str,
                                                                                 config_name_file), "w"))
    file_logs.append(open("./Logs/exp{}_{}_numProc_{}_{}_{}_Logs.txt".format(training_graph_names,
                                                                             "+".join(HYPERPARAMS["eval_graph_topologies"]),
                                                                             mpi_size, differentiation_str,
                                                                             config_name_file), "w"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor)

    # Total time
    total_time = 0.0
    t1 = t2 = t3 = t4 = t5 = t6 = t7 = None
    rollout_time_buff = np.empty(1, dtype=np.float32)

    # Initial evaluation
    total_eval_rewards = np.empty((EVAL_LOADING_INDICES[-1], len(HYPERPARAMS["eval_graph_topologies"])),
                                  dtype=np.float32)
    # Perform evaluations
    local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank], "eval_init.pckl")
    # Gather results
    mpi_comm.Gatherv(local_eval_rewards, [total_eval_rewards, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES,
                                          MPI.FLOAT], 0)
    # Write results
    for ii, file_log in enumerate(file_logs[:-1]):
        file_log.write("MeanREW," + str(total_eval_rewards[:, ii].mean()) + ",\n")
        file_log.write("MaxREW," + str(total_eval_rewards[:, ii].max()) + ",\n")
        file_log.write("STDREW," + str(np.std(total_eval_rewards[:, ii])) + ",\n")
        file_log.write("TrTime," + str(total_time) + ",\n")
        file_log.flush()
    # Log for global results
    file_logs[-1].write("MeanREW," + str(total_eval_rewards.mean()) + ",\n")
    file_logs[-1].write("MaxREW," + str(total_eval_rewards.max()) + ",\n")
    file_logs[-1].write("STDREW," + str(np.std(total_eval_rewards)) + ",\n")
    file_logs[-1].write("TrTime," + str(total_time) + ",\n")
    file_logs[-1].flush()

    # Main loop
    for iters in range(HYPERPARAMS["episode_iterations"]):
        print("OTN ROUTING ({} Topology) ES Iteration {}".format(training_graph_names, iters))

        # Obtain global perturbations
        t1 = perf_counter()
        local_perturbations, global_perturbations = perturbations_gen.global_obtain_perturbations(EPISODES_PER_WORKER,
                                                                                                  LOADING_INDICES,
                                                                                                  HYPERPARAMS["number_mutations"])
        t2 = perf_counter()
        # Allocate memory for total rewards
        global_positive_rewards = np.empty(HYPERPARAMS["number_mutations"], dtype=np.float32)
        global_negative_rewards = np.empty_like(global_positive_rewards)
        t3 = perf_counter()

        # Perform own episodes (if necessary)
        local_positive_rewards, local_negative_rewards = agent.perform_tr_rollouts(local_perturbations, iters)

        mpi_comm.Reduce(np.zeros(1, dtype=np.float32), rollout_time_buff, op=MPI.MAX)
        t4 = perf_counter()
        mpi_comm.Gatherv(local_positive_rewards, [global_positive_rewards, EPISODES_PER_WORKER, MPI_INDICES, MPI.FLOAT])
        mpi_comm.Gatherv(local_negative_rewards, [global_negative_rewards, EPISODES_PER_WORKER, MPI_INDICES, MPI.FLOAT])
        t5 = perf_counter()

        # Perform reward scaling
        ranked_positive_rewards = obtain_ranking(global_positive_rewards)
        ranked_negative_rewards = obtain_ranking(global_negative_rewards)
        del global_positive_rewards, global_negative_rewards
        # obtain positive and negative rewards
        final_rewards = ranked_positive_rewards - ranked_negative_rewards
        del ranked_positive_rewards, ranked_negative_rewards
        assert final_rewards.size == global_perturbations.shape[0]
        t6 = perf_counter()

        print("OTN ROUTING ({} Topology) ES Iteration {} -- Calculating gradient".format(training_graph_names, iters))

        # Obtain gradients
        # We need to invert it since we want to maximize, but the optimizer is meant for minimization
        gradient = -1 * np.dot(final_rewards, global_perturbations) * GRADIENT_FACTOR
        # We also consider the L2 regularization
        l2_regularization = agent.current_weights * HYPERPARAMS["l2_coeff"]
        # Obtain updated parameters
        parameters = optimizer.optimize(gradient + l2_regularization)
        mpi_comm.Bcast(parameters, root=0)
        agent.update_actor_params(parameters)
        del final_rewards, global_perturbations, l2_regularization, parameters

        # Time calculation
        t7 = perf_counter()
        total_time += t7 - t1
        noise_selection_time = t2 - t1
        rewards_allocation_time = t3 - t2
        rollout_time = rollout_time_buff[0] if mpi_size > 1 else t4 - t3
        mpi_allgather_time = t5 - t4
        normalizing_reward_time = t6 - t5
        backpropagation_time = t7 - t5

        # Logging (only if rank==0)
        checkpoint_actor.save(checkpoint_prefix + '_ACT')

        # Perform evaluation
        if iters % HYPERPARAMS["evaluation_period"] == 0:
            print("OTN ROUTING ({} Topology) ES Iteration {} -- Evaluation          "
                  .format(training_graph_names, iters))
            # Perform evaluations
            local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank],
                                                             "eval_{}.pckl".format(iters))
            # Gather results
            mpi_comm.Gatherv(local_eval_rewards, [total_eval_rewards, ADJ_EVAL_EPISODES_PER_WORKER,
                                                  ADJ_EVAL_MPI_INDICES, MPI.FLOAT], 0)
            # Write results
            for ii, file_log in enumerate(file_logs[:-1]):
                file_log.write("MeanREW," + str(total_eval_rewards[:, ii].mean()) + ",\n")
                file_log.write("MaxREW," + str(total_eval_rewards[:, ii].max()) + ",\n")
                file_log.write("STDREW," + str(np.std(total_eval_rewards[:, ii])) + ",\n")
                file_log.write("TrTime," + str(total_time) + ",\n")
                file_log.write("NoiseSelTime," + str(noise_selection_time) + ",\n")
                file_log.write("MemAllocTime," + str(rewards_allocation_time) + ",\n")
                file_log.write("RolloutTime," + str(rollout_time) + ",\n")
                file_log.write("MPITime," + str(mpi_allgather_time) + ",\n")
                file_log.write("NormTime," + str(normalizing_reward_time) + ",\n")
                file_log.write("BackpropTime," + str(backpropagation_time) + ",\n")
                file_log.flush()
            # Log for global results
            file_logs[-1].write("MeanREW," + str(total_eval_rewards.mean()) + ",\n")
            file_logs[-1].write("MaxREW," + str(total_eval_rewards.max()) + ",\n")
            file_logs[-1].write("STDREW," + str(np.std(total_eval_rewards)) + ",\n")
            file_logs[-1].write("TrTime," + str(total_time) + ",\n")
            file_logs[-1].write("NoiseSelTime," + str(noise_selection_time) + ",\n")
            file_logs[-1].write("MemAllocTime," + str(rewards_allocation_time) + ",\n")
            file_logs[-1].write("RolloutTime," + str(rollout_time) + ",\n")
            file_logs[-1].write("MPITime," + str(mpi_allgather_time) + ",\n")
            file_logs[-1].write("NormTime," + str(normalizing_reward_time) + ",\n")
            file_logs[-1].write("BackpropTime," + str(backpropagation_time) + ",\n")
            file_logs[-1].flush()


def worker_process():
    # Initial evaluation
    # Perform evaluations
    local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank], "eval_init.pckl")
    # Gather results
    mpi_comm.Gatherv(local_eval_rewards, [NULL_BUFF, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES, MPI.FLOAT], 0)

    # Prepare time vars
    t1 = t2 = 0
    # Prepare parameters buffer
    parameters = np.empty(number_params, dtype=np.float32)

    # Main loop
    for iters in range(HYPERPARAMS["episode_iterations"]):

        # Sample random numbers
        local_perturbations = perturbations_gen.obtain_perturbations(EPISODES_PER_WORKER[mpi_rank])
        # Perform policy rollouts
        t1 = perf_counter()
        local_positive_rewards, local_negative_rewards = agent.perform_tr_rollouts(local_perturbations, iters)
        t2 = perf_counter()

        # Send numbers
        mpi_comm.Reduce(np.array([t2-t1], dtype=np.float32), NULL_BUFF, op=MPI.MAX)
        mpi_comm.Gatherv(local_positive_rewards, [NULL_BUFF, EPISODES_PER_WORKER, MPI_INDICES,
                                                  MPI.FLOAT])
        mpi_comm.Gatherv(local_negative_rewards, [NULL_BUFF, EPISODES_PER_WORKER, MPI_INDICES,
                                                  MPI.FLOAT])
        
        # Update gradients
        mpi_comm.Bcast(parameters, root=0)
        agent.update_actor_params(parameters)

        # Perform evaluation
        if iters % HYPERPARAMS["evaluation_period"] == 0:
            # Perform evaluations
            local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank],
                                                             "eval_{}.pckl".format(iters))
            # Gather results
            mpi_comm.Gatherv(local_eval_rewards, [NULL_BUFF, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES,
                                                  MPI.FLOAT], 0)


if __name__ == "__main__":
    mpi_print("script start")
    # Command to train the PPO agent:
    # python train_ES_agent.py -e 1000 -f dataset_Topologies -g Netrail
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-c', help='Configuration file with hyperparams', type=str, required=True)
    args = parser.parse_args()

    NULL_BUFF = np.empty(0, dtype=np.float32)

    # Load Configuration
    with open(args.c, 'r') as config_file:
        HYPERPARAMS = json.loads(config_file.read())

    # Get the environment and extract the number of actions.
    agent = ESPolicy(HYPERPARAMS["gnn-params"],
                     HYPERPARAMS["list_of_demands"],
                     HYPERPARAMS.get("params_file", None),
                     HYPERPARAMS["tr_dataset_folder_name"],
                     HYPERPARAMS["tr_graph_topologies"],
                     HYPERPARAMS["eval_dataset_folder_name"],
                     HYPERPARAMS["eval_graph_topologies"],
                     HYPERPARAMS["param_noise_std"],
                     HYPERPARAMS["action_noise_std"],
                     HYPERPARAMS["number_mutations"])
    # Get the number of weights
    number_params = agent.get_num_params()

    # The true number of mutations will be double due to mirror sampling
    EPISODES_PER_WORKER, MPI_INDICES, LOADING_INDICES = distribute_episodes(HYPERPARAMS["number_mutations"], mpi_size)

    EVAL_EPISODES_PER_WORKER, EVAL_MPI_INDICES, EVAL_LOADING_INDICES = distribute_episodes(
        HYPERPARAMS["evaluation_episodes"], mpi_size, 0)
    # Adjusted loading constants for multiple-environment evaluation
    ADJ_EVAL_EPISODES_PER_WORKER = tuple(ii * len(HYPERPARAMS["eval_graph_topologies"])
                                         for ii in EVAL_EPISODES_PER_WORKER)
    ADJ_EVAL_MPI_INDICES = tuple(ii * len(HYPERPARAMS["eval_graph_topologies"]) for ii in EVAL_MPI_INDICES)

    # Sample random numbers
    perturbations_gen = PerturbationsGenerator(number_params)

    if mpi_rank:
        worker_process()
    else:
        main_process()

    mpi_comm.Barrier()
