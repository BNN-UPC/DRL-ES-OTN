# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import numpy as np
import gym
import gc
import os
import gym_environments
import random
import criticPPO as critic
import actorPPO as actor
import tensorflow as tf
from collections import deque
#import time as tt
import argparse
import pickle
import heapq
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ENV_NAME = 'GraphEnv-v1'

BUFF_SIZE = 10000 # Experience buffer size. Careful to don't have more samples from one TM!

EVALUATION_EPISODES = 40 # Number of evaluation episodes. Usually 40-50 it's enough but we can make it shorter for faster experiments
TRAINING_EPISODES = 100 # Very important parameter!
SEED = 9
PPO_EPOCHS = 50 # Very important parameter!
MINI_BATCH_SIZE = 20 # Very important parameter!

DECAY_STEPS = 80 # To indicate every how many PPO EPISODES we decay the lr
DECAY_RATE = 0.96

# If using a neural network architecture that shares parameters between the policy and value function, 
# we must use a loss function that combines the policy surrogate and a value function error term.
CRITIC_DISCOUNT = 0.8

# if agent struggles to explore the environment, increase BETA
# if the agent instead is very random in its actions, not allowing it to take good decisions, you should lower it
ENTROPY_BETA = 0.01
ENTROPY_STEP = 120
# check https://github.com/dennybritz/reinforcement-learning/issues/34 for understanding entropy

clipping_val = 0.1
gamma = 0.99
lmbda = 0.95

max_grad_norm = 0.5

differentiation_str = "PPO_agent"
checkpoint_dir = "./models"+differentiation_str

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(1)

train_dir = "./TensorBoard/"+differentiation_str
summary_writer = tf.summary.create_file_writer(train_dir)
global_step = 0
NUM_ACTIONS = 4 # We limit the actions to the K=4 shortest paths

# PPO trick from https://costa.sh/blog-the-32-implementation-details-of-ppo.html
hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)

listofDemands = [8, 32, 64]

hparams = {
    'l2': 0.005,
    'dropout_rate': 0.1,
    'link_state_dim': 25, # Very important parameter!
    'readout_units': 20, # Very important parameter!
    'learning_rate': 0.0002,
    'T': 5,
    'num_demands': len(listofDemands)
}

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

def decayed_learning_rate(step):
    lr = hparams['learning_rate']*(DECAY_RATE ** (step / DECAY_STEPS))
    if lr<10e-8:
        lr = 10e-8
    return lr

class PPOActorCritic:
    def __init__(self, env_training):
        self.memory = deque(maxlen=BUFF_SIZE)
        self.inds = None
        self.listQValues = None
        self.softMaxQValues = None
        self.global_step = global_step

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None
        self.K = env_training.K

        self.capacity_feature = None

        # print(agent.optimizer._decayed_lr(tf.float32).numpy())
        # One step is one PPO_EPOCH or what is the same as one pass on the _train_step_combined()
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    hparams['learning_rate'],
                    decay_steps=DECAY_STEPS,
                    decay_rate=DECAY_RATE,
                    staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, beta_1=0.9, epsilon=1e-05)

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        self.critic = critic.myModel(hparams, hidden_init_critic, kernel_init_critic)
        self.critic.build()
    
    def pred_action_node_distrib(self, env, source, destination, demand):
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
        self.listQValues = tf.reshape(r, (1, len(r)))
        self.softMaxQValues = tf.nn.softmax(self.listQValues)

        # Return action distribution
        return self.softMaxQValues.numpy()[0], tensor
    
    def get_graph_features(self, env, source, destination):
        # We normalize the capacities
        self.capacity_feature = env.edge_state[:,0] / env.maxCapacity

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=env.bw_allocated_feature, dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        # The hidden states of the links are composed of the link capacity and bw_allocated padded with zeros
        # Notice that the bw_allocated is stored as one-hot vector encoding to make it easier to learn for the GNN
        hiddenStates = tf.concat([sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 1 - hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def critic_get_graph_features(self, env):

        self.capacity_feature = env.edge_state[:,0] / env.maxCapacity

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        # In the critic we don't use the bw_allocated link feature
        hiddenStates = tf.concat([sample['capacity']], axis=1)
        # Adds padding
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 1]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    
    def _write_tf_summary(self, actor_loss, critic_loss, final_entropy):
        # This function is to write the gradient evolution during training.
        # Probably it needs to be adapted as I don't use it since long ago. The self.global_step is very important
        # as it marks the time-step for the gradients
        with summary_writer.as_default():
            tf.summary.scalar(name="actor_loss", data=actor_loss, step=self.global_step)
            tf.summary.scalar(name="critic_loss", data=critic_loss, step=self.global_step)  
            tf.summary.scalar(name="entropy", data=-final_entropy, step=self.global_step)                      

            tf.summary.histogram(name='ACTOR/FirstLayer/kernel:0', data=self.actor.variables[0], step=self.global_step)
            tf.summary.histogram(name='ACTOR/FirstLayer/bias:0', data=self.actor.variables[1], step=self.global_step)
            tf.summary.histogram(name='ACTOR/kernel:0', data=self.actor.variables[2], step=self.global_step)
            tf.summary.histogram(name='ACTOR/recurrent_kernel:0', data=self.actor.variables[3], step=self.global_step)
            tf.summary.histogram(name='ACTOR/bias:0', data=self.actor.variables[4], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/kernel:0', data=self.actor.variables[5], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/bias:0', data=self.actor.variables[6], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/kernel:0', data=self.actor.variables[7], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/bias:0', data=self.actor.variables[8], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/kernel:0', data=self.actor.variables[9], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/bias:0', data=self.actor.variables[10], step=self.global_step)
            
            tf.summary.histogram(name='CRITIC/FirstLayer/kernel:0', data=self.critic.variables[0], step=self.global_step)
            tf.summary.histogram(name='CRITIC/FirstLayer/bias:0', data=self.critic.variables[1], step=self.global_step)
            tf.summary.histogram(name='CRITIC/kernel:0', data=self.critic.variables[2], step=self.global_step)
            tf.summary.histogram(name='CRITIC/recurrent_kernel:0', data=self.critic.variables[3], step=self.global_step)
            tf.summary.histogram(name='CRITIC/bias:0', data=self.critic.variables[4], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/kernel:0', data=self.critic.variables[5], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/bias:0', data=self.critic.variables[6], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/kernel:0', data=self.critic.variables[7], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/bias:0', data=self.critic.variables[8], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/kernel:0', data=self.critic.variables[9], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/bias:0', data=self.critic.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1
    
    # tf.function is an optimization to avoid generating the tf graph for each new call. It works
    # much faster but we should be careful how to use it. If 'sample' has different shapes in each call,
    # this results in multiple tf graph generations, and this leads to a kind of memory leak!
    @tf.function
    def _critic_step(self, sample):
        value = self.critic(sample['link_state_critic'], sample['first_critic'], sample['second_critic'],
                        sample['num_edges_critic'], training=True)[0]
        critic_sample_loss = K.square(sample['return'] - value)
        return critic_sample_loss
    
    @tf.function
    def _actor_step(self, sample):
        r = self.actor(sample['link_state'], sample['graph_id'], sample['first'], sample['second'], 
                sample['num_edges'], training=True)
        qvalues = tf.reshape(r, (1, len(r)))
        newpolicy_probs = tf.nn.softmax(qvalues)
        newpolicy_probs2 = tf.math.reduce_sum(sample['old_act'] * newpolicy_probs[0])

        ratio = K.exp(K.log(newpolicy_probs2) - K.log(tf.math.reduce_sum(sample['old_act']*sample['old_policy_probs'])))
        surr1 = -ratio*sample['advantage']
        surr2 = -K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * sample['advantage']
        loss_sample = tf.maximum(surr1, surr2)

        entropy_sample = -tf.math.reduce_sum(K.log(newpolicy_probs) * newpolicy_probs[0])
        return loss_sample, entropy_sample

    def _train_step_combined(self, inds):
        entropies = []
        actor_losses = []
        critic_losses = []
        # Optimize weights
        with tf.GradientTape() as tape:
            for minibatch_ind in inds:
                sample = self.memory[minibatch_ind]

                # ACTOR
                loss_sample, entropy_sample = self._actor_step(sample)
                actor_losses.append(loss_sample)
                entropies.append(entropy_sample)
                
                # CRITIC
                critic_sample_loss = self._critic_step(sample)
                critic_losses.append(critic_sample_loss)
            
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = tf.math.reduce_mean(actor_losses) - ENTROPY_BETA * final_entropy
            total_loss = actor_loss + critic_loss
        
        grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)
        #gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        grad, _grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
        del tape
        entropies.clear()
        actor_losses.clear()
        critic_losses.clear()
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, env_training, tensors, critic_features, returns, advantages):
        num_samples_buff = len(actions_probs)

        for pos in range(0, num_samples_buff):

            tensor = tensors[pos]
            critic_feature = critic_features[pos]
            action = actions[pos]
            ret_value = returns[pos]
            adv_value = advantages[pos]
            action_dist = actions_probs[pos]
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'],
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      

            self.memory.append(final_tensors)  

        # We use the indices to iterate over all samples but in different order 
        # To make shuffles over all samples for each epoch
        self.inds = np.arange(num_samples_buff)

        for i in range(PPO_EPOCHS):
            np.random.shuffle(self.inds)

            for start in range(0, num_samples_buff, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                actor_loss, critic_loss, final_entropy = self._train_step_combined(self.inds[start:end])

        self.memory.clear()
        self._write_tf_summary(actor_loss, critic_loss, final_entropy)
        gc.collect()
        return actor_loss, critic_loss

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    # Normalize advantages to reduce variance
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # Command to train the PPO agent:
    # python train_PPO_agent.py -e 1000 -f dataset_Topologies -g Netrail
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    parser.add_argument('-f', help='dataset folder name', type=str, required=True, nargs='+')
    parser.add_argument('-g', help='graph topology name', type=str, required=True, nargs='+')
    args = parser.parse_args()

    dataset_folder_name = args.f[0]+'/'
    graph_topology_name = args.g[0]

    # Get the environment and extract the number of actions.
    env_training = gym.make(ENV_NAME)
    env_training.seed(SEED)
    env_training.generate_environment(listofDemands, dataset_folder_name, graph_topology_name, NUM_ACTIONS)

    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(listofDemands, dataset_folder_name, graph_topology_name, NUM_ACTIONS)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    # In this file we store the logs
    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    # Manual Decay lr used in other experiments but could be useful
    # if args.i%DECAY_STEPS==0:
    #     hparams['learning_rate'] = decayed_learning_rate(args.i)

    agent = PPOActorCritic(env_training)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    max_reward = -1000
    reward_id = 0
    evalMeanReward = 0
    counter_store_model = 0

    rewards_test = np.zeros(EVALUATION_EPISODES)
    error_links = np.zeros(EVALUATION_EPISODES)
    max_link_uti = np.zeros(EVALUATION_EPISODES)
    min_link_uti = np.zeros(EVALUATION_EPISODES)
    uti_std = np.zeros(EVALUATION_EPISODES)

    training_tm_ids = set(range(100))

    for iters in range(args.e):
        # We add the list of tmids that we used during training to avoid repetitions
        list_repeated_tm_ids = set()

        states = []
        critic_features = []
        tensors = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []


        # We are intereseted to reduce exploration after some steps
        if iters>=ENTROPY_STEP:
            ENTROPY_BETA = ENTROPY_BETA/10

        print("OTN ROUTING("+graph_topology_name+" Topology) PPO ITERATION: ", iters)
        episode_number = 0
        number_samples_reached = False
        while not number_samples_reached:
            demand, source, destination = env_training.reset()
            while 1:
                # Used to clean the TF cache. I had some strange memory leak with old tf verisons
                # and I still use it as I'm not sure If they have solved it
                tf.random.set_seed(1)
                # Predict probabilities over actions
                action_dist, tensor = agent.pred_action_node_distrib(env_training, source, destination, demand)
                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                features = agent.critic_get_graph_features(env_training)

                q_value = \
                agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                             features['num_edges_critic'], training=False)[0].numpy()[0]

                # Allocate the traffic of the demand to the shortest path
                reward, done, new_demand, new_source, new_destination = env_training.step(action, demand, source, destination)
                mask = not done



                states.append((env_training.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                if done:
                    episode_number += 1
                    break
                
            # If we have enough samples
            if TRAINING_EPISODES == episode_number:
                number_samples_reached = True
                break

        features = agent.critic_get_graph_features(env_training)
        q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                features['num_edges_critic'], training=False)[0].numpy()[0]       
        values.append(q_value)
        # Compute GAEs
        returns, advantages = get_advantages(values, masks, rewards)

        # Training happens here
        actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, env_training, tensors, critic_features, returns, advantages)
        fileLogs.write("a," + str(actor_loss.numpy()) + ",\n")
        fileLogs.write("c," + str(critic_loss.numpy()) + ",\n")
        fileLogs.flush()

        # We evaluate the agent for EVALUATION_EPISODES
        for eps in range(EVALUATION_EPISODES):
            demand, source, destination = env_eval.reset()
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_node_distrib(env_eval, source, destination, demand)
                
                action = np.argmax(action_dist)
                reward, done, new_demand, new_source, new_destination = env_eval.step(action, demand, source, destination)
                rewardAddTest += reward

                if done:
                    break

                demand = new_demand
                source = new_source
                destination = new_destination

            rewards_test[eps] = rewardAddTest

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write("ENTR," + str(ENTROPY_BETA) + ",\n")
        fileLogs.write("REW," + str(evalMeanReward) + ",\n")
        fileLogs.write("lr," + str(agent.optimizer._decayed_lr(tf.float32).numpy()) + ",\n")
  
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        
        # Store trained model
        # Storing the model and the tape.gradient make the memory increase
        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        gc.collect()

    f = open("./tmp/" + differentiation_str + "tmp.pckl", 'wb')
    pickle.dump((max_reward, hparams['learning_rate']), f)
    f.close()
