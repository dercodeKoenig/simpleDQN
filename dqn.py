import tensorflow as tf
import random
from collections import deque
import numpy as np

class DQNAgent:
    def __init__(self, model, n_actions, memory_size = 10000, p_exploration_start = 1,optimizer = tf.keras.optimizers.Adam(0.0005), p_exploration_decay_factor = 0.9999, gamma = 0.99, batch_size =32, name = "dqn1", target_model_sync = 1000):
        self.target_model_sync = target_model_sync
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.model = model
        self.name = name
        self.memory_size = memory_size
        self.optimizer = optimizer
        self.p_exploration = p_exploration_start
        self.p_exploration_decay_factor = p_exploration_decay_factor
        self.m1 = np.eye(self.n_actions)
        self.target_model = tf.keras.models.clone_model(self.model)
        self.total_steps_trained = 0
        
        
        self.memory = deque(maxlen = self.memory_size)
        
    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())
        
    def load_weights(self):
        self.model.load_weights(self.name)
    def save_weights(self):
        self.model.save_weights(self.name, overwrite = True)
        
    def select_actions(self, current_states, training = False):
        if training:
            self.p_exploration *= self.p_exploration_decay_factor
            if np.random.uniform(0,1) < self.p_exploration:
                return [np.random.choice(range(self.n_actions)) for _ in range(np.array(current_states).shape[0])]
            
        q_values = self.model(np.array(current_states, dtype="float32"))
        return [np.argmax(i) for i in q_values]
        
    def observe_sasrt(self, state, action, next_state, reward, terminal):
        self.memory.append([state, action, reward, terminal, next_state])
        
        
    def tstep(self, x, y, masks):
        with tf.GradientTape() as t:
            estimated_q_values = tf.math.reduce_sum(self.model(x, training=True) * masks, axis=1)
            loss = tf.keras.losses.mean_squared_error(y, estimated_q_values)
        
        gradient = t.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return loss
        
    def update_parameters(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        
        sarts_batch = random.sample(self.memory, self.batch_size)
        
        states = [x[0] for x in sarts_batch]
        actions = [x[1] for x in sarts_batch]
        rewards = [x[2] for x in sarts_batch]
        terminals = [x[3] for x in sarts_batch]
        next_states = [x[4] for x in sarts_batch]
        
        masks = np.array(self.m1[actions], dtype="float32")
        
        estimated_q_values_next = self.target_model(np.array(next_states, dtype="float32"))
        q_batch = np.max(estimated_q_values_next, axis=1).flatten()
        target_q_values = np.array([rewards[i] + self.gamma*q_batch[i] if terminals[i] == False else rewards[i] for i in range(self.batch_size)], dtype="float32")
        
        loss = self.tstep(np.array(states, dtype="float32"), target_q_values, masks)
        
        self.total_steps_trained+=1
        if self.total_steps_trained % self.target_model_sync == 0:
            self.copy_weights()
       
        return np.mean(loss), np.mean(estimated_q_values_next)
    
    
    def train(self, num_steps, envs, render = False, warmup = 0, train_steps_per_step = 1):
        total_steps = 0
        num_envs = len(envs)
        states = [x.reset() for x in envs]
        
        losses = [0]
        q_v = [0]
        rewards = []
        rewards_per_episode = []
        n_episodes = 0
        
        current_episode_reward_sum = [0 for _ in range(num_envs)]
        
        progbar = tf.keras.utils.Progbar(num_steps, interval=0.05, stateful_metrics = ["ep_rewards", "n_ep"])
        
        for i in range(num_steps):
            total_steps+=1
            
            actions = self.select_actions(states, training=True)
            returns = []
            for o in range(num_envs):
                returns.append(envs[o].step(actions[o]))

            sasrt_pairs = []
            for index, sample in enumerate(returns):
                sasrt_pairs.append([states[index], actions[index]]+[x for x in sample])

            next_states = [x[2] for x in sasrt_pairs]
                
            reward = [x[3] for x in sasrt_pairs]
            current_episode_reward_sum = [current_episode_reward_sum[i] + reward[i] for i in range(num_envs)]
            rewards.extend(reward)
            
            for index, o in enumerate(sasrt_pairs):
                #print(o)
                if o[4] == True:
                    rewards_per_episode.append(current_episode_reward_sum[index])
                    n_episodes += 1
                    current_episode_reward_sum[index] = 0
                    next_states[index] = envs[index].reset()
                    
                self.observe_sasrt(o[0], o[1], o[2], o[3], o[4])
    
            if render:
                [x.render() for x in envs]
            states = next_states
            if total_steps > warmup:
                for _ in range(train_steps_per_step):
                    loss, q = self.update_parameters()
                    losses.append(loss)
                    q_v.append(q)
            else:
                loss, q = 0, 0
            
            progbar.update(i+1, values = [("loss", np.mean(losses[-train_steps_per_step:])), ("mean q", np.mean(q_v[-train_steps_per_step:])), ("rewards", np.mean(reward)), ("ep_rewards", 0 if len(rewards_per_episode) == 0 else np.mean(rewards_per_episode)), ("n_ep", n_episodes)])
        return losses, q_v, rewards, rewards_per_episode