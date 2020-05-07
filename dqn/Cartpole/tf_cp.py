import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import pickle

tf1 = tf.compat.v1

ENV_NAME = "CartPole-v1"
record_filename = './record.dat'

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.9988

MAX_EPISODES = 150


class DQNSolver:

    def __init__(self, observation_space, action_space,seed=3651):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.size_input = observation_space
        self.size_output = action_space
        self.size_hidden1 = 24
        self.size_hidden2 = 24

        self.x = tf.placeholder(tf.dtypes.float64,name = 'x')
        self.y = tf.placeholder(tf.dtypes.float64,name = 'y')
        
        self.wh1 = tf.get_variable(shape=[self.size_input,self.size_hidden1],dtype=tf.dtypes.float64,name='wh1')
        self.bh1 = tf.get_variable(shape=[1,self.size_hidden1],dtype=tf.dtypes.float64,name='bh1')
        self.wh2 = tf.get_variable(shape=[self.size_hidden1,self.size_hidden2],dtype=tf.dtypes.float64,name='wh2')
        self.bh2 = tf.get_variable(shape=[1,self.size_hidden2],dtype=tf.dtypes.float64,name='bh2')
        self.wy = tf.get_variable(shape=[self.size_hidden2,self.size_output],dtype=tf.dtypes.float64,name='wy')
        self.by = tf.get_variable(shape=[1,self.size_output],dtype=tf.dtypes.float64,name='by')

        self.a1 = tf.add(tf.matmul(self.x,self.wh1),self.bh1)
        self.lh1 = tf.nn.relu(self.a1)
        self.a2 = tf.add(tf.matmul(self.lh1,self.wh2),self.bh2)
        self.lh2 = tf.nn.relu(self.a2)
        self.pred = tf.add(tf.matmul(self.lh2,self.wy),self.by)
        
        self.cost = tf.reduce_mean(tf.pow(self.y - self.pred,2.0))
        self.optimizer = tf1.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state,sess):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.pred.eval(feed_dict={self.x:state})
        return np.argmax(q_values[0])

    def experience_replay(self,sess):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        state = np.float32([arr[0] for arr in batch])
        action = np.int32([arr[1] for arr in batch])
        reward = np.float32([arr[2] for arr in batch])
        state_next = np.float32([arr[3] for arr in batch])
        terminal = np.float32([arr[4] for arr in batch])

        qvalues = self.pred.eval(feed_dict={self.x:state_next})
        qvalues = np.float32([np.amax(qvalue) for qvalue in qvalues])
        q_update = reward
        for i in range(BATCH_SIZE):
            if not terminal[i]:
                q_update[i] = (reward[i] + GAMMA * qvalues[i])
        q_values = self.pred.eval(feed_dict={self.x:state})

        for i in range(BATCH_SIZE):
            q_values[i][action[i]] = q_update[i]

        sess.run([self.optimizer, self.cost], feed_dict = {self.x: state, self.y: q_values})
        
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)



env = gym.make(ENV_NAME)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

dqn_solver = DQNSolver(observation_space, action_space)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for ep in range(1,MAX_EPISODES+1):
        state = env.reset()
        step = 0
        while True:
            step += 1
            env.render()
            state = np.reshape(state, [1, observation_space])
            action = dqn_solver.act(state,sess)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state = np.reshape(state, [observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(ep) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                data = [ep,step,dqn_solver.exploration_rate]
                with open(record_filename, "ab") as f:
                    pickle.dump(data, f)
                break
            dqn_solver.experience_replay(sess)