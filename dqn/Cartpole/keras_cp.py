import random
import gym
import numpy as np
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_NAME = "CartPole-v1"
record_filename = './record.dat'

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

MAX_EPISODES = 1000


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        state = np.float32([arr[0] for arr in batch])
        action = np.int32([arr[1] for arr in batch])
        reward = np.float32([arr[2] for arr in batch])
        state_next = np.float32([arr[3] for arr in batch])
        terminal = np.float32([arr[4] for arr in batch])

        qvalues = self.model.predict(state_next)
        qvalues = np.float32([np.amax(qvalue) for qvalue in qvalues])
        q_update = reward
        for i in range(BATCH_SIZE):
            if not terminal[i]:
                q_update[i] = (reward[i] + GAMMA * qvalues[i])
        q_values = self.model.predict(state)

        for i in range(BATCH_SIZE):
            q_values[i][action[i]] = q_update[i]

        self.model.fit(state, q_values, verbose=0,epochs=1)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


env = gym.make(ENV_NAME)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
dqn_solver = DQNSolver(observation_space, action_space)
run = 0
for i in range(1,MAX_EPISODES+1):
    run += 1
    state = env.reset()
    step = 0
    while True:
        step += 1
        env.render()
        state = np.reshape(state, [1, observation_space])
        action = dqn_solver.act(state)
        state_next, reward, terminal, info = env.step(action)
        reward = reward if not terminal else -reward
        state = np.reshape(state, [observation_space])
        dqn_solver.remember(state, action, reward, state_next, terminal)
        state = state_next
        if terminal:
            print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
            data = [run,step,dqn_solver.exploration_rate]
            with open(record_filename, "ab") as f:
                pickle.dump(data, f)
            break
        dqn_solver.experience_replay()
