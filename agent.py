import tensorflow as tf
import numpy as np
from tensorflow import keras

class DDDQN(tf.keras.Model):
    def __init__(self, state_shape, action_size):
        super(DDDQN, self).__init__()
        self.state_shape = state_shape  # Store state_shape as an instance variable
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(action_size, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, state):
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a

    def build_model(self, input_shape):
        # Build the model by calling it with a sample input
        sample_input = tf.random.normal(shape=(1, *input_shape))
        self.call(sample_input)

    def ensure_built(self, state_shape):
        if not self.built:
            sample_input = tf.random.normal(shape=(1, *state_shape))
            self(sample_input)

class exp_replay():
    def __init__(self, state_shape, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_mem = np.zeros(self.buffer_size, dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
        self.done_mem = np.zeros(self.buffer_size, dtype=np.bool_)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx  = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch_size = min(batch_size, max_mem)  # Ensure batch size does not exceed the number of samples
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones

class agent():
    def __init__(self, state_shape, action_size, gamma=0.99, replace=100, lr=0.001):
        self.state_shape = state_shape
        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1e-3
        self.replace = replace
        self.trainstep = 0
        self.memory = exp_replay(state_shape, action_size)  # Pass the correct shapes
        self.batch_size = 64
        self.q_net = DDDQN(state_shape, action_size)
        self.target_net = DDDQN(state_shape, action_size)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.q_net.a.units)  # use self.q_net.a.units for action size
        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
            return action



    def update_mem(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)


    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon


    def train(self):
        # Check if enough samples are available
        if self.memory.pointer < self.batch_size:
            return

        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
        actual_batch_size = len(actions)  # Determine the actual number of samples returned

        # Skip training if there aren't enough samples
        if actual_batch_size < self.batch_size:
            return

        # Update target network if needed
        if self.trainstep % self.replace == 0:
            self.update_target()

        # Predict Q-values for current states and next states
        target = self.q_net.predict(states)
        next_state_val = self.target_net.predict(next_states)

        # Find the max Q-value action for each next state
        max_action = np.argmax(self.q_net.predict(next_states), axis=1)

        # Ensure batch_index aligns with the actual batch size
        batch_index = np.arange(actual_batch_size, dtype=np.int32)

        # Debugging: Print array shapes
        print("Batch Index Shape:", batch_index.shape)
        print("Actions Shape:", actions.shape)
        print("Rewards Shape:", rewards.shape)
        print("Dones Shape:", dones.shape)
        print("Max Action Shape:", max_action.shape)

        # Update Q values
        q_target = np.copy(target)
        q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action] * dones

        # Train the Q-network
        self.q_net.train_on_batch(states, q_target)

        # Update epsilon and training step
        self.update_epsilon()
        self.trainstep += 1

    def save_model(self, model_name="model"):
        # Ensure the models are built and have undergone a forward pass
        self.q_net.ensure_built(self.state_shape)
        self.target_net.ensure_built(self.state_shape)

        self.q_net.save(f"{model_name}_q_net", save_format="tf")
        self.target_net.save(f"{model_name}_target_net", save_format="tf")

    def load_model(self, model_name="model"):
        self.q_net = tf.keras.models.load_model(f"{model_name}_q_net")
        self.target_net = tf.keras.models.load_model(f"{model_name}_target_net")
