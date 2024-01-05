import os
from stock_trading_environment import StockTradingEnv
from stock_data_preprocessing import load_and_preprocess_data
from agent import agent
import argparse

# Create the parser and add argument
parser = argparse.ArgumentParser(description='Train a DDDQN agent for stock trading.')
parser.add_argument('epsilon', type=float, help='Target epsilon value for training termination')

# Parse the argument
args = parser.parse_args()
target_epsilon = args.epsilon

# Load and preprocess the data
data = load_and_preprocess_data('data/CADUSD=X.csv')

# Initialize the environment and agent
env = StockTradingEnv(data)
state_shape = (env.observation_space_shape,)
action_size = env.action_space_n
agent_instance = agent(state_shape, action_size)

# Define training loop parameters
log_interval = 100  # Log every 100 iterations
save_interval = 500  # Save model every 500 iterations

# Load model if it exists
model_name = "model"
q_net_model_dir = f"{model_name}_q_net"
target_net_model_dir = f"{model_name}_target_net"

# Check if both model directories exist
if os.path.isdir(q_net_model_dir) and os.path.isdir(target_net_model_dir):
    print("Loading existing models...")
    agent_instance.load_model(model_name)
else:
    print("Starting training from scratch...")

# Training loop
step = 0
while agent_instance.epsilon > target_epsilon:
    done = False
    state = env.reset()
    total_reward = 0
    while not done:
        action = agent_instance.act(state)
        next_state, reward, done, _ = env.step(action)
        agent_instance.update_mem(state, action, reward, next_state, done)

        if agent_instance.memory.pointer >= agent_instance.batch_size:
            agent_instance.train()

        state = next_state
        total_reward += reward

    # Logging
    if step % log_interval == 0:
        print(f"Step: {step}, Total Reward: {total_reward}, Epsilon: {agent_instance.epsilon}")

    # Save the model
    if step % save_interval == 0 and step != 0:
        agent_instance.save_model(model_name)

    step += 1

# Save the model at the end
agent_instance.save_model(model_name)
print("Training completed and model saved.")
