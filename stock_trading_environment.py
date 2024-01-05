import numpy as np

class StockTradingEnv:
    def __init__(self, data, initial_balance=1000, max_buy=5, max_sell=5):
        self.data = data
        self.initial_balance = initial_balance
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.reset()

        # Define action and observation space attributes
        self.action_space_n = 3  # Buy, Sell, Hold
        self.observation_space_shape = data.shape[1]  # Number of features in the data

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.current_price = self.data.iloc[self.current_step]['Close']
        return self._next_observation()

    def _next_observation(self):
        frame = self.data.iloc[self.current_step]
        # Include all relevant features in the state
        obs = np.array([frame['Open'], frame['High'], frame['Low'], frame['Close'],
                        frame['SMA'], frame['STD'], frame['Upper_Band'], frame['Lower_Band'],
                        self.balance, self.shares_held])  # Example - adjust as needed
        return obs

    def step(self, action):
        self.current_price = self.data.iloc[self.current_step]['Close']
        reward = 0
        done = False

        if action == 0:  # Buy
            reward = self._buy()
        elif action == 1:  # Sell
            reward = self._sell()

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        obs = self._next_observation()
        return obs, reward, done, {}

    def _buy(self):
        if self.total_shares_bought < self.max_buy and self.balance >= self.current_price:
            self.shares_held += 1
            self.total_shares_bought += 1
            self.balance -= self.current_price
            return -self.current_price  # cost of buying
        return 0

    def _sell(self):
        if self.shares_held > 0 and self.total_shares_sold < self.max_sell:
            self.shares_held -= 1
            self.total_shares_sold += 1
            self.balance += self.current_price
            return self.current_price  # profit from selling
        return 0
