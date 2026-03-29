
import numpy as np
import matplotlib.pyplot as plt

class qLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space)
        else:
            q_values = [self.get_q_value(state, a) for a in self.action_space]
            max_q = max(q_values)
            best_actions = [a for a in self.action_space if self.get_q_value(state, a) == max_q]
            return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state):
        best_next_q = max([self.get_q_value(next_state, a) for a in self.action_space])
        new_q = (1 - self.learning_rate) * self.get_q_value(state, action) + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[(state, action)] = new_q
    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
    def plot_q_values(self):
        states = set(state for state, action in self.q_table.keys())
        actions = set(action for state, action in self.q_table.keys())
        q_matrix = np.zeros((len(states), len(actions)))
        
        state_to_index = {state: idx for idx, state in enumerate(states)}
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        
        for (state, action), q_value in self.q_table.items():
            s_idx = state_to_index[state]
            a_idx = action_to_index[action]
            q_matrix[s_idx, a_idx] = q_value
        
        plt.imshow(q_matrix, cmap='viridis')
        plt.colorbar(label='Q-value')
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.title('Q-values Heatmap')
        plt.show()

            