
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
        # 1. SORT the states and actions (Crucial for consistency!)
        # Using a set() scrambles the order every time. Sorting ensures 
        # the rows and columns stay in the same place across different runs.
        states = sorted(list(set(state for state, action in self.q_table.keys())))
        actions = sorted(list(set(action for state, action in self.q_table.keys())))
        
        # Safety check in case you call it before training
        if not states or not actions:
            print("Q-table is empty. Nothing to plot.")
            return

        q_matrix = np.zeros((len(states), len(actions)))
        
        state_to_index = {state: idx for idx, state in enumerate(states)}
        action_to_index = {action: idx for idx, action in enumerate(actions)}
        
        for (state, action), q_value in self.q_table.items():
            s_idx = state_to_index[state]
            a_idx = action_to_index[action]
            q_matrix[s_idx, a_idx] = q_value
        
        # 2. Make the figure larger and higher quality
        plt.figure(figsize=(10, 8), dpi=100)
        
        # 'aspect=auto' prevents the plot from looking squished if you have 
        # way more states than actions (which is common in traffic RL)
        im = plt.imshow(q_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        
        # Format the colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Expected Future Reward (Q-Value)', rotation=270, labelpad=20, fontsize=12)
        
        # 3. Clean up the titles and labels
        plt.xlabel('Traffic Light Actions', fontsize=12, fontweight='bold', labelpad=10)
        plt.ylabel('Discretized Traffic States', fontsize=12, fontweight='bold', labelpad=10)
        plt.title('Agent Learning: Q-Table Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        # 4. Handle tick marks smartly
        plt.xticks(ticks=np.arange(len(actions)), labels=actions)
        
        # If the state space grows massive, plotting 500 y-ticks becomes a black smear.
        # This hides them if there are too many, keeping it clean.
        if len(states) > 25:
            plt.yticks([]) 
            plt.ylabel(f'Discretized Traffic States ({len(states)} unique states)', fontsize=12, fontweight='bold')
        else:
            plt.yticks(ticks=np.arange(len(states)), labels=[str(s) for s in states], fontsize=8)
            
        # Ensures labels don't get cut off on the edges
        plt.tight_layout() 
        plt.show()