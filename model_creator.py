import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from GameDataset import GameDataset
from ActorCritic import ActorCritic

df = pd.read_csv('combined.csv')
max_time_elapsed = df['time_elapsed'].max()
df['time_elapsed'] = df['time_elapsed'] / max_time_elapsed
df['current_difficulty'] = df['current_difficulty'] / 2
states = torch.tensor(df[['health', 'health_to_time', 'time_elapsed', 'damaged_last_wave', 'avg_wave_damage', 'factor_difference', 'current_difficulty']].values, dtype=torch.float32)
actions = torch.tensor(df['agent_action'].values, dtype=torch.float32)
rewards = torch.tensor(df['agent_reward'].values, dtype=torch.float32)
dataset = GameDataset(states, actions, rewards)
value_loss_fn = nn.MSELoss()
policy_loss_fn = nn.MSELoss()
num_epochs = 250
lr = 0.001
gamma = 0.99
batch_size = 32
softplus = nn.Softplus()


def train(model, dataset, epochs, batch_size, lr):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (states, actions, rewards) in enumerate(dataloader):
            optimizer.zero_grad()

            action_preds, value_preds = model(states)

            policy_dist = torch.distributions.normal.Normal(action_preds, 0.1)
            log_probs = policy_dist.log_prob(actions)
            softplus_actions = softplus(actions)
            log_softplus_actions = torch.log(softplus_actions + 1e-8)
            actor_loss = -torch.mean(log_probs - log_softplus_actions)

            critic_loss = nn.MSELoss()(value_preds.squeeze(), rewards)
            loss = critic_loss + actor_loss

            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    torch.save(model, 'trained_model_1.pt')


model = ActorCritic(input_size=7, hidden_size=15, output_size=1)
train(model, dataset, num_epochs, batch_size, lr)

