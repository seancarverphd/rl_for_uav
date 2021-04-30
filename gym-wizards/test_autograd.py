import torch
import one_step_ac_train

HIDDEN_SIZE = 48

class Critic(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Critic, self).__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(obs_size, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, 1),
        )
    def forward(self, inputs):
        critic_out = self.out(inputs.unsqueeze(dim=1))
        return critic_out

critic = Critic(2, 9)
for p in critic.parameters():
    p.requires_grad = True
Q = critic(torch.tensor([[2., 3.]]))
Q.retain_grad()
G = torch.ones(Q.shape)
Q.backward(gradient=G)
print(Q.grad)


