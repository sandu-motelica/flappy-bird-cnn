import torch
import torch.optim as optim
import torch.nn as nn
import flappy_bird_gymnasium
import gymnasium
from utils import *
import random
import time
import os
from cnn_model import NeuralNetwork


def train(model, start):

    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    criterion = nn.MSELoss()

    # make env and start it
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    obs, _ = env.reset()

    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1

    image_data = env.render()
    _, reward, terminal, _, _ = env.step(0)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    # epsilon = 0.0001
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main loop
    while True:

        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        new_image_data = env.render()
        if action[0]:
            _, reward, terminal, _, _ = env.step(0)
        else:
            _, reward, terminal, _, _ = env.step(1)

        if reward == -0.5:
            reward = -1  # maximum penalization for hitting borders

        new_image_data = resize_and_bgr2gray(new_image_data)
        new_image_data = image_to_tensor(new_image_data)
        new_state = torch.cat((state.squeeze(0)[1:, :, :], new_image_data)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, new_state, terminal))

        # remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        next_output_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(next_output_batch[i])
                                  for i in range(len(minibatch))))

        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # reset gradients
        optimizer.zero_grad()

        # new tensor detached from the current graph
        y_batch = y_batch.detach()

        loss = criterion(q_value, y_batch)

        loss.backward()
        optimizer.step()

        state = new_state
        iteration += 1

        if terminal:
            env.reset()

        if iteration % 100000 == 0:
            torch.save(model, "pretrained_model/new_current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))


def main():
    cuda_is_available = torch.cuda.is_available()

    if not os.path.exists('pretrained_model/'):
        os.mkdir('pretrained_model/')

    model = NeuralNetwork()

    if cuda_is_available:
        model = model.cuda()

    model.apply(init_weights)
    start = time.time()

    train(model, start)


if __name__ == "__main__":
    main()