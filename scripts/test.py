from utils import *
import flappy_bird_gymnasium
import gymnasium
import torch
from cnn_model import NeuralNetwork


def test(model, episodes=100):
    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False)
    highest_score = 0

    for episode in range(episodes):
        obs, _ = env.reset()
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1

        image_data = env.render()
        _, _, terminal, _, info = env.step(0)
        image_data = resize_and_bgr2gray(image_data)
        image_data = image_to_tensor(image_data)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        score = 0

        while not terminal:
            output = model(state)[0]

            action = torch.zeros([model.number_of_actions], dtype=torch.float32)
            if torch.cuda.is_available():
                action = action.cuda()

            action_index = torch.argmax(output)
            if torch.cuda.is_available():
                action_index = action_index.cuda()
            action[action_index] = 1

            new_image_data = env.render()
            obs, reward, terminal, _, info = env.step(action_index.item())
            score = info["score"]

            new_image_data = resize_and_bgr2gray(new_image_data)
            new_image_data = image_to_tensor(new_image_data)
            new_state = torch.cat((state.squeeze(0)[1:, :, :], new_image_data)).unsqueeze(0)

            state = new_state

        if score > highest_score:
            highest_score = score

        print(f"Episode {episode + 1}/{episodes}: Score = {score}")

    print(f"\nHighest Score after {episodes} episodes: {highest_score}")


def main():
    cuda_is_available = torch.cuda.is_available()
    model = torch.load(
        'pretrained_model/new_current_model_7000000.pth',
        map_location='cpu' if not cuda_is_available else None
    ).eval()

    if cuda_is_available:
        model = model.cuda()

    test(model, episodes=100)


if __name__ == '__main__':
    main()