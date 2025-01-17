import matplotlib.pyplot as plt

# plot action noise

def plot_logs(episode_rewards, success_rate, training_time, loss_actor, loss_critic, update_freq):
    episodes = list(range(1, len(episode_rewards) + 1))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(episodes, episode_rewards, label='Total Reward per Episode', color='b')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(training_time, success_rate, label='Success rate per training time', color='b')
    plt.xlabel('training time')
    plt.ylabel('Success rate')
    plt.title('Success rate per training time')
    plt.legend()
    plt.grid(True)

    epochs = list(range(update_freq, (len(loss_critic)* update_freq) + 1, update_freq))
    # Plot Actor Loss in the top subplot
    plt.subplot(2, 2, 2)  # 2 rows, 1 column, first subplot
    plt.plot(epochs,loss_actor, label='Actor Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Actor Loss')
    plt.legend()
    plt.grid(True)

    # Plot Critic Loss in the bottom subplot
    plt.subplot(2, 2, 4)  # 2 rows, 1 column, second subplot
    plt.plot(epochs,loss_critic, label='Critic Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Critic Loss')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_test(rewards):
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Total Reward per Episode', color='b')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    plt.show()