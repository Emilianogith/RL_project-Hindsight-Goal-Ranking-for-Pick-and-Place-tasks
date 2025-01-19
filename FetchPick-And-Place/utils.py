import matplotlib.pyplot as plt

# plot action noise

def plot_logs(episode_rewards, success_rate, training_time, loss_actor, loss_critic, update_freq, per_episodes_evaluation):
    episodes = list(range(1, len(episode_rewards) + 1))

    # Plot total reward
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(episodes, episode_rewards, label='Total Reward per Episode', color='b')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    # Plot Success rate 
    plt.subplot(2, 2, 3)
    plt.plot(training_time, success_rate, label='Success rate per training time', color='b')
    plt.xlabel('training time')
    plt.ylabel('Success rate')
    plt.title(f'Success rate per {per_episodes_evaluation} episodes')
    plt.legend()
    plt.grid(True)

    epochs = list(range(update_freq, (len(loss_critic)* update_freq) + 1, update_freq))
    # Plot Actor Loss 
    plt.subplot(2, 2, 2) 
    plt.plot(epochs,loss_actor, label='Actor Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Actor Loss')
    plt.legend()
    plt.grid(True)

    # Plot Critic Loss 
    plt.subplot(2, 2, 4)  
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
    """used to plot the total reward in some test experiments"""
    episodes = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label='Total Reward per Episode', color='b')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    plt.show()