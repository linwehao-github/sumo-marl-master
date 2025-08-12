import os
import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd  # 新增：用于处理CSV输出
from datetime import datetime  # 新增：用于时间戳

from env import TrafficEnv
from maddpg import MADDPG
from utils import get_average_travel_time

parser = argparse.ArgumentParser()
parser.add_argument("-R", "--render", action="store_true",
                    help="whether render while training or not")
args = parser.parse_args()

if __name__ == "__main__":
    # Before the start, should check SUMO_HOME is in your environment variables
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    
    env = TrafficEnv("gui") 
    # configuration
    state_dim = env.state_dim
    action_dim = env.n_phase
    n_agents = env.n_intersections
    n_episode = 1

    # Create an Environment and RL Agent
    agent = MADDPG(n_agents, state_dim, action_dim)

    # Train your RL agent
    performance_list = []
    reward_list = []  # 用于存储每轮的总奖励
    step_rewards_data = []  # 新增：用于存储每一步的详细奖励数据

    # 新增：创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 新增：准备CSV文件头
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"results/step_rewards_{timestamp}.csv"

    for episode in range(n_episode):
        state = env.reset()
        reward_epi = []
        actions = [None for _ in range(n_agents)]
        action_probs = [None for _ in range(n_agents)]
        done = False
        step_count = 0  # 新增：记录当前episode的步数

        while not done:
            # select action according to a given state
            for i in range(n_agents):
                action, action_prob = agent.select_action(state[i, :], i)
                actions[i] = action
                action_probs[i] = action_prob

            # apply action and get next state and reward
            before_state = state
            state, reward, done = env.step(actions)
            reward_epi.append(reward)  # 收集每一步的奖励

            # 新增：记录每一步的详细数据
            step_data = {
                'episode': episode + 1,
                'step': step_count + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_reward': sum(reward),
                **{f'agent_{i}_reward': reward[i] for i in range(n_agents)},
                'done': done
            }
            step_rewards_data.append(step_data)

            # make a transition and save to replay memory
            transition = [before_state, action_probs, state, reward, done]
            agent.push(transition)

            # train an agent
            if agent.train_start():
                for i in range(n_agents):
                    agent.train_model(i)
                agent.update_eps()

            step_count += 1
            if done:
                break

        env.close()
        average_traveling_time = get_average_travel_time(episode)
        performance_list.append(average_traveling_time)
        
        # 计算本轮的总奖励
        total_reward = sum([sum(r) for r in reward_epi])
        reward_list.append(total_reward)

        print(f"Episode: {episode+1}\t Steps: {step_count}\t Average Traveling Time:{average_traveling_time}\t Total Reward:{total_reward}\t Eps:{agent.eps}")
    
    # Save performance list and reward list to txt file
    with open("results/performance.txt", "w") as f:
        for i, (perf, reward) in enumerate(zip(performance_list, reward_list)):
            f.write(f"Episode {i+1}: Average Traveling Time = {perf}, Total Reward = {reward}\n")
    
    # 新增：保存每一步的奖励数据到CSV
    step_rewards_df = pd.DataFrame(step_rewards_data)
    step_rewards_df.to_csv(csv_filename, index=False)
    print(f"\nStep rewards saved to {csv_filename}")

    # Save the model
    agent.save_model("results/trained_model.th")