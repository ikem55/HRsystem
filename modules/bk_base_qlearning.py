import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

from copy import deepcopy

# https://www.tcom242242.net/entry/2019/08/01/%E3%80%90%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%81%E5%85%A5%E9%96%80%E3%80%91q%E5%AD%A6%E7%BF%92_%E8%BF%B7%E8%B7%AF%E3%82%92%E4%BE%8B%E3%81%AB/
# https://www.tcom242242.net/entry/2019/08/12/%e3%80%90%e6%b7%b1%e5%b1%a4%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92%e3%80%91deep_q_network_%e3%82%92tensorflow%e3%81%a7%e5%ae%9f%e8%a3%85/

Experience = namedtuple(
    'Experience', 'state0, action, reward, state1, terminal')

def sample_batch_indexes(low, high, size):
    r = range(low, high)
    batch_idxs = random.sample(r, size)

    return batch_idxs

class Memory:
    def __init__(self, limit, maxlen):
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.observations = deque(maxlen=limit)
        self.maxlen = maxlen
        self.recent_observations = deque(maxlen=maxlen)

    def sample(self, batch_size):
        batch_idxs = sample_batch_indexes(
            0, len(self.observations) - 1, size=batch_size)
        for (i, idx) in enumerate(batch_idxs):
            terminal = self.terminals[idx-1]
            while terminal:
                idx = sample_batch_indexes(
                    0, len(self.observations)-1, size=1)[0]
                terminal = self.terminals[idx-1]
            batch_idxs[i] = idx

        experiences = []
        for idx in batch_idxs:
            state0 = self.observations[idx]
            action = self.actions[idx]
            reward = self.rewards[idx]
            terminal = self.terminals[idx]
            state1 = self.observations[idx+1]
            experiences.append(Experience(state0=state0,
                                          action=action,
                                          reward=reward,
                                          state1=state1,
                                          terminal=terminal))

        return experiences

    def append(self, observation, action, reward, terminal=False):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        self.recent_observations.append(observation)


class EpsGreedyQPolicy:
    def __init__(self, eps=.1, eps_decay_rate=.99, min_eps=.1):
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate
        self.min_eps = min_eps

    def select_action(self, q_values, is_training=True):
        nb_actions = q_values.shape[0]

        if is_training:
            if np.random.uniform() < self.eps:
                action = np.random.randint(0, nb_actions)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)

        return action

    def decay_eps_rate(self):
        self.eps = self.eps*self.eps_decay_rate
        if self.eps < self.min_eps:
            self.eps = self.min_eps

def build_model(input_shape, nb_output):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(nb_output, activation="linear")(x)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model

class DQNAgent():
    """
        Deep Q Network Agent
    """

    def __init__(self,
                 training=True,
                 policy=None,
                 gamma=.99,
                 learning_rate=.001,
                 actions=None,
                 memory=None,
                 memory_interval=1,
                 update_interval=100,
                 train_interval=1,
                 batch_size=32,
                 nb_steps_warmup=200,
                 observation=None,
                 input_shape=None,
                 obs_processer=None):

        self.training = training
        self.policy = policy
        self.actions = actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.obs_processer = obs_processer
        self.recent_action_id = None
        self.recent_observation = self.obs_processer(observation)
        self.previous_observation = None
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.model = build_model(input_shape, len(self.actions))
        self.target_model = build_model(input_shape, len(self.actions))
        self.nb_actions = len(self.actions)
        self.train_interval = train_interval
        self.step = 0
        self.trainable_model = None
        self.update_interval = update_interval

    def compile(self):
        mask = tf.keras.layers.Input(name="mask", shape=(self.nb_actions, ))
        output = tf.keras.layers.multiply([self.model.output, mask])
        trainable_model = tf.keras.models.Model(
            inputs=[self.model.input, mask],
            outputs=[output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        trainable_model.compile(loss="mse", optimizer=optimizer)
        self.trainable_model = trainable_model

    def act(self):
        action_id = self.forward()
        action = self.actions[action_id]
        return action

    def forward(self):
        q_values = self.compute_q_values(self.recent_observation)
        action_id = self.policy.select_action(
            q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id

        return action_id

    def observe(self, observation, reward=None, is_terminal=None):
        self.previous_observation = copy.deepcopy(self.recent_observation)
        self.recent_observation = self.obs_processer(observation)

        if self.training and reward is not None:
            if self.step % self.memory_interval == 0:
                self.memory.append(self.previous_observation,
                                   self.recent_action_id,
                                   reward,
                                   terminal=is_terminal)
            self.experience_replay()
            self.policy.decay_eps_rate()

        self.step += 1

    def experience_replay(self):
        if self.step > self.nb_steps_warmup \
                and self.step % self.train_interval == 0:

            experiences = self.memory.sample(self.batch_size)

            state0_batch = []
            reward_batch = []
            action_batch = []
            state1_batch = []
            terminal_batch = []

            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            reward_batch = np.array(reward_batch)
            target_q_values = self.predict_on_batch(state1_batch)
            target_q_values = np.max(target_q_values, axis=1)
            discounted_reward_batch = (self.gamma * target_q_values)
            discounted_reward_batch *= terminal_batch

            targets = reward_batch + discounted_reward_batch
            mask = np.zeros((len(action_batch), len(self.actions)))
            target_batch = np.zeros((self.batch_size, len(self.actions)))
            for idx, (action, target) in enumerate(zip(action_batch, targets)):
                target_batch[idx][action] = target
                mask[idx][action] = 1.

            self.train_on_batch(state0_batch,
                                mask,
                                target_batch)

        if self.update_interval > 1:
            # hard update
            self.update_target_model_hard()
        else:
            # soft update
            self.update_target_model_soft()

    def train_on_batch(self, state_batch, mask, targets):
        state_batch = np.array(state_batch)
        self.trainable_model.train_on_batch([state_batch, mask],
                                            [targets])

    def predict_on_batch(self, state1_batch):
        state1_batch = np.array(state1_batch)
        q_values = self.target_model.predict(state1_batch)
        return q_values

    def compute_q_values(self, state):
        q_values = self.target_model.predict(np.array([state]))
        return q_values[0]

    def update_target_model_hard(self):
        """ for hard update """
        if self.step % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())

    def update_target_model_soft(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self):
        self.recent_observation = None
        self.previous_observation = None
        self.recent_action_id = None

class QLearningAgent:
    """
        Q学習
    """
    def __init__(self, alpha=.2, epsilon=.1, gamma=.99, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_q_values()

    def _init_q_values(self):
        """
           Q テーブルの初期化
        """
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values

    def init_state(self):
        """
            状態の初期化
        """
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state

    def act(self):
        """ 行動の選択。購入するかしないか。 """
        # ε-greedy選択
        if np.random.uniform() < self.epsilon:  # random行動
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:   # greedy 行動
            action = np.argmax(self.q_values[self.state])

        self.previous_action = action
        return action

    def observe(self, next_state, reward=None):
        """
            次の状態と報酬の観測
        """
        next_state = str(next_state)
        if next_state not in self.q_values: # 始めて訪れる状態であれば
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))

        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)

    def learn(self, reward):
        """
            Q値の更新
        """
        q = self.q_values[self.previous_state][self.previous_action] # Q(s, a)
        max_q = max(self.q_values[self.state]) # max Q(s')
        # Q(s, a) = Q(s, a) + alpha*(r+gamma*maxQ(s')-Q(s, a))
        self.q_values[self.previous_state][self.previous_action] = q + (self.alpha * (reward + (self.gamma*max_q) - q))

class KeibaWorld:
    """ 競馬の世界をモデル化 """
    def __init__(self, start_pos=10000):
        self.actions = {
            "STAY":0,
            "1BUY":1,
            "2BUY":2,
            "3BUY":3
        }
        self.start_pos = start_pos #スタート位置（最初の残高）
        self.agent_pos = copy.deepcopy(self.start_pos) #現在の残高


    def step(self, action, res):
        """ 行動の実行。状態、報酬を返却 """
        to_pos = copy.deepcopy(self.agent_pos)
        # 払戻を計算
        if action == 0:
            temp_return = 0
        else:
            temp_return = (-100 + res) * action
        self.agent_pos = to_pos + temp_return

        # 報酬を計算
        reward = self.agent_pos
#        reward = self._compute_reward(action, res)
        is_end = self._is_end_episode() # エピソードの終了の確認
        return self.agent_pos, reward, is_end

    def _compute_reward(self, action, res):
        temp_return = 0
        if action == 1:
            temp_return = res - 100
        elif action == 0:
            temp_return = - 10
        return temp_return

    def _is_end_episode(self):
        if self.agent_pos <= 0:
            return True
        else:
            return False

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos


class GridWorld:

    def __init__(self):

        self.filed_type = {
                "N": 0,  # 通常
                "G": 1,  # ゴール
                "W": 2,  # 壁
                "T": 3,  # トラップ
                }

        self.actions = {
            "UP": 0,
            "DOWN": 1,
            "LEFT": 2,
            "RIGHT": 3
            }

        self.map = [[3, 2, 0, 1],
                    [0, 0, 0, 2],
                    [0, 0, 2, 0],
                    [2, 0, 2, 0],
                    [0, 0, 0, 0]]


        self.start_pos = 0, 4   # エージェントのスタート地点(x, y)
        self.agent_pos = copy.deepcopy(self.start_pos)  # エージェントがいる地点


    def step(self, action):
        """
            行動の実行
            状態, 報酬、ゴールしたかを返却
        """
        to_x, to_y = copy.deepcopy(self.agent_pos)

        # 移動可能かどうかの確認。移動不可能であれば、ポジションはそのままにマイナス報酬
        if self._is_possible_action(to_x, to_y, action) == False:
            return self.agent_pos, -1, False

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        is_goal = self._is_end_episode(to_x, to_y) # エピソードの終了の確認
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = to_x, to_y
        return self.agent_pos, reward, is_goal

    def _is_end_episode(self, x, y):
        """
            x, yがエピソードの終了かの確認。
        """
        if self.map[y][x] == self.filed_type["G"]:      # ゴール
            return True
        elif self.map[y][x] == self.filed_type["T"]:    # トラップ
            return True
        else:
            return False

    def _is_wall(self, x, y):
        """
            x, yが壁かどうかの確認
        """
        if self.map[y][x] == self.filed_type["W"]:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        """
            実行可能な行動かどうかの判定
        """
        to_x = x
        to_y = y

        if action == self.actions["UP"]:
            to_y += -1
        elif action == self.actions["DOWN"]:
            to_y += 1
        elif action == self.actions["LEFT"]:
            to_x += -1
        elif action == self.actions["RIGHT"]:
            to_x += 1

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False
        elif self._is_wall(to_x, to_y):
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == self.filed_type["N"]:
            return 0
        elif self.map[y][x] == self.filed_type["G"]:
            return 100
        elif self.map[y][x] == self.filed_type["T"]:
            return -100

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos


# ゴール：回収率が100%を超える
# 前提：残高が０になるとゲームオーバー
from modules.lb_extract import LBExtract

class RaceData:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.ext = LBExtract(start_date, end_date, False)
        self.set_raceuma_df()

    def set_raceuma_df(self):
        self.raceuma_df = self.ext.get_raceuma_table_base()

    def get_learning_df(self):
        learning_df = self.raceuma_df[["競走コード", "馬番", "予想オッズ", "予想タイム指数", "デフォルト得点", "得点V1", "得点V2", "得点V3", "投票直前単勝オッズ", "予想展開"]]
        return learning_df

    def get_result_df(self):
        result_df = self.raceuma_df[["競走コード", "馬番", "単勝配当"]]
        return result_df

# http://mokichi-blog.com/2019/03/03/%e3%80%90openai-gym%e3%80%91%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92%e3%81%a7rpg%e3%81%ae%e3%83%9c%e3%82%b9%e3%82%92%e5%80%92%e3%81%99/
# GYMについて
class HRgym(gym.Env):
    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_rangeを設定する
        self.action_space = gym.spaces.Discrete(3) #行動の数
        self.ovservation_space = gym.spaces.Box()

def run_q_learning():
    grid_env = KeibaWorld() # grid worldの環境の初期化
    ini_state = grid_env.start_pos  # 初期状態（エージェントのスタート地点の位置）
    agent = QLearningAgent(epsilon=.1, actions=np.arange(4), observation=ini_state)  # Q学習エージェント
    nb_episode = 200   #エピソード数
    rewards = []    # 評価用報酬の保存
    is_end_episode = False # エージェントがゴールしてるかどうか？
    start_date = '2018/01/01'
    end_date = '2018/01/03'
    rd = RaceData(start_date, end_date)
    learning_df = rd.get_learning_df()
    result_df = rd.get_result_df()
    print(learning_df.head())
    print(result_df.head())
    for episode in range(nb_episode):
        print(episode)
        episode_reward = [] # 1エピソードの累積報酬
        for index, row in learning_df.iterrows():
            sr = row.drop(["競走コード", "馬番"])
            res = result_df.iloc[index]["単勝配当"]
#            if index == 1:
#                print(sr)
#                print(res)
            action = agent.act() #行動選択
            state, reward, is_end_episode = grid_env.step(action, res)
            if is_end_episode:
                break
            agent.observe(state, reward)
            episode_reward.append(reward)
        rewards.append(np.sum(episode_reward)) # このエピソードの平均報酬を与える
        if not is_end_episode:
            print("回避！！")
            print(reward)
        state = grid_env.reset()    #  初期化
        agent.observe(state)    # エージェントを初期位置に
        is_end_episode = False

    # 結果のプロット
    plt.plot(np.arange(nb_episode), rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("result.png")
    plt.show()

def obs_processer(row_obs):
    return row_obs

def run_dqn_learning():
    env = gym.make('CartPole-v0')
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n
    actions = np.arange(nb_actions)
    policy = EpsGreedyQPolicy(eps=1., eps_decay_rate=.999, min_eps=.01)
    memory = Memory(limit=50000, maxlen=1)
    obs = env.reset()
    agent = DQNAgent(actions=actions,
                     memory=memory,
                     update_interval=200,
                     train_interval=1,
                     batch_size=32,
                     observation=obs,
                     input_shape=[len(obs)],
                     policy=policy,
                     obs_processer=obs_processer)

    agent.compile()

    result = []
    nb_epsiodes = 1000
    for episode in range(nb_epsiodes):
        agent.reset()
        observation = env.reset()
        observation = deepcopy(observation)
        agent.observe(observation)
        done = False
        while not done:
            action = deepcopy(agent.act())
            observation, reward, done, info = env.step(action)
            observation = deepcopy(observation)
            agent.observe(observation, reward, done)
            if done:
                break

        agent.training = False
        observation = env.reset()
        agent.observe(observation)
        done = False
        step = 0
        while not done:
            # env.render() # 表示
            step += 1
            action = agent.act()
            observation, reward, done, info = env.step(action)
            agent.observe(observation)
            if done:
                print("Episode {}: {} steps".format(episode, step))
                result.append(step)
                break

        agent.training = True

    x = np.arange(len(result))
    plt.ylabel("time")
    plt.xlabel("episode")
    plt.plot(x, result)
    plt.savefig("result.png")

if __name__ == '__main__':
    # run_q_learning()
    run_dqn_learning()