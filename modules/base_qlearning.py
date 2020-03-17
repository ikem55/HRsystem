import sys
import gym
import numpy as np
import gym.spaces

# http://mokichi-blog.com/2019/03/03/%e3%80%90openai-gym%e3%80%91%e5%bc%b7%e5%8c%96%e5%ad%a6%e7%bf%92%e3%81%a7rpg%e3%81%ae%e3%83%9c%e3%82%b9%e3%82%92%e5%80%92%e3%81%99/
import random
import math
import io
import os
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class Race():
    # 対象レース群。レース情報と結果、払戻
    def __init__(self, raceuma_df, result_df):
        self.raceuma_df = raceuma_df
        self.result_df = result_df
        self.df_len = len(raceuma_df)

    def provide_race_info(self):
        idx = random.randint(0,self.df_len -1)
        sr = self.raceuma_df.iloc[idx]
        return idx, sr

    def provide_race_result(self, idx):
        sr = self.result_df.iloc[idx]
        return sr



class Player():
    # Bet者。所持金と行動を定義
    # 0：見
    # 1: 参加 100円かける
    # 2: 勝負 所持金のN%をかける
    # 3: 均等買い 所持金のN%が払い戻されるようにかける
    ZANDAKA = 20000
    RATE_2 = 1
    RATE_3 = 8
    def __init__(self):
        self.zandaka = self.ZANDAKA
        self.bet = 0
        self.res = 0

    def select_action(self, race_info):
        # モデルを作成
        # テストなのでとりあえずランダム
        self.action = np.random.randint(0, 4)
        self.bet = self._bet_money(race_info)

    def _bet_money(self, race_info):
        """ 掛け金を計算 """
        if self.action == 1:
            bet = 100
        elif self.action == 2:
            bet = math.ceil(self.zandaka / 10000 * self.RATE_2) * 100
        elif self.action == 3:
            odds = race_info["予想オッズ"]
            if odds ==0:
                print("---check !")
                bet = 100
            else:
                bet = math.ceil(self.zandaka * (self.RATE_3 / 10000) / odds) * 100
        else:
            bet = 0
        if bet <= 100:
            bet = 0
        return bet

    def update_zandaka(self):
        """ 残高を更新 """
        self.zandaka = self.zandaka - self.bet + self.res

class RaceGame(gym.Env):
    def __init__(self, start_date, end_date):
        super().__init__()
        # action_space（Agentの行動の選択肢：出力層）, observation_space（Agentが観測できる環境のパラメタ：入力層）, reward_range（報酬の最小値から最大値の範囲）を設定する
        self.start_date = start_date
        self.end_date = end_date
        #
        self.action_space = gym.spaces.Discrete(3) #行動の数
        self.observation_space = gym.spaces.Box(
            low=0., #最小値
            high=100., # 最大値
            shape=(2,) # 観測値（例、HP,MP）
        )
        self.reward_range = [-500000., 500000.] # WIN or LOSE
        self._get_base_df()
        self.reset()

    def _get_base_df(self):
        rd = RaceData(self.start_date, self.end_date)
        self.learning_df = rd.get_learning_df()
        self.result_df = rd.get_result_df()
        print(self.learning_df.shape)
        print(self.result_df.shape)

    def reset(self):
        """ 環境を初期化 """
        self.Race = Race(self.learning_df, self.result_df)
        self.Player = Player()
        self.done = False # 所持金が０
        self.steps = 0
        return self._observe()

    def step(self, action):
        """ Agentの行動（Action)に対する結果（Observation, reward, done, info)を返す。１ステップ進める処理（RaceのBet）を記述 """
        # レース情報を受け取る
        idx, race_info = self.Race.provide_race_info()
        # 取得した情報から行動を選択する
        self.Player.select_action(race_info)
        # actionが０（見）以外は答え合わせ
        self.Player.res = 0
        if self.Player.action != 0:
            self.Player.res = self.Race.provide_race_result(idx)["単勝配当"]
            self.Player.update_zandaka()

        self.steps += 1

        observation = self._observe()
        reward = self._get_reward()
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def render(self, mode='human', close=False):
        """ humanの場合はコンソールに出力、ansiの場合は StringIOを返す """
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(" log - action:" + str(self.Player.action) + " zandaka: " + str(self.Player.zandaka) + " bet: " + str(self.Player.bet) + " res: " + str(self.Player.res) +'\n')
        return outfile

    def _close(self):
        pass

    def _seed(selfself, seed=None):
        pass

    def _get_reward(self):
        """ 報酬を返す
        zandakaが０になったら - 10000
        ステップ終了まで残高が０にならなかったら5000
        的中したら + 20
        はずれたら -5
        見したら -1
        """
        if self.Player.zandaka <= 0:
            return - 10000
        elif self.steps >= self.Race.df_len:
            return 5000
        else:
            if self.Player.action == 0:
                return -1
            else:
                if self.Player.res > 0:
                    return 20
                else:
                    return -5

    def _observe(self):
        observation = np.array([self.Player.zandaka, self.Player.bet],)
        return observation

    def _is_done(self):
        """ 残高が尽きたかの判定 """
        if self.Player.zandaka <= 0:
            return True
        elif self.steps >=  self.Race.df_len:
            return True
        else:
            return False

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

ENV_NAME = 'tansho_v1'
MODEL_DIR = '../for_test_model/Q_base/' + ENV_NAME


class AgentPlayer():
    def __init__(self,start_date, end_date, filename):
        #env = gym.make(ENV_NAME)
        self.env = RaceGame(start_date, end_date)
        self.filename = filename
        np.random.seed(123)
        self.env.seed(123)
        self.nb_actions = self.env.action_space.n
        self._set_model()

    def _set_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        model.add(Reshape((2, 1), input_shape=(2, )))
        model.add(LSTM(50, input_shape=(2, 1),
                  return_sequences=False,
                  dropout=0.0))
        model.add(Dense(self.nb_actions))
        model.add(Activation('linear'))

        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        self.dqn = DQNAgent(model=model, nb_actions=self.nb_actions, memory=memory, nb_steps_warmup=10,
                       target_model_update=1e-2, policy=policy)
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def load_model(self):
        if os.path.exists(self.filename):
            print("load_model")
            self.dqn.load_weights(self.filename)

    def train_model(self):
        # Okay, now it's time to learn something! We visualize the training here for show, but this
        # slows down training quite a lot. You can always safely abort the training prematurely using
        # Ctrl + C.
        history = self.dqn.fit(self.env, nb_steps=3, visualize=False, verbose=2)
        print(history.history)

        # 結果を表示
        plt.subplot(2, 1, 1)
        plt.plot(history.history["nb_episode_steps"])
        plt.ylabel("step")

        plt.subplot(2, 1, 2)
        plt.plot(history.history["episode_reward"])
        plt.xlabel("episode")
        plt.ylabel("reward")

        plt.show()

        # After training is done, we save the final weights.
        self.dqn.save_weights(self.filename, overwrite=True)

    def test_model(self):
        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=2, visualize=False)

    def return_action(self):
        state0 = self.dqn.recent_observation
        q_values = self.dqn.predict(np.asarray([state0]), batch_size=1)
        action = np.argmax(q_values)
        return action

print("モデルを生成、トレーニング")

start_date = '2018/01/01'
end_date = '2018/01/03'
filename = MODEL_DIR + '/dqn_{}_weights.h5f'.format(ENV_NAME)
train_agent = AgentPlayer(start_date, end_date, filename)
train_agent.load_model()
train_agent.train_model()
train_agent.test_model()

print("生成したモデルで別日でテスト")
test_start_date = '2018/02/01'
test_end_date = '2018/02/03'
train_agent = AgentPlayer(test_start_date, test_end_date, filename)
train_agent.load_model()
train_agent.test_model()

print("１レースごとに結果を返すテスト")

