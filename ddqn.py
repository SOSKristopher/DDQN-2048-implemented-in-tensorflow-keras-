import random
import numpy as np

from game2048.game import Game
from models import model_fn

from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *

import tensorflow as tf
from tensorflow.keras import backend as K

from collections import deque


def _huber_loss(y_true, y_pred, clip_delta=1.0):
	error = y_true - y_pred
	cond  = K.abs(error) <= clip_delta

	squared_loss = 0.5 * K.square(error)
	quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

	return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def one_hot_encoding(board, types):
	board = np.log2(board, where=board != 0)
	return to_categorical(board, types)


class DDQNAgent:
	def __init__(self, game, display=None):
		self.game = game
		self.display = display

		self.state_shape = (game.size, game.size, int(np.log2(game.score_to_win) + 1))
		self.action_size = 4

		self.memory = deque(maxlen=2000)
		self.model = self._build_model()
		self.target_model = self._build_model()

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.05
		self.epsilon_decay = 0.99
		self.learning_rate = 0.001

	def step(self):
		return self.act(self.game.board)

	def play(self, verbose=False):
		n_iter = 0
		while not self.game.end:
			direction = self.step()
			self.game.move(direction)
			n_iter += 1
			if verbose:
				print("Iter: {}".format(n_iter))
				print("======Direction: {}======".format(["left", "down", "right", "up"][direction]))
				if self.display is not None:
					self.display.display(self.game)

	def _build_model(self):
		model = model_fn(self.state_shape, self.action_size)
		return model

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return np.random.randint(self.action_size)
		else:
			return np.argmax(self.model.predict(state)[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)
			if done:
				target[0][action] = reward
			else:
				t = self.target_model.predict(next_state)[0]
				target[0][action] = reward + self.gamma * np.amax(t)
			self.model.fit(state, target, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def train(self, batch_size=32, episodes=5000, period=50):
		types = self.state_shape[-1]
		steps = 0
		for e in range(episodes):
			game = Game(self.game.size, self.game.score_to_win)
			state = one_hot_encoding(game.board, types)[np.newaxis, :]
			done = 0
			while not done:
				if steps % period == 0:
					self.update_target_model()
				action = self.act(state)
				game.move(action)

				next_state = one_hot_encoding(game.board, types)[np.newaxis, :]
				done = game.end
				reward = np.log2(game.score)
				if done == 1:
					reward = -20
				elif done == 2:
					reward = 20

				self.remember(state, action, reward, next_state, done)
				state = next_state
				if len(self.memory) > batch_size:
					self.replay(batch_size)
				steps += 1
			print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, game.score, self.epsilon))

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)


if __name__ == "__main__":
	path = '2048.h5'

	game = Game(score_to_win=2048)
	agent = DDQNAgent(game)
	agent.load(path)
	agent.train()
	agent.save(path)