# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:59:12 2021

@author: santc
"""

import math
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import pandas_datareader as data_reader

from tqdm import tqdm
from collections import deque

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

print("Tensorflow",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## Etapa 3: Construção da IA para negociação de ações
class AI_Trader():
	# Construtor
	# self: parâmetro do python que faz referência ao objeto e não a classe
	def __init__(self, state_size, action_space = 3, model_name = "AITrader"):
		# ESTADOS vindo do ambiente, nessa implementação são os valores das
		# entradas da rede neural
		self.state_size = state_size
		# AÇÕES possíveis, nesse caso 3, comprar, vender ou não fazer nada
		self.action_space = action_space
		# tamanho da memória da EXPERIÊNCIA DE REPLAY
		# Memória tem 2000 ações. Será executado um treinamento após 2000 ações.
		# Evita o treinamento passo por passo em ambientes repetitivos e o
		# overffiting.
		self.memory = deque(maxlen = 10)
		self.model_name = model_name
		# FATOR DE DESCONTO da equação de Bellman
		self.gamma = 0.95
		#####
		# Definições do EXPLORATION vs EXPLOITATION (Depois de treinada a rede
		# continuará executando as ações boas, contudo continuará explorando o ambiente)
		# Define se as ações serão randômicas ou por meio da rede neural, nesse
		# caso 1 = 100%, todas as ações serão randômicas, pois no início os pesos
		# não estarão otimizados, desse modo suas ações devem ser randômicas
		# para que ele possa aprender com o passar das épocas (episódios).
		self.epsilon = 1.0
		# Define um teto para o decremento, nesse caso 0.01 para que haja uma probabilidade
		# baixa de a rede efetuar uma ação randômica. É interessante que a rede
		# não execute ações somente em seu conhecimento, mas sim com uma probabilidade
		# baixa de executar ações aleatórias para que o agente não corra o risco
		# de ficar preso em um mínimo local na descida do gradiente.
		self.epsilon_final = 0.01
		# Decrementa o self.epsilon, pois conforme a rede se especializa e os
		# pesos ficam otimizados, as ações dela não devem mais ser randômicas e
		# sim por ações com base no que a rede conhece.
		#####
		self.epsilon_decay = 0.995
		self.model = self.model_builder()
	
	# Criação da rede neural
	def model_builder(self):
		model = tf.keras.models.Sequential()
		model.add(tf.keras.Input(shape=(self.state_size,)))
		model.add(tf.keras.layers.Dense(3, kernel_initializer = 'random_uniform'))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.2))
		
		model.add(tf.keras.layers.Dense(3, kernel_initializer = 'random_uniform'))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.2))
		#model.add(tf.keras.layers.Dense(units = 4, activation = "relu"))
		#model.add(tf.keras.layers.Dense(units = 4, activation = "relu"))
		model.add(tf.keras.layers.Dense(units = self.action_space, activation = "linear"))
		#model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr = 0.001))
		model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(lr = 0.001))
		
		return model
	# AÇÃO
	def trade(self, state):
		# AÇÃO ALEATÓRIA, o agente executará a maioria das ações aleatórias 
		# até estar com seus Qs ajustados
		if random.random() <= self.epsilon:
			return random.randrange(self.action_space)
		
		# Busca um ESTADO do ambiente, que nesse caso são os preços do ativo.
		# O retorno será os valores de Q, o agente escolherá o maior valor de Q e
		# isso será a ação a ser executada.
		actions = self.model.predict(state)
		# Os valores de Q estão em actions[0], argmax pega o maior valor de Q
		# entre os 3 Qs do array, conforme o comentário anterior.
		return np.argmax(actions[0])
	
	# EXPERIÊNCIA DE REPLAY
	# batch_size: de quantos em quantos registros serão efetuadas as atualizações
	# do pesos
	def batch_train(self, batch_size):
		batch = []
		# Busca na memória da rede os últimos registros de negociação
		for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
			batch.append(self.memory[i])
		
		# INTUIÇÃO DEEP Q-LEARNING, similar a Diferença Temporal de Markov, indicará
		# qual será a melhor a ação a ser tomada pelo agente através da aproximação
		# entre Q e Q-Target: L = Ʃ(Q-Target - Q)²
		# state: Estado atual
		# action: Ação que será executada
		# reward: Recompensa após a ação, nesse caso lucro ou prejuízo monetário
		# next_state: Próxima ação que será executada
		# done: Final da época (episódio)
		for state, action, reward, next_state, done in batch:
			if not done:
				# Valor da recompensa no próximo estado, conforme o componente 
				# da equação de Bellman: R(s,a) + γV(s')
				reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
	        # Q-Target
			target = self.model.predict(state)
			# Ação que foi executada pelo agente
			target[0][action] = reward
			
			self.model.fit(state, target, epochs=1, verbose=0)
		# Atualiza o epsilon, variável relacionada ao EXPLORATION vs EXPLOITATION
		if self.epsilon > self.epsilon_final:
			self.epsilon *= self.epsilon_decay

## Etapa 4: Pré-processamento da base de dados
### Definição de funções auxiliares
#### Sigmoid
# Usada apenas para a normalização de dados transformando-os na escala entre 0 e 1.
# NÃO está sendo utilizada com a finalidade de função de ativação! Como o range
# de dados não é conhecido a sigmoid é uma alternativa para esse tipo de caso.
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

#### Formatação de preços para exibição na saída padrão
def stocks_price_format(n):
	if n < 0:
		return "- {}pts".format(abs(n))
	else:
		return " {}pts".format(abs(n))
	
##### TESTANDO VARIÁVEIS
stocks_price_format(100)
#####

#### Carregador da base de dados
def dataset_loader(stock_name):
	dataset = data_reader.DataReader(stock_name, data_source = "yahoo")
	close = dataset['Close']
	return close

### Criador de ESTADOs
# window_size: Período (Número de preços passados que serão coletados para ser
# utilizados no treinamento da rede neural)
def state_creator(data, timestep, window_size):
	starting_id = timestep - window_size + 1

	# Quando a coleta começa no índice 0 não há dados passados. Para contornar isso
	# se starting_id < 0 então o dado do índice 0 será repedido
	if starting_id >= 0:
		windowed_data = data[starting_id:timestep + 1]
	else:
		windowed_data = - starting_id * [data[0]] + list(data[0:timestep + 1])
	
	state = []
	for i in range(window_size - 1):
		# Diferença entre o próximo preço e o preço anterior
		state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))
    
	return np.array([state]), windowed_data

### Carregando a base de dados
#stock_name = "AAPL"
#data = dataset_loader(stock_name)

dataset = pd.read_csv('winm21.csv')

values = dataset['close'].values
data = pd.Series(values, index = dataset['date']+' '+ dataset['time'])

data_normalize = (data-data.min())/(data.max()-data.min())

values = values.reshape((len(values), 1))
normalizador = MinMaxScaler(feature_range=(0,1))
normalizador = normalizador.fit(values)

#data_normalized = normalizador.transform(values)

#data = dataset['close']

##### TESTANDO VARIÁVEIS
# s: estados (Diferenças entre os preços subsequentes e anteriores)
# w: janela (Data e preço de fechamento)
# Coleta 5 registros da base de dados iniciando no índice 0
# O tamanho do array de w é 5 enquanto de s é 4 porque conforme comentado dentro
# da função s tem o cálculo da diferença entre o preço subsequente e anterior em
# cada passo dentro do array de w.
# s, w = state_creator(data_normalize, 0, 5)
#####

## Etapa 5: Treinando a IA
### Configuração dos hyper parâmetros
window_size = 4
# Conceito parecido com o de épocas
episodes = 5000
batch_size = 2
# Total de registro da base de dados
data_samples = len(data_normalize) - 1




##### TESTANDO VARIÁVEIS
# data_samples
#####

### Definição do modelo
trader = AI_Trader(window_size)
trader.model.summary()

desempenhos = []
h5_anterior = 1

### Loop de treinamento
for episode in range(1, episodes + 1):
	print("Episódio: {}/{}".format(episode, episodes))
	state, windowed_data = state_creator(data_normalize, 0, window_size + 1)
	total_profit = 0
	# Quantidade de ações na conta do agente
	ultimo_preco_negociado = 0
	
	acao_ativa = 3
	desempenho = []
	
	# Itera todos os registros da base de dados a cada episódio
	for t in tqdm(range(data_samples)):
		action = trader.trade(state)
		next_state, next_windowed_data = state_creator(data_normalize, t + 1, window_size + 1)
		reward = 0
		
		datat = normalizador.inverse_transform(data_normalize[t].reshape(-1, 1)).astype(int)
		datat = datat[0][0]
		if action == 1:
			if acao_ativa == 1 or acao_ativa == 2:
				print(' tentou comprar, há operação aberta')
			else:
				ultimo_preco_negociado = datat
				print(" compra: {}".format(stocks_price_format(datat)))
				acao_ativa = 1
		if action == 2:
			if acao_ativa == 1 or acao_ativa == 2:
				print(' tentou vender, há operação aberta')
			else:
				ultimo_preco_negociado = datat
				print(" venda: {}".format(stocks_price_format(datat)))
				acao_ativa = 2
		elif ultimo_preco_negociado > 0:			
			if acao_ativa == 1:
				resultado = datat - ultimo_preco_negociado
				if resultado != 0:
					reward = max(data_normalize[t] - ultimo_preco_negociado, 0)
					total_profit += resultado
					print(" zerada compra: {} | resultado: {}".format(stocks_price_format(datat),stocks_price_format(resultado)))
					acao_ativa = 3
				else:
					acao_ativa = 1
			elif acao_ativa == 2:
				resultado = ultimo_preco_negociado - datat
				if resultado != 0:
					reward = max(ultimo_preco_negociado - data_normalize[t], 0)
					total_profit += resultado
					print(" zerada venda: {} | resultado: {}".format(stocks_price_format(datat),stocks_price_format(resultado)))
					acao_ativa = 3
				else:
					acao_ativa = 2
			else:
				print(" tentou zerar, não há operação aberta")
				acao_ativa = 3
		
		desempenho.append(total_profit)
		
		#ultima_acao = action
		
		if t == data_samples - 1:
			done = True
		else:
			done = False
			
		trader.memory.append((state, action, reward, next_state, done))
		
		state = next_state
		
		if done:
			print("########################")
			print("Total profit: {}".format(total_profit))
			print("########################")
			
			desempenhos.append(desempenho)
			desempenho_plot = dataset.iloc[0:len(dataset)-1,0]
			desempenho_plot = pd.Series(desempenhos[episode - 1], index = desempenho_plot.index)
			fechamentos = dataset.get('close')
			
			fig, (dat, des) = plt.subplots(2)
			
			dat.plot(dataset.get('close'), label = 'Saldo do WINM21')
			dat.set_title('Episódio {} - WINM21'.format(episode))
			
			des.plot(desempenho_plot, label = 'Saldo da IA')
			des.set_title('Episódio {} - Saldos'.format(episode))
			
			plt.subplots_adjust(hspace=0.5)
			
			fig.savefig('episodio{}.png'.format(episode), dpi=300)
			
			plt.close()
			
		if len(trader.memory) > batch_size:
			trader.batch_train(batch_size)
			trader.memory = []
		
	if episode % 2 == 0:
		trader.model.save("ai_trader_ep{}ao{}.h5".format(h5_anterior,episode))
		h5_anterior = episode + 1
		