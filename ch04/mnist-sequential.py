from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("data/", one_hot=True)

""" モデル構築開始 """
# 入力データ整形
num_seq = 28
num_input = 28

x = tf.placeholder(tf.float32, [None, 784])
# (バッチサイズ, 高さ, 幅)の3階テンソルに変換
input = tf.reshape(x, [-1, num_seq, num_input])

# ユニット数128個ののLSTMセル
# 3段に積む
stacked_cells = []
for i in range(3):
    stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)

# dynamic_rnnによる時間展開
outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)

# 最後の時間軸のTensorを取得
last_output = outputs[:, -1, :]

w = tf.Variable(tf.truncated_normal([128, 10], stddev=0.1))
b = tf.Variable(tf.zeros([10]))

out = tf.nn.softmax(tf.matmul(last_output, w) + b)

