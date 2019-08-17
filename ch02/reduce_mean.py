import tensorflow as tf

a = [[1., 2., 3.], [4., 5., 6.]]

x = tf.placeholder(tf.float32, [2, 3])
y_1 = tf.reduce_mean(x, axis=0)
y_2 = tf.reduce_mean(x, axis=1)
y_3 = tf.reduce_mean(x)

with tf.Session() as sess:
    print(a)
    print(sess.run(y_1, feed_dict={x:a}))
    print(sess.run(y_2, feed_dict={x:a}))
    print(sess.run(y_3, feed_dict={x:a}))