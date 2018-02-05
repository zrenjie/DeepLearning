
import tensorflow as tf

def main():
    str1 = tf.placeholder(tf.string, [], "Hello")
    str2 = tf.placeholder(tf.string, [], "world")
    str3 = str1 + str2
    
    with tf.Session() as sess:
        print(sess.run(str3, feed_dict = {str1:"Hello", str2:"World"}))


if __name__ == "__main__":
    main()
