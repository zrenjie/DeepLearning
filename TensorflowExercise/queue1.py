import tensorflow as tf
import threading 
import time

def add(coord, i):
    while not coord.should_stop():
        sess.run(enque)
        
        if (i == 11):
            coord.request_stop()

        
if __name__ == '__main__':
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    queue = tf.FIFOQueue(capacity=100, dtypes=tf.float32, shapes=())
    get_random_value = tf.random_normal(shape=())
    enque = queue.enqueue(get_random_value)
    threads = [threading.Thread(target=add, args=(coord, i)) for i in range(10)]
    coord.join(threads)
    for i in range(len(threads)):
        threads[i].start()
     
    print(sess.run(queue.size()))   
    time.sleep(0.001)
    print(sess.run(queue.size()))
    time.sleep(0.001)
    print(sess.run(queue.size()))
        
    
    