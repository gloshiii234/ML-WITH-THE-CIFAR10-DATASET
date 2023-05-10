```python
def set_up_images(self):
    print('Setting  up training images and labels')
    self.training_images = np.vstack([d[b'data'] for d in self.all_train_batches])
    train_len = len(self.training_images)
    
    self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
    self.training_labels = one_hot_encoder(np.hstack([d[b'labels'] for d in self.all_train_batches]),10)
    
    print("I am setting up Test Images and labels")
    self.test_images = no.vstack([d[b'data'] for d in self.test_batch])
    test_len = len(self.test_images)
    
    self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
    self.test_labels = one_hot_encode(np.hstack([d[b'labels'] for d in self.test_batch]),10)
    
def next_batch(self,batchsize):
    x = self.training_images[self.i:self.i + batch_size].reshape(100,32,32,3)
    y = self.training_labels[self.i:self.i + batch_size]
    self.i = (self.i + batch_size)% len(self.training_images)
    return(x,y)
```


```python
    #Before your tf session, run this:

ch = CifarHelper()
ch.set_up_images()
#batch = ch.next_batch(100)
```


```python
import tensorflow as tf
```


```python
x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y = tf.placeholder(tf.float32,shape = [None,10])
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_9380\4251000757.py in <module>
    ----> 1 x = tf.placeholder(tf.float32, shape=[None,32,32,3])
          2 y = tf.placeholder(tf.float32,shape = [None,10])
    

    AttributeError: module 'tensorflow' has no attribute 'placeholder'



```python
##MNST
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

def max_pool_3by2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1],paddings='SAME')

def convolutional_layer(input_x,shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b
```


```python

```
