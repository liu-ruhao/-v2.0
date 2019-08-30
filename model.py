from sklearn import preprocessing
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
import tensorflow as tf
import config as CF
import numpy as np
import gc

class model(object):
    def __init__(self,sess,config,logging):
        self.sess=sess
        self.config=config
        self.logging=logging
        self._checkpoint_path=self.config["ckpt"]
        self.hight=CF.config["IMG_H"]
        self.width=CF.config["IMG_W"]
        self.n_classes=CF.config["N_CLASSES"]
        self.model_type=CF.config["num"]
        self.global_step=tf.Variable(0,trainable=False)
        self.build()
        self.print_var()
        self.loggingAll()
        self._saver=tf.train.Saver(tf.global_variables(),max_to_keep=10)
        self.initialize()
    def loggingAll(self):
        for name in dir(self):
            if name.find("_")==0 and name.find("_")==-1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))
    def _input(self):
        self.image_batch= tf.placeholder(tf.float32, shape=[None,self.hight, self.width, 3])
        self.image_label_batch=tf.placeholder(tf.int32, shape=[None,self.n_classes])
        #self.test=tf.placeholder(tf.float32, shape=[None,self.hight, self.width, 3])       
        #self.labels=
        #self.
    def build(self):
        self._input()
        if self.model_type==1:
            print("选用模型1")
            self.structure_1(self.image_batch)
        elif self.model_type=="vgg16":
              print("选用模型2")
              self.is_training = tf.placeholder(tf.bool)
              self.is_use_l2 = tf.placeholder(tf.bool)
              self.lam = tf.placeholder(tf.float32)
              self.keep_prob = tf.placeholder(tf.float32)
              self.structure_2(self.image_batch)
        elif self.model_type=="Alexnet":
            print("选用模型3")
            self.keep_prob = tf.placeholder(tf.float32)
            self.weights = {
                            'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
                            'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
                            'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
                            'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
                            'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
                            'wd1': tf.Variable(tf.random_normal([9216, 4096])),
                            'wd2': tf.Variable(tf.random_normal([4096, 4096])),
                            'out': tf.Variable(tf.random_normal([4096, self.n_classes]))
                        }

            self.biases = {
                            'bc1': tf.Variable(tf.random_normal([96])),
                            'bc2': tf.Variable(tf.random_normal([256])),
                            'bc3': tf.Variable(tf.random_normal([384])),
                            'bc4': tf.Variable(tf.random_normal([384])),
                            'bc5': tf.Variable(tf.random_normal([256])),
                            'bd1': tf.Variable(tf.random_normal([4096])),
                            'bd2': tf.Variable(tf.random_normal([4096])),
                            'out': tf.Variable(tf.random_normal([self.n_classes]))
                        }
            self.alex_net(self.image_batch,self.weights, self.biases, self.keep_prob)
        elif self.model_type=="Googlenet":
            print("选用模型4")
            self.keep_prob = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)
            self.restore_logits=None
            self.googlenet(self.image_batch)
        #self.tra_acc(self.image_batch,self.image_label_batch) 
        #self.tes_acc(self.image_batch,self.image_label_batch)
        self.loss(self.image_label_batch)
        self.tra_acc()
        #self.tes_acc()
        self.opt() 
        self.logging.info("model is built")
#####################################################################################################################
#########################################################################################################################
############################################################################################################################
    def structure_1(self,images):
        #if ((int(np.log2(self.hight))==np.log2(self.hight))&self.hight>=8):
            with tf.variable_scope('conv1') as scope:
                weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                                      name='weights', dtype=tf.float32)
        
                biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                     name='biases', dtype=tf.float32)
        
                conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
                #print(conv)
                pre_activation = tf.nn.bias_add(conv, biases)
                conv1 = tf.nn.relu(pre_activation, name=scope.name)
            with tf.variable_scope('pooling1_lrn') as scope:
                pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
                #print(pool1)
                norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
            with tf.variable_scope('conv2') as scope:
                weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                                      name='weights', dtype=tf.float32)
        
                biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                                     name='biases', dtype=tf.float32)
        
                conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name='conv2')
    
            with tf.variable_scope('pooling2_lrn') as scope:
                norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
                pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
                #print(pool2)
            with tf.variable_scope('local3') as scope:
                reshape = tf.reshape(pool2,shape=[-1, 32*32*16])
                #print(reshape)
                dim = reshape.get_shape()[1].value
                #print(dim)
                
                weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                                      name='weights', dtype=tf.float32)
                biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                     name='biases', dtype=tf.float32)
                
                local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
                #print(local3)
            with tf.variable_scope('local4') as scope:
                weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                                      name='weights', dtype=tf.float32)
                biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                     name='biases', dtype=tf.float32)
        
                local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
                #print(local4)
            with tf.variable_scope('softmax_linear') as scope:
                weights = tf.Variable(tf.truncated_normal(shape=[128, self.n_classes], stddev=0.005, dtype=tf.float32),
                                      name='softmax_linear', dtype=tf.float32)
                biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[self.n_classes]),
                                     name='biases', dtype=tf.float32)
                self.softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
                #print(self.softmax_linear)    
###########################################################################################################################
##########################################################################################################################
#############################################################################################################################
    def weight_variable(self,shape,n,use_l2,lam):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=1 / n))
        # L2正则化
        if use_l2 is True:
            weight_loss = tf.multiply(tf.nn.l2_loss(weight), lam, name='weight_loss')
            tf.add_to_collection('losses', weight_loss)
        return weight

    def bias_variable(self, shape):
        bias = tf.Variable(tf.constant(0.1, shape=shape))
        return bias

    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
    #learning_rate
    def structure_2(self, images):
        #image=[B,64,64,3]
        with tf.name_scope('conv1_layer'):
            w_conv1 =self.weight_variable([3, 3, 3, 64], 64, use_l2=False, lam=0)
            b_conv1 = self.bias_variable([64])
            conv_kernel1 = self.conv2d(images, w_conv1)#[B,64,64,64]
            
            bn1 = tf.layers.batch_normalization(conv_kernel1, training=self.is_training)
            conv1 = tf.nn.relu(tf.nn.bias_add(bn1, b_conv1))
        
            w_conv2 = self.weight_variable([3, 3, 64, 64], 64, use_l2=False, lam=0)
            b_conv2 = self.bias_variable([64])
            conv_kernel2 = self.conv2d(conv1, w_conv2)#[B,64,64,64]
            bn2 = tf.layers.batch_normalization(conv_kernel2, training=self.is_training)
            conv2 = tf.nn.relu(tf.nn.bias_add(bn2, b_conv2))
        
            pool1 = self.max_pool_2x2(conv2)  #[B,32,32,64]
            result1 = pool1
       
        with tf.name_scope('conv2_layer'):
            w_conv3 = self.weight_variable([3, 3, 64, 128], 128, use_l2=False, lam=0)
            b_conv3 = self.bias_variable([128])
            conv_kernel3 = self.conv2d(result1, w_conv3)#[B,32,32,128]
            bn3 = tf.layers.batch_normalization(conv_kernel3, training=self.is_training)
            conv3 = tf.nn.relu(tf.nn.bias_add(bn3, b_conv3))
        
            w_conv4 = self.weight_variable([3, 3, 128, 128], 128, use_l2=False, lam=0)
            b_conv4 = self.bias_variable([128])
            conv_kernel4 = self.conv2d(conv3, w_conv4)
            bn4 = tf.layers.batch_normalization(conv_kernel4, training=self.is_training)
            conv4 = tf.nn.relu(tf.nn.bias_add(bn4, b_conv4))#[B,32,32,128]
        
            pool2 = self.max_pool_2x2(conv4)  # 112*112 -> 56*56#[B,16,16,128]
            result2 = pool2          
       
        with tf.name_scope('conv3_layer'):
            w_conv5 = self.weight_variable([3, 3, 128, 256], 256, use_l2=False, lam=0)
            b_conv5 = self.bias_variable([256])
            conv_kernel5 = self.conv2d(result2, w_conv5)#[B,16,16,256]
            bn5 = tf.layers.batch_normalization(conv_kernel5, training=self.is_training)
            conv5 = tf.nn.relu(tf.nn.bias_add(bn5, b_conv5))
        
            w_conv6 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
            b_conv6 = self.bias_variable([256])
            conv_kernel6 = self.conv2d(conv5, w_conv6)
            bn6 = tf.layers.batch_normalization(conv_kernel6, training=self.is_training)
            conv6 = tf.nn.relu(tf.nn.bias_add(bn6, b_conv6))
        
            w_conv7 = self.weight_variable([3, 3, 256, 256], 256, use_l2=False, lam=0)
            b_conv7 = self.bias_variable([256])
            conv_kernel7 = self.conv2d(conv6, w_conv7)
            bn7 = tf.layers.batch_normalization(conv_kernel7, training=self.is_training)
            conv7 = tf.nn.relu(tf.nn.bias_add(bn7, b_conv7))
            
            pool3 = self.max_pool_2x2(conv7)  # 56*56 -> 28*28
            result3 = pool3   
       
        with tf.name_scope('conv4_layer'):
            w_conv8 = self.weight_variable([3, 3, 256, 512], 512, use_l2=False, lam=0)
            b_conv8 = self.bias_variable([512])
            conv_kernel8 = self.conv2d(result3, w_conv8)
            bn8 = tf.layers.batch_normalization(conv_kernel8, training=self.is_training)
            conv8 = tf.nn.relu(tf.nn.bias_add(bn8, b_conv8))
        
            w_conv9 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv9 = self.bias_variable([512])
            conv_kernel9 = self.conv2d(conv8, w_conv9)
            bn9 = tf.layers.batch_normalization(conv_kernel9, training=self.is_training)
            conv9 = tf.nn.relu(tf.nn.bias_add(bn9, b_conv9))
            
            w_conv10 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv10 = self.bias_variable([512])
            conv_kernel10 = self.conv2d(conv9, w_conv10)
            bn10 = tf.layers.batch_normalization(conv_kernel10, training=self.is_training)
            conv10 = tf.nn.relu(tf.nn.bias_add(bn10, b_conv10))
            
            pool4 = self.max_pool_2x2(conv10)  # 28*28 -> 14*14
            result4 = pool4
       
        with tf.name_scope('conv5_layer'):
            w_conv11 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv11 = self.bias_variable([512])
            conv_kernel11 = self.conv2d(result4, w_conv11)
            bn11 = tf.layers.batch_normalization(conv_kernel11, training=self.is_training)
            conv11 = tf.nn.relu(tf.nn.bias_add(bn11, b_conv11)) 
            
            w_conv12 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv12 =self. bias_variable([512])
            conv_kernel12 = self.conv2d(conv11, w_conv12)
            bn12 = tf.layers.batch_normalization(conv_kernel12, training=self.is_training)
            conv12 = tf.nn.relu(tf.nn.bias_add(bn12, b_conv12))
            
            w_conv13 = self.weight_variable([3, 3, 512, 512], 512, use_l2=False, lam=0)
            b_conv13 = self.bias_variable([512])
            conv_kernel13 =self.conv2d(conv12, w_conv13)
            bn13 = tf.layers.batch_normalization(conv_kernel13, training=self.is_training)
            conv13 = tf.nn.relu(tf.nn.bias_add(bn13, b_conv13))
            
            pool5 = self.max_pool_2x2(conv13)  # 14*14 -> 7*7#[B,2,2,512]
            result5 = pool5      
        
        with tf.name_scope('fc1_layer'):
            w_fc14 = self.weight_variable([2 * 2 * 512, 4096], 4096, use_l2=self.is_use_l2, lam=self.lam)
            b_fc14 =self.bias_variable([4096])
            #print(b_fc14)
            #print(result5)
            #exit()
            result5_flat = tf.reshape(result5, [-1, 2 * 2 * 512])
            fc14 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result5_flat, w_fc14), b_fc14))#[B,4096]
            result6 = tf.nn.dropout(fc14, self.keep_prob)#?
        
        with tf.name_scope('fc2_layer'):
            w_fc15 = self.weight_variable([4096, 4096], 4096, use_l2=self.is_use_l2, lam=self.lam)
            b_fc15 =self.bias_variable([4096])
            fc15 = tf.nn.relu(tf.nn.bias_add(tf.matmul(result6, w_fc15), b_fc15))
            result7 = tf.nn.dropout(fc15, self.keep_prob)        
        # 输出层
        with tf.name_scope('output_layer'):
            w_fc16 = self.weight_variable([4096, self.n_classes], self.n_classes, use_l2=self.is_use_l2, lam=self.lam)
            b_fc16 = self.bias_variable([self.n_classes])
            fc16 = tf.matmul(result7, w_fc16) + b_fc16
            self.softmax_linear = tf.nn.softmax(fc16)
    
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
    def dense_to_one_hot(self,labels_dense):
        enc = preprocessing.OneHotEncoder(sparse=True, n_values=self.n_classes)
        enc.fit(labels_dense)
        array = enc.transform(labels_dense).toarray()
        return array
# 对原始数据洗牌，跑不动
    def shuffle_1(self,*arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
            print(arrs[i])
        p = np.random.permutation(len(arrs[0]))
        data_shape = arrs[0].shape
        new_data = np.empty(data_shape, np.float)
        new_label = np.empty(arrs[1].shape, np.float)
        data = arrs[0]
        label = arrs[1]
        for i in range(len(data)):
            tmp_data = data[p[i]]
            new_data[i] = tmp_data   
            tmp_label = label[p[i]]
            new_label[i] = tmp_label   
        return new_data, new_label
        
    def shuffle_2(self,*arrs):
        arrs = list(arrs)
        for i, arr in enumerate(arrs):
            assert len(arrs[0]) == len(arrs[i])
            arrs[i] = np.array(arr)
        p = np.random.permutation(len(arrs[0]))
        return tuple(arr[p] for arr in arrs)
    
    
    def shuffle_3(self,size):
        p = np.random.permutation(size)
        return p
    # 卷积层
    def conv2d_A(self, name, input, w, b, stride, padding='SAME'):
        # 测试
        x = input.get_shape()[-1]
        x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)
        x = tf.nn.bias_add(x, b)
        data_result = tf.nn.relu(x, name=name)
        # 输出参数
        # tf.histogram_summary(name + '/卷积层', data_result)
        gc.collect()
        return data_result
    # 最大下采样
    def max_pool(self,name, input, k, stride):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
    # 归一化操作 ToDo 正则方式待修改
    def norm(self,name, input, size=4):
        return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
     
    # 定义整个网络 
    def alex_net(self,x, weights, biases,dropout):
        # 卷积层1  96个11*11*3 filter
        dropout=1-self.keep_prob 
        conv1 = self.conv2d_A('conv1', x, weights['wc1'], biases['bc1'], stride=2)
        # 下采样层 kernel：3*3 步长：2
        pool1 = self.max_pool('pool1', conv1, k=3, stride=2)       
        # 正则化 96 个...
        norm1 = self.norm('norm1', pool1, size=5)      
        # 卷积层2 ToDo 两组filter， padding=2
        conv2 = self.conv2d_A('conv2', norm1, weights['wc2'], biases['bc2'], stride=1, padding="VALID")       
        # 下采样
        pool2 = self.max_pool('pool2', conv2, k=3, stride=1)        
        # 归一化
        norm2 = self.norm('norm2', pool2, size=5)
        # 卷积层3 padding=1
        conv3 = self.conv2d_A('conv3', norm2, weights['wc3'], biases['bc3'], stride=1, padding="VALID")
        # 卷积层4 padding=1
        conv4 = self.conv2d_A('conv4', conv3, weights['wc4'], biases['bc4'], stride=1, padding="VALID")        
        # 卷积层5 padding=1
        conv5 = self.conv2d_A('conv5', conv4, weights['wc5'], biases['bc5'], stride=1, padding="VALID")              
        pool5 = self.max_pool('pool5', conv5, k=3, stride=1)
    
        # 全连接层1
        # 先把特征图转为向量
        fc1 = tf.reshape(pool5, [-1, 9216])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1, name='fc1')
        # Dropout
        drop1 = tf.nn.dropout(fc1, dropout)
    
        # 全连接层2
        fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2, name='fc2')
        
        # Dropout
        drop2 = tf.nn.dropout(fc2, dropout)      
        self.softmax_linear= tf.add(tf.matmul(drop2, weights['out']), biases['out'])
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
    def get_inception_layer(self, inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size ):
        with tf.variable_scope("conv_1x1"):
            conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )    
            
        with tf.variable_scope("conv_3x3"):
            conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
            conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
            
        with tf.variable_scope("conv_5x5"):
            conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
            conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
            
        with tf.variable_scope("pool_proj"):
            pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
            pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
            
        if tf.__version__ == '0.11.0rc0':
            return tf.concat(3, [conv11, conv33, conv55, pool11])
        return tf.concat([conv11, conv33, conv55, pool11], 3)
    
    def aux_logit_layer(self,inputs, n_classes, is_training ):        
        with tf.variable_scope("pool2d"):
            pooled = layers.avg_pool2d(inputs, [ 2, 2 ], stride = 2 )  
            
        with tf.variable_scope("conv11"):
            conv11 = layers.conv2d( pooled, 128, [1, 1] )
            
        with tf.variable_scope("flatten"):
            flat = tf.reshape( conv11, [-1, 2048] )

        with tf.variable_scope("fc"):
            fc = layers.fully_connected( flat, 1024, activation_fn=None )
            
        with tf.variable_scope("drop"):
            drop = layers.dropout( fc, 0.3, is_training = self.is_training )
            
        with tf.variable_scope( "linear" ):
            linear = layers.fully_connected( drop, n_classes, activation_fn=None )
            
        with tf.variable_scope("soft"):
            soft = tf.nn.softmax( linear )
        return soft
    
     # 定义整个网络
    def googlenet(self,inputs):
        dropout=1-self.keep_prob
        '''
        Implementation of https://arxiv.org/pdf/1409.4842.pdf
        '''        
        with tf.name_scope( "google_net", "googlenet", [inputs] ):
            with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME' ):
                conv0 = layers.conv2d( inputs, 64, [ 7, 7 ], stride = 1, scope = 'conv0' )
                pool0 = layers.max_pool2d(conv0, [3, 3], scope='pool0')
                conv1_a = layers.conv2d( pool0, 64, [ 1, 1 ], scope = 'conv1_a' )
                conv1_b = layers.conv2d( conv1_a, 192, [ 3, 3 ], scope = 'conv1_b' )              
                pool1 = layers.max_pool2d(conv1_b, [ 3, 3 ], scope='pool1')                
                
                with tf.variable_scope("inception_3a"):
                    inception_3a = self.get_inception_layer( pool1, 64, 96, 128, 16, 32, 32 )
                    
                with tf.variable_scope("inception_3b"):
                    inception_3b = self.get_inception_layer( inception_3a, 128, 128, 192, 32, 96, 64 )
                    
                pool2 = layers.max_pool2d(inception_3b, [ 3, 3 ], scope='pool2')
                
                with tf.variable_scope("inception_4a"):
                    inception_4a = self.get_inception_layer( pool2, 192, 96, 208, 16, 48, 64 )
                    
                #with tf.variable_scope("aux_logits_1"):
                    #aux_logits_1 = self.aux_logit_layer( inception_4a, self.n_classes, self.is_training )
                with tf.variable_scope("inception_4b"):
                    inception_4b = self.get_inception_layer( inception_4a, 160, 112, 224, 24, 64, 64 )
                    
                with tf.variable_scope("inception_4c"):
                    inception_4c = self.get_inception_layer( inception_4b, 128, 128, 256, 24, 64, 64 )
    
                with tf.variable_scope("inception_4d"):
                    inception_4d = self.get_inception_layer( inception_4c, 112, 144, 288, 32, 64, 64 )
    
                #with tf.variable_scope("aux_logits_2"):
                    #aux_logits_2 = self.aux_logit_layer( inception_4d, self.n_classes, self.is_training )                    
                with tf.variable_scope("inception_4e"):
                    inception_4e = self.get_inception_layer( inception_4d, 256, 160, 320, 32, 128, 128 )
                    
                pool3 = layers.max_pool2d(inception_4e, [ 3, 3 ], scope='pool3')               
                with tf.variable_scope("inception_5a"):
                    inception_5a = self.get_inception_layer( pool3, 256, 160, 320, 32, 128, 128 )  
                    
                with tf.variable_scope("inception_5b"):
                    inception_5b = self.get_inception_layer( inception_5a, 384, 192, 384, 48, 128, 128 ) 
                    
                pool4 = layers.avg_pool2d(inception_5b, [ 2, 2 ], stride = 1, scope='pool4')
                
                reshape = tf.reshape( pool4, [-1, 1024*3*3] )
                
                dropout = layers.dropout( reshape,self.keep_prob, is_training = self.is_training )
                
                logits = layers.fully_connected( dropout, self.n_classes, activation_fn=None, scope='logits')
                
                predictions = tf.nn.softmax(logits, name='predictions')
        self.softmax_linear=predictions         
        
        
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################        
    def loss(self,image_label_batch):
        with tf.variable_scope('loss'):
            #print(image_label_batch)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_linear, 
                                                                           labels=self.image_label_batch,
                                                                           name='xentropy_per_example')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
    #def loss(self):        
     #   self.predict=self.softmax_linear
       # self.loss=tf.nn.l2_loss(self.predict-self.y)
    def print_var(self):
        for item in dir(self):
            type_string=str(type(getattr(self,item)))
            print(item,type_string)
    def opt(self):
        self._opt=tf.train.AdadeltaOptimizer(self.config["learning_rate"])
        self._train_opt=self._opt.minimize(self.loss,global_step=self.global_step)
        
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    def train(self,train_batch,train_label_batch,i):
        #print(train_batch)
        #print("QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ")
        #print(train_label_batch)
        if self.model_type==1:
           feed_dict={self.image_batch:train_batch,self.image_label_batch:train_label_batch}
           #_,tra_loss,global_step,tra_acc=self.sess.run([self._train_opt,self.loss,self.global_step,self.tra_accurate],feed_dict=feed_dict)
           #if i%10==0:
               #print("loss is %s,global_step is %s,tra_acc is %.2f%%,i is %s"%(tra_loss,global_step,tra_acc*100.0,i))
               #print("loss is %s,global_step is %s,tra_acc is %.2f%%"%(tra_loss,global_step,tra_acc*100.0))
               #self.logging.info("loss is %s,global_step is %s,tra_acc is %.2f%%,i is %s"%(tra_loss,global_step,tra_acc*100.0,i))
               #if i %1000==0:
                   #self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)
        elif self.model_type=="vgg16":       
            feed_dict={self.image_batch:train_batch, self.image_label_batch:train_label_batch,self.keep_prob:0.8, self.is_training:False,self.is_use_l2:True,self.lam:0.001}
            #_,tra_loss,global_step,tra_acc=self.sess.run([self._train_opt,self.loss,self.global_step,self.tra_accurate],feed_dict=feed_dict)
        elif self.model_type=="Alexnet":
            feed_dict={self.keep_prob:0.8,self.image_batch:train_batch, self.image_label_batch:train_label_batch}
        elif self.model_type=="Googlenet":
            feed_dict={self.image_batch:train_batch, self.image_label_batch:train_label_batch,self.keep_prob:0.4,self.is_training:True}
            
        _,tra_loss,global_step,tra_acc=self.sess.run([self._train_opt,self.loss,self.global_step,self.tra_accurate],feed_dict=feed_dict)
        if i%10==0:
            #print("loss is %s,global_step is %s,tra_acc is %.2f%%,i is %s"%(tra_loss,global_step,tra_acc*100.0,i))
            print("loss is %s,global_step is %s,tra_acc is %.2f%%"%(tra_loss,global_step,tra_acc*100.0))
            self.logging.info("loss is %s,global_step is %s,tra_acc is %.2f%%,i is %s"%(tra_loss,global_step,tra_acc*100.0,i))
            if i %1000==0:
               self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)
            

#    def test(self,test,test_label,i):
#        if self.model_type==1:
#            feed_dict={self.image_batch:test,self.image_label_batch:test_label}
#            loss,global_step,tes_acc=self.sess.run([self.loss,self.global_step,self.tra_accurate],feed_dict=feed_dict)
#            if i%2==0:
#                print("测试集：loss is %s,global_step is %s,tes_acc is %.2f%%,i is %s"%(loss,global_step,tes_acc*100.0,i))

    def tra_acc(self):
        with tf.variable_scope('train_accuracy'):     
            correct = tf.nn.in_top_k(self.softmax_linear, tf.argmax(self.image_label_batch,1), 1)      
            correct=tf.cast(correct,tf.float32) 
            self.tra_accurate=tf.reduce_mean(correct)
           
    
    def tes_acc(self,test,test_label,k):
        if k%2==0:
            if self.model_type==1:
                feed_dict={self.image_batch:test,self.image_label_batch:test_label}                
            elif self.model_type=="vgg16":
                feed_dict={self.image_batch:test, self.image_label_batch:test_label,self.keep_prob:0.8, self.is_training:False,self.is_use_l2:True,self.lam:0.001}
            elif self.model_type=="Alexnet":
                feed_dict={self.image_batch:test, self.image_label_batch:test_label,self.keep_prob:0.8}
            elif self.model_type=="Googlenet":
                feed_dict={self.image_batch:test, self.image_label_batch:test_label,self.keep_prob:0.4,self.is_training:True}
            loss,accurate = self.sess.run([self.loss, self.tra_accurate],feed_dict=feed_dict)
            print("epoch is %s,测试集：loss is %s,tes_acc is %.2f%%"%(k,loss,accurate*100.0))
       # with tf.variable_scope('test_accuracy'):
        #    predict=self.sess.run([self.softmax_linear],feed_dict={self.image_batch:test,
       #                           self.image_label_batch:test_label})
      #      predict=tf.cast(predict,tf.int32)
       #     right=0
      #      total=0
      #      for pre,rea in zip(predict,test):
     #           if max(pre)==rea:
      #              right +=1
     #           total +=1
     #       tes_acc=float(right)/total
    #        return tes_acc
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        