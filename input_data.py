import os
import glob
import config as CF
import numpy as np
from PIL import Image
#import tensorflow as tf

#通过将训练集按照指定格式放入input_data文件夹下，运行get_file函数，给出训练集和测试集图片的路径。
def to_onehot(num,length):
    ret=[0]*length
    ret[num-1]=1
    return ret
def get_files(file_dir="./input_data/",ratio=0.9):
    #print(CF.config)
    print("调用数据get_file")
    if glob.glob(file_dir+"label2id")!=[]:
        print("删除遗留的label2id文件")
        os.remove(file_dir+"label2id")
    label_files=glob.glob(file_dir+"*")
    #print(label_files)
    label_lst=[x.split("\\")[-1] for x in label_files]
    total_data=[]
    t_label2id={}
    #print(label_lst)
    for idd,label in enumerate(label_lst):
        temp={}      
        temp["id"]=int(idd)       
        temp["num"]=0   
        t_label2id[label]=temp
            
        path_label=file_dir+label
        #print(path_label)
        path_image=glob.glob(path_label+"/*")
        #print(path_image)
        for path in path_image:
            total_data.append([path,label])
            #print(total_data)
            t_label2id[label]["num"]+=1
    print(t_label2id)
            #print(t_label2id[label]["num"])
    print("数据一共有%s个，一共有%s个类别，其中:"%(len(total_data),len(label_lst)))
    with open(file_dir+"label2id","w",encoding="utf-8") as f:
        for label,temp in t_label2id.items():
            idd=temp["id"]
            num=temp["num"]
            print("类别%s的id是%s，有样本%s个！"%(label,idd,num))
            f.write("%s\t%s\t%s\n"%(label,idd,num))
    train_data=[]
    test_data=[]
    for path_label in total_data:#?
        #print(path_label)
        path,label=path_label
        if np.random.random()>ratio:
            test_data.append([path,t_label2id[label]["id"]])
        else:
            train_data.append([path,t_label2id[label]["id"]])
    n_class=len(label_lst)
    tra_images = [x[0] for x in train_data]
    tra_labels = [to_onehot(x[1],n_class) for x in train_data]
    val_images = [x[0] for x in test_data]
    val_labels = [to_onehot(x[1],n_class) for x in test_data]
    
    #print(label_lst)
    #print(len(label_lst))
    #print(tra_images,tra_labels,val_images,val_labels)
    #print(val_images)
    print(tra_images)
    return tra_images,tra_labels,val_images,val_labels,n_class

def give_batch(paths,labels,batch_size,k):
    #if k==1:
        x_batch=[]
        for path in paths:
            img = Image.open(path)
        #plt.imshow(img)
        #plt.show()
            imag = img.resize([64,64])
            image = np.array(imag)
            #print(image)
            #print(image.shape)
            x_batch.append(image)
            #print(x_batch)
        #print(len(x_batch))
        num_cut=int(len(x_batch)/batch_size)
        #print(num_cut)
        ret_x=[]
        ret_y=[]
        for i in  range(num_cut):
            ret_x.append(x_batch[i*batch_size:(i+1)*batch_size])
            ret_y.append(labels[i*batch_size:(i+1)*batch_size])
        #按照 batchsize 分开  
        return ret_x,ret_y


def give_batch_test(paths,k):
    #if k==1:
        x_batch=[]
        for path in paths:
            img = Image.open(path)
        #plt.imshow(img)
        #plt.show()
            imag = img.resize([64, 64])
            image = np.array(imag)
            #print(image)
            #print(image.shape)
            x_batch.append(image)
            #print(x_batch)
        return x_batch



if __name__=="__main__":
    train,train_label,val,val_label,n_class=get_files(file_dir="./input_data/",ratio=0.9)
    batch_x,batch_y=give_batch(val,val_label,CF.config["BATCH_SIZE"],CF.config["num"])
   # train_batch,train_label_batch=get_batch(train,train_label,CF.config["BATCH_SIZE"],CF.config["CAPACITY"],CF.config["num"])
    #print(train_batch,train_label_batch)
    #print(train_label)
    #print(batch_y)
    #shape_x=np.array(batch_x)
    #print(shape_x.shape)
    #print(len(batch_x),len(batch_y))
    #print(len(batch_x[0]))
    #print(len(batch_x[0][0]))
    #print(len(batch_x[0][0][0]))
    #print(len(batch_x[0][0][0][0]))
    #print(batch_x[0][0])
    #batch_x,batch_y=give_batch(train,train_label,128,64,64)
    
    #x_batch,y_batch=give_batch(val,val_label,CF.config["BATCH_SIZE"],64,64)
    #print(x_batch,y_batch)
    #train_batch, train_label_batch = get_batch(train,train_label,20,200,CF.config["num"])
    #print(len(train))
    #print(len(train_label))    
    
'''
    if k=="vgg16":
        x_batch=[]
        for path in paths:
            img = Image.open(path)
        #plt.imshow(img)
        #plt.show()
            imag = img.resize([224,224])
            image = np.array(imag)
            #print(image)
            #print(image.shape)
            x_batch.append(image)
            #print(x_batch)
        #print(len(x_batch))
        num_cut=int(len(x_batch)/batch_size)
        #print(num_cut)
        ret_x=[]
        ret_y=[]
        for i in  range(num_cut):
            ret_x.append(x_batch[i*batch_size:(i+1)*batch_size])
            ret_y.append(labels[i*batch_size:(i+1)*batch_size])
        #按照 batchsize 分开  
        return ret_x,ret_y
'''
'''
    if k=="vgg16":
        x_batch=[]
        for path in paths:
            img = Image.open(path)
        #plt.imshow(img)
        #plt.show()
            imag = img.resize([224, 224])
            image = np.array(imag)
            #print(image)
            #print(image.shape)
            x_batch.append(image)
            #print(x_batch)
        return x_batch
'''
'''
def get_batch(image,label,batch_size,capacity,k):
    if k==1:
        image=tf.cast(image,tf.string)
        label=tf.cast(label,tf.int32)
        input_queue=tf.train.slice_input_producer([image,label])#?
        #input_queue = tf.data.Dataset.from_tensor_slices([image, label])
        #print(input_queue)
        label=input_queue[1]
        #print(label)
        image_contents=tf.read_file(input_queue[0])
        #print(image_contents)
        image=tf.image.decode_jpeg(image_contents,channels=3)
        image=tf.image.resize_image_with_crop_or_pad(image,64,64)
        image=tf.image.per_image_standardization(image)
        #print(image)
        image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=32,capacity=capacity)
        #print(image_batch)
        #重新排列label，行数为[batch_size]
        label_batch=tf.reshape(label_batch,[batch_size])
        #print(label_batch)
        image_batch=tf.cast(image_batch,tf.float32)
        #print(image_batch)
        return image_batch,label_batch
    elif k=="vgg16":
            image=tf.cast(image,tf.string)
            label=tf.cast(label,tf.int32)
            input_queue=tf.train.slice_input_producer([image,label])#?
            #input_queue = tf.data.Dataset.from_tensor_slices([image, label])
            label=input_queue[1]
            image_contents=tf.read_file(input_queue[0])
            image=tf.image.decode_jpeg(image_contents,channels=3)
            image=tf.image.resize_image_with_crop_or_pad(image,224,224)
            image=tf.image.per_image_standardization(image)
            
            image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=32,capacity=capacity)
            
            #重新排列label，行数为[batch_size]
            label_batch=tf.reshape(label_batch,[batch_size])
            image_batch=tf.cast(image_batch,tf.float32)
            #print(image_batch)
            return image_batch,label_batch
'''    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        