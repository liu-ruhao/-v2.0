
import model as m
import input_data 
import logging
import config as CF
import tensorflow as tf

train, train_label, val, val_label,n_class = input_data.get_files(CF.config["train_dir"], CF.config["ratio"])
CF.config["N_CLASSES"]=n_class
#train_batch,train_label_batch=input_data.get_batch(train, train_label,CF.config["BATCH_SIZE"], CF.config["CAPACITY"],CF.config["num"])
test=input_data.give_batch_test(val,CF.config["num"])
#print(test,val_label)
train_batch,train_label_batch=input_data.give_batch(train,train_label,CF.config["BATCH_SIZE"],CF.config["num"])

#get_batch
print(CF.config["BATCH_SIZE"])

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m_%d %H:%M:%S",
                    filename=CF.config["logging_name"],
                    filemode="w")
sess=tf.Session()
model_1=m.model(sess=sess,config=CF.config,logging=logging)
model_1.print_var()

for k in range(CF.config["MAX_STEP"]):
    for step, batch_x_batch_y in enumerate(zip(train_batch,train_label_batch)):
        batch_x,batch_y=batch_x_batch_y
        model_1.train(batch_x,batch_y,step)
        #print(step, batch_x_batch_y)
        #exit()
    model_1.tes_acc(test,val_label,k)
    #model_1.test(test,val_label,k)





'''

for k in range(CF.config["MAX_STEP"]):
    for batch_x_batch_y in zip(train_batch,train_label_batch):
        batch_x,batch_y=batch_x_batch_y
        model_1.train(batch_x,batch_y,k)
    model_1.test(test,val_label,k)
    
    
'''

'''
for k,(batch_x,batch_y) in enumerate(zip(train_batch,train_label_batch)):           
        model_1.train(batch_x,batch_y,k)
        model_1.test(test,val_label,k)
'''













