# -*- coding: utf-8 -*-
# http://www.robots.ox.ac.uk/~vgg/data/pets/
# 개 25종, 고양이 12종

#import tensorflow as tf
import numpy as np
import glob
import utils
import skimage
import matplotlib.pyplot as plt
import  os,datetime,time
from sys import getsizeof
import gzip, pickle, struct,sys
import matplotlib.pyplot as plt
import vgg16_trainable as vgg16
import tensorflow as tf
img_size = 224
# category 
def convert_imgs_to_npz():
    category_num = int(sys.argv[1])  # 지정한 category 1개만 npz로 만든다.
    filenames = glob.glob('.\\test_data\\VGG_Pet\\*.jpg')

#     category = []
#     for f in filenames:
#         category.append( "_".join(  f.split('\\')[-1].split('_')[:-1])) 
#     
#     category =list(set(category))
#     category.sort()
    
    category =['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    category_kor = ['아비시니아 고양이', '벵갈 고양이' '버만 고양이' '봄베이 고양이', '브리티시 쇼트헤어 고양이', '이집션 마우 고양이', '메인쿤 고양이', '페르시안 고양이', '래그돌 고양이', '러시안 블루 고양이','시암 고양이', '스핑크스 고양이', '아메리칸 불독 개', '아메리칸 핏불 테리어 개', '바셋하운드 개', '비글 개', '복서 개', '치와와 개' '잉글리시 코커 스패니얼 개' '잉글리시 세터', '저먼 쇼트헤어드 포인터','그레이트 피레니즈', '하바나 개', '제페니스 친 개', '키스혼드 개', '레온베르거 개', '미니어처 핀셔 개', '뉴펀들랜드 개', '포메라니안 개', '퍼그 개', '세인트버나드 개', '사모예드 개', '스코티시 테리어 개', '시바이누 개', '스타포드셔 불 테리어 개', '휘튼 테리어 개', '요크셔 테리어 개']
    
    # Map wnids to integer labels
    category_to_index = {ca: i for i, ca in enumerate(category)}
    index_to_category = {i:ca for i, ca in enumerate(category) }
    
    
    
   
    x_train = None
    y_train =np.array([],dtype = np.int)    
    
    category_filenames = [x for x in filenames if index_to_category[category_num] in x]
    
    
    for file in category_filenames:
        img = utils.load_image(file,img_size=img_size,float_flag=False)
        if img.shape == (img_size,img_size,3):
            img = img.reshape((1, img_size, img_size, 3))
            if x_train is None:
                x_train=img
            else:
                x_train = np.concatenate((x_train,img),0)
            y_train = np.append(y_train,category_num) 
    
    if x_train is not None:
        output_filename = 'VGG_Pet_int_' + str(category_num) + '.npz'
        np.savez_compressed(output_filename, x_train=x_train, y_train=y_train) 
        print(len(y_train), getsizeof(x_train))        
    


def concat_shuffle_npz():
    
    n_split = 10
    
    filenames = glob.glob('.\\test_data\\VGG_Pet\\*.jpg')

    category = []
    for f in filenames:
        category.append( "_".join(  f.split('\\')[-1].split('_')[:-1])) 
    
    category =list(set(category))
    category.sort()
    
    # Map wnids to integer labels
    category_to_index = {ca: i for i, ca in enumerate(category)}
    index_to_category = {i:ca for i, ca in enumerate(category) }
    index_to_category_list = [[int(a),b] for a,b in index_to_category.items()]
    
    
    npz_filenames = []
    for i in range(37):
        npz_filenames.append('VGG_Pet_int_'+str(i)+'.npz')
 
 
    x_train = None
    y_train = None
    for i in range(len(npz_filenames)):
        print(i)
        data = np.load(npz_filenames[i])
        x_train_temp = data['x_train']
        y_train_temp = data['y_train']
        if x_train_temp.shape == (): continue
        if x_train is None:
            x_train = x_train_temp
            y_train = y_train_temp
        else:
            x_train = np.concatenate((x_train,x_train_temp),0)
            y_train = np.concatenate((y_train,y_train_temp),0)
 
    np.savez_compressed("VGG_Pet_int__All.npz", x_train = x_train,y_train=y_train)
    print(x_train.shape)
    print(getsizeof(x_train))

    # shuffle
#     data = np.load("Caltech101_int_All.npz")
#     x_train = data['x_train']
#     y_train = data['y_train']    
    
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)
    x_train = x_train[s]
    y_train = y_train[s]    
    
    imgs_per_shffle = int(len(x_train)/n_split)
    
    for i in range(n_split):
        print(i)
        np.savez_compressed("VGG_Pet_int_shuffle_" + str(i), 
                            x_train = x_train[i*imgs_per_shffle:(i+1)*imgs_per_shffle], 
                            y_train = y_train[i*imgs_per_shffle:(i+1)*imgs_per_shffle],ind = index_to_category_list )
    


def test_npz():
    data = np.load('.\\test_data\\VGG_Pet_int_shuffle_4.npz')

    x_train = data['x_train']
    y_train = data['y_train']
    ind = data['ind']
    index_to_category = {int(c[0]):c[1] for i,c in enumerate(ind)}
    print(x_train.shape)

    choice = 200

    plt.figure()
    plt.title('correct answer: ' + index_to_category[y_train[choice]])
    plt.imshow(x_train[choice]/255.0)
    plt.show()
    plt.close()

def train_from_pretrained():
    n_class = 37
    category =['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    category_kor = ['아비시니아 고양이', '벵갈 고양이', '버만 고양이', '봄베이 고양이', '브리티시 쇼트헤어 고양이', '이집션 마우 고양이', '메인쿤 고양이', '페르시안 고양이', '래그돌 고양이', '러시안 블루 고양이','시암 고양이', '스핑크스 고양이', '아메리칸 불독 개', '아메리칸 핏불 테리어 개', '바셋하운드 개', '비글 개', '복서 개', '치와와 개', '잉글리시 코커 스패니얼 개', '잉글리시 세터', '저먼 쇼트헤어드 포인터','그레이트 피레니즈', '하바나 개', '제페니스 친 개', '키스혼드 개', '레온베르거 개', '미니어처 핀셔 개', '뉴펀들랜드 개', '포메라니안 개', '퍼그 개', '세인트버나드 개', '사모예드 개', '스코티시 테리어 개', '시바이누 개', '스타포드셔 불 테리어 개', '휘튼 테리어 개', '요크셔 테리어 개']
    
    data = np.load('.\\test_data\\VGG_Pet_int_All.npz')

    x_train = data['x_train']
    y_train = data['y_train']

    
    y_train_one_hot = np.zeros((y_train.size, n_class),dtype=np.int)
    y_train_one_hot[np.arange(y_train.size),y_train] = 1
    
    
    batch_size =10
    cost_all = []


    sess = tf.Session()

    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.float32, [None, n_class])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy',n_class = n_class)
    vgg.build(images, train_mode,int_image=True)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print(vgg.get_var_count())

    sess.run(tf.global_variables_initializer())



    # simple 1-step training
    cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    for step in range(100000):
        batch_mask = np.random.choice(x_train.shape[0],batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train_one_hot[batch_mask]
        _, c = sess.run([train,cost], feed_dict={images: x_batch, true_out: y_batch, train_mode: True})
        if step % 1000 ==0:
            cost_all.append([step,c])
            print(step,c)
        
    np.savetxt("cost.txt",cost_all)
    # test classification again, should have a higher probability about tiger
    
    # test accuracy 계산
    n_sample =50
    n = int(x_train.shape[0]/n_sample)
    accuracy = 0
    for i in range(n):
        prob = sess.run(vgg.prob, feed_dict={images: x_train[i*50:(i+1)*50], train_mode: False})
        prob = np.argmax(prob,axis=1)
        y_batch = y_train[i*50:(i+1)*50]
        accuracy += np.sum(prob == y_batch) / float(n_sample)
    print("accuracy: ", accuracy/n)
    
    

    
    #plt.figure()
    #plt.title(category[y_batch[0]])
    #plt.imshow(x_batch[0]/255.0)
    #plt.show()
    #plt.close()        
    #utils.print_prob(prob[0], './synset.txt')

    # test save
    vgg.save_npy(sess, './test-save.npy')    
        
def predict_from_pretrained():
    n_class = 37
    category =['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    category_kor = ['아비시니아 고양이', '벵갈 고양이', '버만 고양이', '봄베이 고양이', '브리티시 쇼트헤어 고양이', '이집션 마우 고양이', '메인쿤 고양이', '페르시안 고양이', '래그돌 고양이', '러시안 블루 고양이','시암 고양이', '스핑크스 고양이', '아메리칸 불독 개', '아메리칸 핏불 테리어 개', '바셋하운드 개', '비글 개', '복서 개', '치와와 개', '잉글리시 코커 스패니얼 개', '잉글리시 세터', '저먼 쇼트헤어드 포인터','그레이트 피레니즈', '하바나 개', '제페니스 친 개', '키스혼드 개', '레온베르거 개', '미니어처 핀셔 개', '뉴펀들랜드 개', '포메라니안 개', '퍼그 개', '세인트버나드 개', '사모예드 개', '스코티시 테리어 개', '시바이누 개', '스타포드셔 불 테리어 개', '휘튼 테리어 개', '요크셔 테리어 개']
    
    test_source = 2
    
    if test_source == 1:
        filenames = ['./test_data/Pet_test_Bombay.jpg', './test_data/Pet_test_keeshond.jpg','./test_data/Pet_test_Ragdoll.jpg',
                     './test_data/Pet_test_scottish_terrier.jpg','./test_data/Pet_test_english_setter.jpg']
        label = [3,24,8,32,19]
        
        test_batch=None
        for i in range(len(filenames)):
            img = utils.load_image(filenames[i],img_size=img_size,float_flag=False).reshape((1,img_size,img_size,3))
            if test_batch is None:
                test_batch = img
            else:
                test_batch = np.concatenate((test_batch,img),0)
    elif test_source == 2:
        
        data = np.load('.\\test_data\\VGG_Pet_int_All.npz')
        
        test_batch = data['x_train']
        label = data['y_train']
        
        
        ndata = min(50, test_batch.shape[0])
        batch_mask = np.random.choice(test_batch.shape[0],ndata)
        
        test_batch = test_batch[batch_mask]
        label = label[batch_mask]
        
        
        
    
    
    
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        train_mode = tf.placeholder(tf.bool)
        vgg = vgg16.Vgg16('./vgg16_Pet_Trained_Weight_GPU.npy',trainable = False,n_class = n_class)
        with tf.name_scope("content_vgg"):
            vgg.build(images,train_mode,int_image=True)

        prob = sess.run(vgg.prob, feed_dict = {train_mode: False, images: test_batch})
        #print(prob)
        prob_argmax = np.argmax(prob,axis=1)
        for i in range(min(100,test_batch.shape[0])):
            print(i, "correct: ", category[label[i]], "predict: ", category_kor[prob_argmax[i]],label[i] ==prob_argmax[i] )

        print("acc: ", np.sum(label==prob_argmax)/float(test_batch.shape[0]))
    
        
if __name__ == "__main__":
    s = time.time()
    #convert_imgs_to_npz()

    #concat_shuffle_npz()
    
    #test_npz();
    
    train_from_pretrained()
    #predict_from_pretrained()
    
    
    e=time.time()
    
    print("경과시간: ", e-s, "sec")




