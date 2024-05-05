import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import os 
from PIL import Image
from Dataset import Dataset
from Model import CGAN


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

def Loader():
            
    absolute_path = os.path.dirname(__file__)


    ################################################################################################################################
    del_all_flags(tf.flags.FLAGS)
    flags = tf.flags.FLAGS
    tf.flags.DEFINE_integer('train_interval', 1, 'training interval between discriminator and generator, default: 1')
    tf.flags.DEFINE_integer('ratio_gan2seg', 20, 'ratio of gan loss to seg loss, default: 10')
    tf.flags.DEFINE_string('discriminator', 'image', 'type of discriminator [pixel|patch1|patch2|image],default: image')
    tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
    tf.flags.DEFINE_float('learning_rate', 2e-4, 'initial learning rate for Adam, default: 2e-4')
    tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of adam, default: 0.5')
    tf.flags.DEFINE_integer('iters', 50000, 'number of iteratons, default: 50000')
    tf.flags.DEFINE_integer('print_freq', 100, 'print frequency, default: 100')
    tf.flags.DEFINE_integer('eval_freq', 500, 'evaluation frequency, default: 500')
    tf.flags.DEFINE_integer('sample_freq', 200, 'sample frequency, default: 200')
    tf.flags.DEFINE_integer('save_freq', 4000, 'save model frequency, default: 4000')
    tf.flags.DEFINE_integer('lambda1', 100, 'ratio seg loss')
    tf.flags.DEFINE_integer('lambda2', 1, 'ratio gan loss')

    tf.flags.DEFINE_string('gpu_index', '2', 'gpu index, default: 0')

    tf.flags.DEFINE_bool('is_test', True, 'default: False (train)')

    tf.flags.DEFINE_string('test_dir','./DataInput/','test input file dir') #Input Dir ###############
    tf.flags.DEFINE_string('model_dir','./ModelDir/','model checkpoint file dir')
    tf.flags.DEFINE_string('output_dir','./test_result_output_dir','test output dir') #output Dir ##########

    tf.flags.DEFINE_bool('is_convert',False,'is_convert')

    tf.flags.DEFINE_integer('mn',3, 'all model checkpoint paths index')
    tf.flags.DEFINE_integer('fn',4000, 'test_output_folder_name')
    tf.flags.DEFINE_bool('is_single','False','is_single')
    tf.flags.DEFINE_string('input_file','input_file.png','if is_single==True, convert_input_file_dir')
    ################################################################################################################################
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session()

    flags = flags
    dataset = Dataset(flags)
    model = CGAN(sess, flags, dataset.image_size)

    score = 0.    #best auc x
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ################################################################################################################################





    # Path to the directory containing your model files
    #model_dir = '/mnt/d/AIrepos/Loader/ModelDir/'



    # Start a new TensorFlow session

    print(' [*] Reading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(flags.model_dir)
    all_models = ckpt.all_model_checkpoint_paths
    #print(all_models[0])
    if ckpt and all_models:
        ckpt_name = os.path.basename(all_models[flags.mn])
        saver.restore(sess, os.path.join(flags.model_dir, ckpt_name))

    imgs = dataset.test_imgs
    for idx in range(len(imgs)):
        x_img = imgs[idx]
        x_img = x_img.reshape(1, x_img.shape[0], x_img.shape[1], 1)
        generated_label = model.sample_imgs(x_img)
        generated_label = np.squeeze(generated_label, axis=(0,3))


        Image.fromarray(np.asarray(generated_label*255).astype(np.uint8)).save('./Output/denoised_image-'+str(idx)+'.png')

    




    # Image.fromarray(np.asarray(generated_label*255).astype(np.uint8)).save('./Output/denoised_image-'+str(idx)+'.png')





    #cd ../../..;cd d/AIrepos/SAGAN2/conditionalGAN-tensorflow-python/src/;conda activate TEST-Lua;conda activate tf;


    #python Loadmodel.py --is_test=True --test_dir='/mnt/d/AIrepos/SAGAN2/conditionalGAN-tensorflow-python/data/test/' --model_dir='/mnt/d/AIrepos/SAGAN2/conditionalGAN-tensorflow-python/src/output_dir/model_gan*1+seg*100/' --output_dir='./test_result_output_dir' --mn=4 --fn=3232