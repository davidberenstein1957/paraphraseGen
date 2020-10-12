import argparse
import os
import time
from random import randint

import numpy as np
import torch as t
from torch.optim import Adam
from torch.autograd import Variable

from utils.batch_loader import BatchLoader
from utils.parameters import Parameters
from utils.functional import *
# from model.rvae_previous import RVAE
from model.rvae import RVAE


if __name__ == "__main__":
    path='paraphraseGen/'
    save_path='/content/drive/My Drive/thesis/'
    if not os.path.exists(path+'data/word_embeddings.npy'):
        raise FileNotFoundError("word embeddings file was't found")
#一次一句，这样容易看，一次两个词
    parser = argparse.ArgumentParser(description='RVAE')
    parser.add_argument('--num-iterations', type=int, default=120000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--use-trained', type=bool, default=False)
    parser.add_argument('--attn-model', type=bool, default=True)

    parser.add_argument('--use-file', type=bool, default=True)
    parser.add_argument('--test-file', type=str, default= path+'/data/test.txt')
    parser.add_argument('--train-file', type=str, default= path+'/data/train.txt')

    parser.add_argument('--num-sample', type=int, default=5)
    parser.add_argument('--beam-top', type=int, default=1)
    parser.add_argument('--beam-size', type=int, default=10)

    parser.add_argument('--ce-result', default='')
    parser.add_argument('--kld-result', default='')
    parser.add_argument('--model-result', default='')
    

    args = parser.parse_args()

    
    ''' =================== Creating batch_loader for encoder-1 =========================================
    '''
    data_files = [path + 'data/train.txt',
                       path + 'data/test.txt']

    idx_files = [path + 'data/words_vocab.pkl',
                      path + 'data/characters_vocab.pkl']

    tensor_files = [[path + 'data/train_word_tensor.npy',
                          path + 'data/valid_word_tensor.npy'],
                         [path + 'data/train_character_tensor.npy',
                          path + 'data/valid_character_tensor.npy']]

    batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size,
                            args.attn_model)


    ''' =================== Doing the same for encoder-2 ===============================================
    '''
    data_files = [path + 'data/super/train_2.txt',
                       path + 'data/super/test_2.txt']

    idx_files = [path + 'data/super/words_vocab_2.pkl',
                      path + 'data/super/characters_vocab_2.pkl']

    tensor_files = [[path + 'data/super/train_word_tensor_2.npy',
                          path + 'data/super/valid_word_tensor_2.npy'],
                         [path + 'data/super/train_character_tensor_2.npy',
                          path + 'data/super/valid_character_tensor_2.npy']]
    batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
    parameters_2 = Parameters(batch_loader_2.max_word_len,
                            batch_loader_2.max_seq_len,
                            batch_loader_2.words_vocab_size,
                            batch_loader_2.chars_vocab_size,
                            args.attn_model)
    '''=================================================================================================
    '''
    modulo_operator = len(open(args.train_file, 'r').readlines())
    text_input =''
    if not args.use_file:
        text_input = input("Input Question : ")
    else:
        file_1 = open(args.test_file, 'r')
        data = file_1.readlines()

    rvae = RVAE(parameters,parameters_2, path)
    if args.use_trained:
        rvae.load_state_dict(t.load('trained_RVAE'))
    if args.use_cuda:
        rvae = rvae.cuda()

    optimizer = Adam(rvae.learnable_parameters(), args.learning_rate)

    train_step = rvae.trainer(optimizer,batch_loader, batch_loader_2)# batchloader里面是原始句子，batechloader2里面存储的是释义句
    validate = rvae.validater(batch_loader,batch_loader_2)

    ce_result = []
    kld_result = []

    start_index = 0
    start_time = time.time()
    
    for iteration in range(args.use_trained, args.num_iterations):
        coef = kld_coef(iteration, 0)
        if coef == 1:
            coef_modulo = iteration
            break

    if int(args.num_iterations/coef_modulo) % 2 == 0:
        args.num_iterations = int(args.num_iterations/coef_modulo) * coef_modulo
    else:
        args.num_iterations = (int(args.num_iterations/coef_modulo)+1) * coef_modulo


    for iteration in range(args.num_iterations):
        #This needs to be changed ##这一步必须保证不大于训练数据数量-每一批数据的大小，否则越界报错######################
        start_index = (start_index+args.batch_size)%(modulo_operator)
        #start_index = (start_index+args.batch_size)%149163 #计算交叉熵损失，等
        
        coef = kld_coef(iteration, coef_modulo)
        
        cross_entropy, kld, _ = train_step(coef, args.batch_size, args.use_cuda, args.dropout, start_index)
       
        if ((iteration % coef_modulo == 0) & (iteration != 0)):
            print('\n')
            print('------------TRAIN-------------')
            print('-------------ETA--------------')
            percentage = (iteration/args.num_iterations)*100
            time_spend = time.time()-start_time
            print(time_spend)
            time_required = (time_spend/percentage)*(100-percentage)
            time_required_min = time_required/60
            print(time_required_min)
            print('----------ITERATION-----------')
            print(iteration, round(percentage,2))
            print('--------CROSS-ENTROPY---------')
            print(cross_entropy.data.cpu().numpy())
            print('-------------KLD--------------')
            print(kld.data.cpu().numpy())
            print('-----------KLD-coef-----------')
            print(coef)
            print('------------------------------')
        
        if ((iteration % coef_modulo == 0) & (iteration != 0)):
            t.save(rvae.state_dict(), save_path+f'/trained_RVAE_{iteration}')
            np.save(save_path+f'/ce_result_{iteration}.npy'.format(args.ce_result), np.array(ce_result))
            np.save(save_path+f'/kld_result_npy_{iteration}'.format(args.kld_result), np.array(kld_result))
            print('MODEL SAVED')

        # if ((iteration % int(coef_modulo/10) == 0)):
        if iteration % 10 == 0:
            index = randint(0, len(data)-1)

            if args.use_file:
                print ('original sentence:     '+data[index])
            else:
                print ('original sentence:     '+text_input + '\n')

            for _ in range(args.num_sample):

                seed = Variable(t.randn([1, parameters.latent_variable_size]))
                seed = seed.cuda()

                results, scores = rvae.sampler(batch_loader, batch_loader_2, 50, seed, args.use_cuda, index, args.beam_size, args.beam_top)
                
                for tt in results:
                    
                    for k in range(args.beam_top):
                        sen = " ". join([batch_loader_2.decode_word(x[k]) for x in tt])
                        if batch_loader.end_token in sen:    
                            print ('generate sentence:     '+sen[:sen.index(batch_loader.end_token)])
                        else :
                            print ('generate sentence:     '+sen) 

        

    t.save(rvae.state_dict(), 'trained_RVAE')

    np.save('ce_result_{}.npy'.format(args.ce_result), np.array(ce_result))
    np.save('kld_result_npy_{}'.format(args.kld_result), np.array(kld_result))
