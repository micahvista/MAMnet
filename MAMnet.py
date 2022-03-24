'''   
 * @Title: MAMnet 
 * @author: Ding Hongyu
 * @date: Jan 5th 2022
 * @version V1.0.0
'''

import pandas as pd

from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape, multiply, add, Activation
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from collections import Counter

import sys





#tf.config.set_visible_devices([], 'GPU')
class se(tf.keras.layers.Layer):
    
    def __init__(self,filters, se_ratio = 0.25):
        CONV_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 2.0,
            'mode': 'fan_out',
            'distribution': 'normal'
            }
        }

        DENSE_KERNEL_INITIALIZER = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 1. / 3.,
                'mode': 'fan_out',
                'distribution': 'uniform'
            }
        }
        super(se, self).__init__()

        
        
        self.filters = filters

        
        self.filters_se = max(1, int(filters*se_ratio))
        self.conv_1 = Conv2D(filters=self.filters_se,
                    kernel_size=1,
                    padding='same',
                    kernel_initializer= CONV_KERNEL_INITIALIZER,
                    use_bias=True)
        self.conv_2 = Conv2D(filters=self.filters,
                    kernel_size=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer= CONV_KERNEL_INITIALIZER,
                    use_bias=True)

    def call(self, inputs):
        



        se = GlobalAveragePooling2D()(inputs)
        se = Reshape((1, 1, self.filters))(se)

        se = self.conv_1(se)

        se = Activation(tf.nn.swish)(se)

        se = self.conv_2(se)

        return multiply([inputs, se])
    def compute_output_shape(self, input_shape):
        return input_shape

def main_model(mc = False):

    inputs = tf.keras.Input(shape=(None, 1800))
    x = tf.keras.layers.TimeDistributed(Reshape((200, 9, 1)))(inputs)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(136,kernel_size=(1,9),strides=(1,9),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(136))(x)
    


    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(3,1),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(3,1),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    
    x = tf.keras.layers.TimeDistributed(Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)

    x = tf.keras.layers.TimeDistributed(layers.Activation('relu'))(x)
    x = tf.keras.layers.TimeDistributed(layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(se(72))(x)
    

    

    
    encoded_frames = tf.keras.layers.TimeDistributed(layers.Flatten())(x)
    encoded_frames = layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences= True))(encoded_frames)
    x = layers.BatchNormalization()(encoded_frames)

    x = tf.keras.layers.Dense(units=64, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(0.4)(x, training = mc)
    x = tf.keras.layers.Dense(units=64, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(0.4)(x, training = mc)

    outputs = tf.keras.layers.Dense(units=2, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)

def geno_model(mc = False):

    inputs = tf.keras.Input(shape=(1800))
    x = (Reshape((200, 9, 1)))(inputs)
    
    x = (Conv2D(136,kernel_size=(1,9),strides=(1,9),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(136))(x)
    


    
    x = (Conv2D(72,kernel_size=(3,1),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    
    x = (Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    
    x = (Conv2D(72,kernel_size=(3,1),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    
    x = (Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    
    x = (Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)
    x = (tf.keras.layers.MaxPool2D((2, 1)))(x)
    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    
    x = (Conv2D(72,kernel_size=(2,1),padding='valid', activation = None))(x)

    x = (layers.Activation('relu'))(x)
    x = (layers.BatchNormalization())(x)
    x = (se(72))(x)
    

    

    
    encoded_frames = (layers.Flatten())(x)


    x = tf.keras.layers.Dense(units=64, activation=None)(encoded_frames)
    x = layers.BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(0.4)(x, training = mc)
    x = tf.keras.layers.Dense(units=64, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = Activation("relu")(x)
    x = layers.Dropout(0.4)(x, training = mc)

    outputs = tf.keras.layers.Dense(units=3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


import multiprocessing
import os
import pyalign
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Value, Array
import time
from numba.typed import List
import pysam
from pysam import VariantFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mamnet



def logify(a):
    return (tf.math.log((tf.cast((a>0), dtype = tf.float32)*a)+1.)-tf.math.log((tf.abs(a)*tf.cast((a<0), dtype = tf.float32))+1.))

def logify_numpy(a):
    return (np.log(((a>0)*a)+1.)-np.log((np.abs(a)*(a<0))+1.))

def baseinfo_AlignedSegment(genotype, bamfilepath, contig, r_start, r_end, meanvalue, window_size, maxcountread, workdir, INTERVAL):

    for AlignedSegment in pysam.AlignmentFile(bamfilepath, 'rb').fetch(contig, r_start, r_end):
        reference_start, reference_end = AlignedSegment.reference_start, AlignedSegment.reference_end
        start = reference_start
        preend = reference_end
        break

    cigarlist = List()
    mdtaglist = List()
    corposlist = List()
    primaryreadidcontigandsa = List()
    primaryssee = List()
    qualityarray = []
    sapresent = False
    

        
    
    
    for AlignedSegment in pysam.AlignmentFile(bamfilepath, 'rb').fetch(contig, r_start, r_end):
        reference_start, reference_end = AlignedSegment.reference_start, AlignedSegment.reference_end

            
        preend = reference_end    
        qualityarray.append([AlignedSegment.query_alignment_length/ AlignedSegment.infer_read_length(), AlignedSegment.query_alignment_length, AlignedSegment.mapping_quality])
        mdtaglist.append(AlignedSegment.get_tag('MD'))
        cigarlist.append(AlignedSegment.cigarstring)
        corposlist.append(List([reference_start - start, reference_end - start]))
        strandcode = decode_flag(AlignedSegment.flag)
        contig = AlignedSegment.reference_name
        readid = AlignedSegment.query_name
        if(AlignedSegment.has_tag('SA') == True):

            sapresent = True
            if((strandcode%2) == 0):
                strandcode = 2
            else:
                strandcode = 1
            primaryreadidcontigandsa.append(List([readid, contig, str(strandcode), AlignedSegment.get_tag('SA')]))
            refstart, refend, readstart, readend =  AlignedSegment.reference_start, AlignedSegment.reference_end, AlignedSegment.query_alignment_start, AlignedSegment.query_alignment_end
            primaryssee.append(List([refstart, refend, readstart, readend]))
        
    while(len(mdtaglist) != 0):

        if(os.path.isfile(workdir+'meanarray.npy') == True):
                
            meanvalue = np.load(workdir+'meanarray.npy').astype('float32')
            maxcountread = closerone(meanvalue[0][-1])
            try:
                baseinfo_AlignedSegment_child(sapresent, qualityarray, mdtaglist, cigarlist, corposlist, contig, start, start+(((preend-start)//window_size) + 1)*window_size, primaryreadidcontigandsa, primaryssee, maxcountread, window_size, meanvalue, workdir)
            except:
                print('error on ', contig, r_start, r_end)
            break
        else:
            time.sleep(2)


def baseinfo_AlignedSegment_child(sapresent, qualityarray, mdtaglist, cigarlist, corposlist, contig, start, end, primaryreadidcontigandsa, primaryssee, maxcountread, window_size, meanvalue, workdir):     
    

    qualityarray = np.array(qualityarray, dtype = 'float32')
    qualityarray = qualityarray - qualityarray.min(axis = 0)
    qualityarray = qualityarray / qualityarray.max(axis = 0)
    
    if(sapresent == True):

        data, cluster_result, cluster_readcount = mamnet.c_cw(mdtaglist, cigarlist, corposlist, start, end-start, primaryreadidcontigandsa, primaryssee, 'victory', np.argsort(qualityarray.sum(axis = 1))[::-1], maxcountread)
        data = data.reshape(((end-start) // window_size, 9 * window_size))
    else:

        data, cluster_result, cluster_readcount = mamnet.c_cn(mdtaglist, cigarlist, corposlist, start, end-start, primaryreadidcontigandsa, primaryssee, 'victory')
        data = data.reshape(((end-start) // window_size, 9 * window_size))
    
    mask = (data.sum(axis = 1) != 0)
    index = np.arange(start, end, window_size)[mask]

    data = logify_numpy((data.reshape(data.size // 9, 9) - meanvalue).reshape(data.shape))[mask].astype('float16')
                
    bp = np.array([[cluster_result[loc][0] + start, cluster_result[loc][1], cluster_readcount[loc]] for loc in range(len(cluster_result))])
    try:
        np.savez_compressed(workdir+str(contig) +':'+str(start) +':'+str(end), data = data, index = index, bp = bp)
        np.savez_compressed(workdir+'tmp/'+str(contig) +':'+str(start) +':'+str(end), data = np.array(0), index = np.array(0))
        print(str(contig) +':'+str(start) +':'+str(end)+'saved') 
    except:
        print()
        print()
        print('#'*30)
        print('File save failed at '+str(contig) +':'+str(start) +':'+str(end))
        print('#'*30)
        print()
        print()

    return 0

    
    
def call_deletion(svlist, contig, index, predict, window_size = 200):
    

    loc = -1
    insv = False
    for p in predict:
        loc += 1
        if(p > 0):
            if(insv == False):
                svstartloc = index[loc] + window_size//2
                qc = p

                insv = True
            waitstep = 0
            qc = max(qc, p)



        else:
            if(insv == True):
                if(waitstep < 0):
                    waitstep += 1

                    continue
                svlist.append(['DEL', contig, svstartloc, index[loc], qc, 1])
                insv = False

                waitstep = 0
                continue



     
    return svlist

def call_Insertion(svlist, contig, index, predict, window_size = 200):


    loc = 0
    insv = False
    for p in predict:
        if(p > 0):
            if(insv == False):
    
                qc = p
                insv = True
                svstartloc = index[loc]

            qc = max(qc, p)
            if(qc > p):
                svstartloc = index[loc]


        else:
            if(insv == True):
                svlist.append(['INS', contig, svstartloc, index[loc], qc, 1])

                insv = False
        loc += 1
        
  
    return svlist
def decode_flag(Flag):
    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}
    return signal[Flag] if(Flag in signal) else 0
def call_genotype(model, prediction, data):

    mask = np.where(prediction>0)[0]

    tmpdata = tf.data.Dataset.from_tensor_slices(data[mask]).batch(8048)
    return model.predict(tmpdata)

def bacth_data(data, batch_size):
    return tf.reshape(data[:data.shape[0]//batch_size*batch_size], [data.shape[0]//batch_size, batch_size, 200, 9, 1])

def batchdata(data, timestep, window_size = 200):#[sample, window_size, maxread, 1] --> (None, timestep, window_size, maxread)
    numoftimestep = (data.size//(window_size*9))//timestep - 1
    tailtimestep = data.shape[0] - numoftimestep*timestep

    return data[:numoftimestep*timestep].reshape((numoftimestep, timestep, window_size*9)), data[numoftimestep*timestep:].reshape((1, tailtimestep, window_size*9))

def call_sv(feedgpuqueue, calledqueue, step, window_size, weightpathdict, meanvalue, workdir, genotype = False, mc = False, Hi = 200):
    print('call sv start')
    print('step ', step)
    model = main_model(mc)
    model.load_weights(weightpathdict['DELINS'])
    print('model load complete')
    genotype = True
    if(genotype == True):
        model_genotype = geno_model()
        model_genotype.load_weights(weightpathdict['GENOTYPE'])
        print('model_genotype loaded')
    modeldict = {}
    modeldict['main'] = model
    modeldict['genotype'] = model_genotype
    
    result = []
    notshowinbp = []
    missedbp = []
    firstrun = True

    

    count = 0
    usedlist = []
    timedict = {}
    while(True):
        data, index, bp = 0, 0, 0
        filelist = [path for path in os.listdir(workdir+'tmp/') if(('npz' in path) and (path not in usedlist))]

        if(len(filelist) == 0 and feedgpuqueue.empty() == False):

            

            tmplist = feedgpuqueue.get()
            if(type(tmplist) == int):
                print('result size ', len(result))
                calledqueue.put(result)
                print('stop signal was recevived, the result was sended')
                break
            
        elif(len(filelist) != 0):
            while(True):
                try:
                    st = time.time()
                    data, index, bp = 0, 0, 0
                    fileiloc = 0
                    datainfo = filelist[fileiloc][:-4].split(':')
                    contig = datainfo[0]
                    start = int(datainfo[1])
                    end = int(datainfo[2])
                    alldata = np.load(workdir+filelist[fileiloc])
                    data, index, bp = alldata['data'].astype(np.float32), alldata['index'], alldata['bp']
                    print('loaded ', workdir+filelist[fileiloc], os.path.isfile(workdir+filelist[fileiloc]))
                    usedlist.append(filelist[fileiloc])
                    timedict['loadtime'] = time.time() - st
                    break
                except:
                    print('load file failed '+filelist[fileiloc], os.path.isfile(workdir+filelist[fileiloc]))
                    
                    time.sleep(3)
                    continue
            
            try:
                print('Predict on', contig, str(start), str(end))
                st = time.time()
                SVtypeprediction, GENOtypeprediction, GENOindex = call_type_geno(data, index, True, window_size, step, modeldict)
                timedict['detecttime'] = time.time() - st
                st = time.time()
                svlist = mergesv(SVtypeprediction, index, contig, window_size)
                timedict['mergesvtime'] = time.time() - st
                st = time.time()
                tmp_notshowinbp, tmp_missedbp, tmp_result = getbreakpoint(svlist, breakpoint = bp, test_type = 'DEL', GENOindex = GENOindex, GENOtypeprediction = GENOtypeprediction, step = step, window_size = window_size)
    
                result += tmp_result

                tmp_notshowinbp, tmp_missedbp, tmp_result = getbreakpoint(svlist, breakpoint = bp, test_type = 'INS', GENOindex = GENOindex, GENOtypeprediction = GENOtypeprediction, step = step, window_size = window_size)

                result += tmp_result
                timedict['bptime'] = time.time() - st
                print(timedict)

                print(contig, str(start), str(end), 'completed', os.path.isfile(workdir+filelist[fileiloc]), str(len(usedlist))+'/'+str(len([path for path in os.listdir(workdir) if(('npz' in path))])))
                firstrun = False
            except:

                print('error found on ', contig, str(start), str(end))
        else:
            time.sleep(3)


  
                        
   

            
    print('call_sv stop')
    return 0
            
        
def baseinfo_main(bamfilepath='', pdict = {}, workdir='./', max_worker = 1e20, step = 200, window_size = 200,  INTERVAL = 1e7, includecontig = [], guesstime = 400, genotype = False, mc = False, Hi = 200, MINSIZE = 0):
    
    outputpath = bamfilepath.split('/')[-1][:-4] + '.vcf'
    if(len(pdict) != 0):
        if('bamfilepath' in pdict):
            bamfilepath = pdict['bamfilepath']
        if('threads' in pdict):
            max_worker = max(pdict['threads'], 1)
        if('step' in pdict):
            step = min(max(pdict['step']//1, 1), 200)
            print('step set to ', step)
        if('INTERVAL' in pdict):
            INTERVAL = int(max(pdict['INTERVAL'], 1000000))
        if('includecontig' in pdict):
            includecontig = pdict['includecontig']
        if('genotype' in pdict):
            genotype = pdict['genotype']
        if('workdir' in pdict):
            workdir = pdict['workdir']
        if('outputpath' in pdict):
            outputpath = pdict['outputpath']
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb')
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    meanvalue, maxcountread = 0, 0
    max_worker = max(min(max_worker, (len(os.sched_getaffinity(0)))), 1)
    print('Process number limit to ', max_worker)
    if(workdir != './'):
        if(workdir[-1] != '/'):
            workdir += '/'
        if(os.path.isdir(workdir) == True):
            [os.remove(workdir+path) for path in os.listdir(workdir) if('np' in path)]
        else:
            os.makedirs(workdir[:-1])
        if(os.path.isdir(workdir+'tmp/') == True):
            [os.remove(workdir+'tmp/'+path) for path in os.listdir(workdir+'tmp/') if('np' in path)]
        else:
            os.makedirs(workdir+'tmp')
    
    print('WorkDir ', workdir)
    AlignedSegmentlist = []
    BridgeAlignedSegmentlist = []
    BATCHCONTIG = ''
    BATCHEND = 0

    feedgpuqueue = Queue()
    calledqueue = Queue()


    multiprocessing.Process(target=guess_summary, args=(bamfilepath, guesstime, workdir)).start()
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    weightpathdict = {
                      'DELINS':'./type',
                      'GENOTYPE':"./geno",
                      'bacth_inner':100,
                      'bacth_outer':50


    }

    p = multiprocessing.Process(target=call_sv, args=(feedgpuqueue, calledqueue, step, window_size, weightpathdict, meanvalue, workdir, genotype, mc, Hi, ))
    p.start()
    if(len(includecontig) == 0):
        includecontig = [str(contig) for contig in contig2length]
    else:
        includecontig = [str(contig) for contig in includecontig]
        
    if(MINSIZE == 0):
        totalsize = 0
        for contig in includecontig:
            totalsize += contig2length[contig]
        MINSIZE = int(min((totalsize // (max_worker)), INTERVAL))
    print('MINSIZE set to ', MINSIZE) 
    orderarray = np.argsort([contig2length[contig] for contig in includecontig])[::-1]
    for contigiloc in orderarray:
        contig = includecontig[contigiloc]
        if(contig2length[contig]<100000):
            continue
        if(contig2length[contig]<200000 or (max_worker == 1)):
            baseinfo_AlignedSegment(genotype, bamfilepath, contig, 0, contig2length[contig], meanvalue, window_size, maxcountread, workdir, INTERVAL)
            continue
        r_start = 0
        while(r_start<contig2length[contig]):

            while(True):
                if(len(multiprocessing.active_children()) < (max_worker+1)):
                    print('working on contig = ', contig, r_start, r_start+MINSIZE )
                    contig = str(contig)
                    if((r_start + int(INTERVAL)) > contig2length[contig]):
                        multiprocessing.Process(target=baseinfo_AlignedSegment, args=(genotype, bamfilepath, contig, r_start, r_start+int(1.5 * MINSIZE), meanvalue, window_size, maxcountread, workdir, INTERVAL)).start()
                        r_start += int(INTERVAL)
                    else:
                        multiprocessing.Process(target=baseinfo_AlignedSegment, args=(genotype, bamfilepath, contig, r_start, r_start+MINSIZE, meanvalue, window_size, maxcountread, workdir, INTERVAL)).start()
                        r_start += MINSIZE
                    break
                else:
                    time.sleep(2)
    


    stopsignal = False
    while(True):
        if((stopsignal == False) and (len(multiprocessing.active_children())) == 1):
            feedgpuqueue.put(1) 
            stopsignal = True
            print('stop signal sended to callsv')
        elif((stopsignal == True) and (calledqueue.empty() == False)):
            result = calledqueue.get()
            tovcf(result, contig2length, outputpath)
            print('All child proccess finished, and result are received')
            break
        else:               
            time.sleep(2)


                            
                    
                
                

def baseinfo_guess(bamfile, contig, start, end, feature_count, maxcountread):
     
    nooverlap = True
    cigarlist = List()
    mdtaglist = List()
    corposlist = List() 
    readtrustedarray = []
    qualityarray = []
    for AlignedSegment in bamfile.fetch(contig, start, end):
        qualityarray.append([AlignedSegment.query_alignment_length/ AlignedSegment.infer_read_length(), AlignedSegment.query_alignment_length, AlignedSegment.mapping_quality])
        mdtaglist.append(AlignedSegment.get_tag('MD'))
        cigarlist.append(AlignedSegment.cigarstring)
        corposlist.append(List([AlignedSegment.reference_start - start, AlignedSegment.reference_end - start]))
        readtrustedarray.append([AlignedSegment.query_alignment_length, AlignedSegment.infer_read_length(), AlignedSegment.mapping_quality])
        nooverlap = False

    if(nooverlap):
        print(dsahjdaj)
    qualityarray = np.array(qualityarray, dtype = 'float32')
    qualityarray = qualityarray - qualityarray.min(axis = 0)
    qualityarray = qualityarray / (qualityarray.max(axis = 0) + 0.000001)
    return mamnet.g_d(mdtaglist, cigarlist, corposlist, end - start, np.argsort(qualityarray.sum(axis = 1))[::-1], maxcountread).astype('float32'), np.array(readtrustedarray)
                            
def guess_summary_depth(bamfilepath, times, window_size = 200, feature_count = 9):
    meanlist = []
    trustsummary = []
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    topcontig = ''
    for AlignedSegment in bamfile.fetch():
        if('chr' in AlignedSegment.reference_name):
            topcontig = 'chr'
        break
            
    for i in range(times):
        while(True):
            contig = str(np.random.randint(1, 23))
            start = np.random.randint(150000000)
            
            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, topcontig+contig, start, start + window_size, feature_count, 2)
                meanlist.append(data.flatten())
            except:
                #print('Bad luck')

                continue
            break
       
    meanlist = np.stack(meanlist)
    return meanlist.reshape(meanlist.size//feature_count, feature_count).astype('float64').mean(axis = 0).astype('int32')[-1]
def guess_summary(bamfilepath, times, window_size = 200, feature_count = 9):
    meanlist = []
    trustsummary = []
    maxcountread = int(guess_summary_depth(bamfilepath, times, window_size = 200, feature_count = 9))+1
    print('depth = ',  maxcountread)
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    topcontig = ''
    for AlignedSegment in bamfile.fetch():
        if('chr' in AlignedSegment.reference_name):
            topcontig = 'chr'
        break
            
    for i in range(times):
        while(True):
            contig = str(np.random.randint(1, 23))
            start = np.random.randint(150000000)
            
            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, topcontig+contig, start, start + window_size, feature_count, maxcountread)
                meanlist.append(data.flatten())

            except:
                #print('Bad luck')

                continue
            break
       

    meanlist = np.stack(meanlist)
    return meanlist.reshape(meanlist.size//feature_count, feature_count).astype('float64').mean(axis = 0, keepdims = True).astype('float32'), 1, 2
def closerone(floatnumber):
    floor = int(floatnumber)
    top = floor + 1
    return np.array([floor, top])[np.argsort(abs(np.array([floor, top]) - floatnumber))[0]]

def guess_summary_depth(bamfilepath, times, window_size = 200, feature_count = 9):
    meanlist = []
    trustsummary = []
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    topcontig = ''
    for AlignedSegment in bamfile.fetch():
        if('chr' in AlignedSegment.reference_name):
            topcontig = 'chr'
        break
            
    for i in range(times):
        while(True):
            contig = str(np.random.randint(1, 23))
            start = np.random.randint(contig2length[topcontig+contig])
            
            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, topcontig+contig, start, start + window_size, feature_count, 2)
                meanlist.append(data.flatten())
            except:
                #print('Bad luck')

                continue
            break
       
    meanlist = np.stack(meanlist)
    return meanlist.reshape(meanlist.size//feature_count, feature_count).astype('float64').mean(axis = 0).astype('int32')[-1]
def guess_summary(bamfilepath, times, workdir, window_size = 200, feature_count = 9):
    meanlist = []
    trustsummary = []
    maxcountread = closerone(guess_summary_depth(bamfilepath, times, window_size = 200, feature_count = 9))
    print('depth = ',  maxcountread)
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    contig2length = {}
    for count in range(len(bamfile.get_index_statistics())):
        contig2length[bamfile.get_index_statistics()[count].contig] = bamfile.lengths[count]
    topcontig = ''
    for AlignedSegment in bamfile.fetch():
        if('chr' in AlignedSegment.reference_name):
            topcontig = 'chr'
        break       
    for i in range(times):
        while(True):
            contig = str(np.random.randint(1, 23))
            start = np.random.randint(contig2length[topcontig+contig])
            
            try:
                data, tmptrustsumary = baseinfo_guess(bamfile, topcontig+contig, start, start + window_size, feature_count, maxcountread)
                meanlist.append(data.flatten())

            except:
                #print('Bad luck')

                continue
            break
       

    meanlist = np.stack(meanlist)
    np.save(workdir+'meanarray', meanlist.reshape(meanlist.size//feature_count, feature_count).astype('float64').mean(axis = 0, keepdims = True).astype('float32'))
     

def tovcf(rawsvlist, contig2length, outputpath):
    top = """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">\n"""
    body = ''
    for contig in contig2length:
        body += "##contig=<ID="+contig+",length="+str(int(contig2length[contig]))+">\n"
    tail = """##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, INS=Insertion">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=RE,Number=1,Type=Integer,Description="Number of read support this record">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n"""

    myvcf = top+body+tail
    genomapper = {0:'0/0', 1:'0/1', 2:'1/1'}
    svlist = [[rec[0], int(rec[1]), int(rec[2]), int(rec[3]), genomapper[int(rec[-1])]] for rec in rawsvlist]
        
    for rec in pd.DataFrame(svlist).sort_values([0, 1]).values:


        contig = rec[0]


        geno = rec[4]
        if(rec[2]<0):
            recinfo = 'SVLEN=' + str(int(rec[2]))+';SVTYPE=' + 'DEL'+';END='+str(int(rec[1])+abs(int(rec[2])))+';RE='+str(int(rec[3])+1)+'\tGT\t'+geno+'\n'
            myvcf += (contig +'\t'+ str(int(rec[1]))+'\t'+ '.'+'\t'+ '.'+'\t'+ 'A'+'\t'+ '.'+'\t'+ 'PASS'+'\t'+recinfo)
        elif(rec[2]>0):
            recinfo = 'SVLEN=' + str(int(rec[2]))+';SVTYPE=' + 'INS'+';END='+str(int(rec[1])+1)+';RE='+str(int(rec[3])+1)+'\tGT\t'+geno+'\n'
            myvcf += (contig +'\t'+ str(int(rec[1]))+'\t'+ '.'+'\t'+ '.'+'\t'+ 'A'+'\t'+ '.'+'\t'+ 'PASS'+'\t'+recinfo)


    with open(outputpath, "w") as f:
        f.write(myvcf)





def paddingdoubleend(mask, times = 3):
    insv = False
    masksize = mask.size
    for loc in range(len(mask)):
        if(mask[loc]>0):
            if(insv == False):
                paddingstart = max(0, loc - times)
                mask[paddingstart: loc] = -1
                insv = True
        else:
            if(insv == True):
                paddingend = min(masksize, loc + times)
                mask[loc: paddingend] = -1
                insv = False
    return mask != 0

def call_type_geno(data, index, genotype, window_size, step, modeldict):
    loc = 0
    GENOtypeprediction = []
    GENOindex = []
    SVtypeprediction = 0
    slicedgenodatalist = []
    while(loc < window_size):
        top, tail = batchdata(data.reshape((data.size // 9, 9))[loc: loc - window_size ].reshape((data.shape[0]-1, data.shape[1])), 100)
        SVtypeprediction += np.concatenate([(modeldict['main'].predict(top, batch_size= 100)>0.5).reshape(-1,2), (modeldict['main'].predict(tail, batch_size= 100)>0.5).reshape(-1,2)], axis = 0)
        loc += step
    

    loc = 0  
    mask = paddingdoubleend(SVtypeprediction.sum(axis = 1))
    slicedgenodata = []
    index = index[:-1][mask]
    while(loc < window_size):
        GENOindex.append(index+loc)
        slicedgenodata.append(data.reshape((data.size // 9, 9))[loc: loc - window_size ].reshape((data.shape[0]-1, data.shape[1]))[mask])
        loc += step
    slicedgenodata = np.concatenate(slicedgenodata, axis = 0)
    GENOtypeprediction = np.argsort((modeldict['genotype'].predict(slicedgenodata, batch_size= 10000)).reshape(-1,3), axis = 1)[:,-1].reshape(window_size//step, -1).T.flatten()
        
    GENOindex = np.array(GENOindex).T.flatten()
    return SVtypeprediction, GENOtypeprediction, GENOindex

def mergesv(SVtypeprediction, index, contig, window_size):
    loc = -1
    in_deletion = False
    in_insertion = False
    svlist = []
    icount = 0
    dcount = 0
    for rec in SVtypeprediction:
        loc += 1
        if(rec[0] != 0):
            if(in_deletion == False):
                tmpdeletion = ['DEL', contig, index[loc]]
                delsp = rec[0]
                delspcache = [rec[0]]
                dcount += 1
            else:
                if(rec[0] > delsp):
                    delsp = rec[0]
                delspcache.append(rec[0])
            in_deletion = True
        if(rec[1] != 0):
            if(in_insertion == False):
                tmpinsertion = ['INS', contig, index[loc]]
                inssp = rec[1]
                insspcache = [rec[1]]
                icount += 1
            else:
                if(rec[1] > inssp):
                    inssp = rec[1]
                insspcache.append(rec[1])
            in_insertion = True
        if(rec[0] == 0):
            if(in_deletion == True):
                svlist.append(tmpdeletion)
                #svlist[-1] += [index[loc], delsp]
                svlist[-1] += [index[loc], np.array(delspcache).mean()]
                in_deletion = False
        if(rec[1] == 0):
            if(in_insertion == True):
                svlist.append(tmpinsertion)
                #svlist[-1] += [index[loc], inssp]
                svlist[-1] += [index[loc], np.array(insspcache).mean()]
                in_insertion = False
    if(in_deletion == True):
        svlist.append(tmpdeletion)
        #svlist[-1] += [index[loc], delsp]
        svlist[-1] += [index[loc], np.array(delspcache).mean()]
    if(in_insertion == True):
        svlist.append(tmpinsertion)
        #svlist[-1] += [index[loc], inssp]
        svlist[-1] += [index[loc], np.array(insspcache).mean()]
    svlist = np.array(svlist)
    svlist = svlist[np.argsort(svlist[:,2].astype('int64'))]
    #print(dcount, icount)
    return svlist

    
    
###########################################################################

def cumpute_geno(bstart, bsvlen, GENOindex, GENOtypeprediction, GENOlookup, GENOlookupend, window_size , step, debug = False):
    if(bsvlen<0):
        svtype = 'DEL'
        FROND = bstart
        TAIL = bstart - bsvlen

    else:
        svtype = 'INS'
        FROND = bstart - window_size//2
        TAIL = bstart + window_size//2
    CENTER = (FROND +TAIL)//2
    try:
        slicestart = GENOlookup[max((((FROND-window_size) - GENOindex[0])//step), 0)]
        sliceend = GENOlookup[min((((TAIL) - GENOindex[0])//step), GENOlookupend)]
    except:
        try:
            slicestart = max(np.where(GENOindex>(FROND-window_size))[0][0], 0)
            sliceend = np.where(GENOindex<(TAIL))[0][-1]+1
        except:
            return 1
        
    if(debug == True):
        print(FROND, TAIL)
        print(slicestart, sliceend)
    maskedstart = GENOindex[slicestart:sliceend]
    maskedgeno = GENOtypeprediction[slicestart:sliceend]
    varmask = maskedgeno != 0
    maskedstart = maskedstart[varmask]
    maskedgeno = maskedgeno[varmask]
    if(maskedgeno.size == 0):
        return 1

    maskedcenter = maskedstart + window_size//2 


    if(svtype == 'DEL'):
        return [1,2][np.argsort(np.bincount(maskedgeno-1))[-1]]
    else:
        return [1,2][np.argsort(np.bincount((maskedgeno-1)[-3:]))[-1]]
def getbreakpoint(svlist, breakpoint, test_type, GENOindex, GENOtypeprediction, window_size , step):

    locin_svlist = 0
    locin_bp = 0
    notshowinbp = []
    missedbp = []
    result = []
    
    waitreachbp = False
    
    bstart = int(breakpoint[0][0])
    count = 0
    
    window_bias = 10
    slicestart = 0
    prestart = GENOindex[0]
    
    GENOindex_start = GENOindex[0]
    GENOlookup = {}
    linerindex = 0
    realindex = (GENOindex - GENOindex_start)//step
    for i in realindex:
        GENOlookup[i] = linerindex
        linerindex += 1
    GENOlookupend = i
    for rec in svlist:

        if(rec[0] != test_type):
            continue
        count += 1    
        rstart, rend = int(rec[2])-500, int(rec[3])+500
        cache = []
        
        if((rend <= bstart)):
            notshowinbp.append(rec)
            continue
        
        candicate_sv = []
        while(True):
            
            if(locin_bp == len(breakpoint)):
                break
            onebp = breakpoint[locin_bp]
            
            locin_bp += 1
            bptype = 'DEL'
            if(onebp[1] > 0):
                bptype = 'INS'
            if(bptype != test_type):
                continue
                
            bstart = int(onebp[0])
            if(int(onebp[1]) < 0):
                bend = bstart - int(onebp[1])
            else:
                bend = bstart + 1
                  
                
            if((rstart <= bend) and (rend > bstart)): #overalp
                
                
                if(len(candicate_sv) != 0): 
                    if(candicate_sv[3] <= int(onebp[2])):
                        candicate_sv = [(rec[1]), bstart, int(onebp[1]), int(onebp[2]), cumpute_geno(bstart, int(onebp[1]), GENOindex, GENOtypeprediction, GENOlookup, GENOlookupend, window_size , step)]
                else:
                    candicate_sv = [(rec[1]), bstart, int(onebp[1]), int(onebp[2]), cumpute_geno(bstart, int(onebp[1]), GENOindex, GENOtypeprediction, GENOlookup, GENOlookupend, window_size , step)]
                continue
                    
                        
            
            

            if(bstart >= rend):
                locin_bp -= 1
                break
 
            
        if(len(candicate_sv) != 0):
            result.append(candicate_sv)

    return notshowinbp, missedbp, result

pstring = sys.argv
forvaule = False
pdict = {}
for item in pstring[1:]:
    if(forvaule == True):
        forvaule = False
        if('path' in op or op == 'workdir'):
            pdict[op] = (item)
            continue
        pdict[op] = eval(item)
        
    if('-' == item[:1]):
        forvaule = True
        op = item[1:]
print(pdict)
st = time.time()
baseinfo_main('', pdict, max_worker = 1e20, step = 50,  INTERVAL = 1e7, includecontig = [], guesstime = 400)
print('Running Time ', time.time() - st)

