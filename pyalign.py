import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
from scipy.sparse import coo_matrix
import time
import numpy as np
import tensorflow as tf
import pysam
import matplotlib.pyplot as plt

def allocation(cigartuples, posinref, readarray, qualityarray, start, end): 
  posinread = 0

  padinsertinfo = []

  softclipinfo = []

  cigararray = np.array(-1)

  itemcount = 0


  slicestart = 0
  reachstart = False
  reachend = False
  softclippresent = False
  softclipcount = 0
  totalcount = readarray.shape[0]

  if(posinref <= start): 


    itemcount = -1

    for item in cigartuples:

      itemcount = itemcount + 1

      if(item[0] in [0, 2, 7, 8]):

        if(posinref + item[1] > start): # Meet start. Slice required.

          #print('front early stopping')
          #print('posinref is', posinref)
          #cigararray = np.append(cigararray, np.ones(posinref + item[1] - start) * item[0])
          cigartuples[itemcount] = [item[0], int(posinref + item[1] - start)] 
          #print(item[1])
          #print('push in cigararray ', cigartuples[itemcount])

          if(item[0] in [0, 7, 8]): # For reference match
            
            slicestart = posinread + start - posinref
            posinread = posinread + start - posinref
            posinref = start

          else: # For deletion

            slicestart = posinread
            posinref = start

          reachstart = True

          break


        if(item[0] in [0, 7, 8]): 

          posinread = posinread + item[1]
          posinref = posinref + item[1]

        else: # For deletion

          posinref = posinref + item[1]

        continue

      
      if(item[0] in [1, 4]): 

        posinread = posinread + item[1]
        if(item[0] == 4):
          softclippresent = True
          softclipcount = softclipcount + item[1]
        continue
      
      if(item[0] == 5):
        
        softclippresent = True
        softclipcount = softclipcount + item[1]
        totalcount = totalcount + item[1]
        continue
        
      print(str(item[0]),' CIGAR TAG DONT SUPPORT ')
      return None
        

  else: 

    if(posinref >= end): 

      return None, np.ones(int(end - start)) * 6, np.column_stack((np.arange(start, end).reshape(int(end - start), 1), np.zeros((int(end - start), 1)))), np.ones(int(end - start)).reshape(int(end - start), 1) * 0
    
    else: 

      readfrontpadding = np.column_stack((np.arange(start, posinref).reshape(int(posinref - start), 1), np.zeros((int(posinref - start), 1))))
      readarray = np.row_stack((readfrontpadding, readarray))
      qualityarray = np.row_stack((np.zeros((int(posinref - start), 1)), qualityarray))
      cigararray = np.ones(int(posinref - start + 1)) * 6 # add 1 to deal with last slice
      posinread = posinread + posinref - start
      reachstart = True


  if(reachstart == False): 
    
    return None, np.ones(int(end - start)) * 6, np.column_stack((np.arange(start, end).reshape(int(end - start), 1), np.zeros((int(end - start), 1)))), np.ones(int(end - start)).reshape(int(end - start), 1) * 0
  
  #print('posinref is', posinref)
  #print('read position is ', readarray[slicestart: slicestart +10])

  
  if(posinref < end):
    
    usefront = False
    usetail = True
    
    locincigartuples = 0
    
    cliploc = []
    
    for item in cigartuples[itemcount:]:
      
      locincigartuples = locincigartuples + 1
      #print('current cigar is ', item, 'posinref is ', posinref, 'posinread is ', posinread, 'start is ', start, 'end is ', end)
      #For reference match: posinref is match part first base
      if(item[0] in [0, 7, 8]):
        
        usefront = True
        
        if(end <= item[1] + posinref): 

          #print('posinref, end ', posinref, end)
          cigararray = np.append(cigararray, np.ones(end - posinref) * item[0])
          readarray = readarray[: posinread + end - posinref]
          qualityarray = qualityarray[: posinread + end - posinref]
          posinref = end
          break
          
        else: 

          cigararray = np.append(cigararray, np.ones(item[1]) * item[0])
            
          posinref = posinref + item[1]

          posinread = posinread + item[1]
          usefront = True

          continue
        
      #For insertion
      if(item[0] == 1):

        cigararray = np.append(cigararray, np.ones(item[1]) * item[0])

        posinread = posinread + item[1]

        padinsertinfo.append([int(posinref - 1), int(item[1])])


        continue

   
      if(item[0] == 2): 
        
        usefront = True
        
        if(end <= item[1] + posinref): 
            
          #print('del posinref, end ', posinref, end)
          cigararray = np.append(cigararray, np.ones(end - posinref) * item[0])
          readarray = np.row_stack((readarray[: posinread], np.column_stack((np.arange(posinref, end).reshape(end - posinref, 1), np.zeros(end - posinref).reshape(end - posinref, 1)))))
          qualityarray = np.row_stack((qualityarray[: posinread], np.zeros(end - posinref).reshape(end - posinref, 1)))

          posinref = end 
          break
          
        else: #Not yet. Following parameter need update: posinref

          cigararray = np.append(cigararray, np.ones(item[1]) * item[0])
          paddelarray = np.column_stack((np.arange(posinref, posinref + item[1]).reshape(item[1], 1), np.zeros(item[1]).reshape(item[1], 1)))
          #print(paddelarray.shape)
          readarray = np.row_stack((np.row_stack((readarray[:posinread], paddelarray)), readarray[posinread:]))
          qualityarray = np.row_stack((qualityarray[: posinread], np.row_stack((np.zeros(item[1]).reshape(item[1], 1), qualityarray[posinread: ]))))

          posinref = posinref + item[1]
          posinread = posinread + item[1]

          usefront = True


          continue

      #For soft clip: posinref is first base follow with this softclip part
      if(item[0] == 4): # We have not meet the end. Following parameter need update: posinread.
        softcigartmp = np.ones(item[1]) * item[0]
        if(locincigartuples == len(cigartuples[itemcount:])):
         
          usetail = False
        
        else:
          
          usetail = False
          for cigarinfo in cigartuples[itemcount + locincigartuples:]:
            if(cigarinfo[0] in [0, 2, 7, 8]):
              usetail = True
              

        if(usefront == True and usetail == True):

          softclipinfo.append([[int(posinref - 1), int(posinref)], readarray[posinread: posinread + item[1], 1]])
          
          softcigartmp[0], softcigartmp[-1] = softcigartmp[0] + 0.1, softcigartmp[-1] + 0.1
          
          cliploc.append(int(posinref - 1))
          cliploc.append(int(posinref))
        
        elif(usefront == True):

          softclipinfo.append([[int(posinref - 1)], readarray[posinread: posinread + item[1], 1]])
          
          softcigartmp[0] = softcigartmp[0] + 0.1
          
          cliploc.append(int(posinref - 1))
        
        else:
          
          softclipinfo.append([[-int(posinref)], readarray[posinread: posinread + item[1], 1]])
          
          softcigartmp[-1] = softcigartmp[-1] + 0.1
          
          cliploc.append(int(posinref))
          
          


        
        cigararray = np.append(cigararray, softcigartmp)
          
        posinread = posinread + item[1]

        padinsertinfo.append([int(posinref - 1), int(item[1])])

        softclippresent = True

        softclipcount = softclipcount + item[1]


        continue
        
      if(item[0] == 5): # Pass the hardclip cigar string.
        
        totalcount = totalcount + item[1]
        
        softclippresent = True

        softclipcount = softclipcount + item[1]
        
        if(locincigartuples == len(cigartuples[itemcount:])):
         
          usetail = False
        
        else:
          
          usetail = False
          for cigarinfo in cigartuples[itemcount + locincigartuples:]:
            if(cigarinfo[0] in [0, 2, 7, 8]):
              usetail = True
              
        hardlen = 1
        tmppadnone = np.array([[None]])
        
        
        if(usefront and usetail):
          hardlen = 2
          tmppadnone = np.array([[None], [None]])
          
          cliploc.append(int(posinref - 1))
          cliploc.append(int(posinref))
        elif(usefront == True):
          
          cliploc.append(int(posinref - 1))
        
        else:
          
          cliploc.append(int(posinref))
          
          
        cigararray = np.append(cigararray, np.ones(hardlen) * item[0])

        padinsertinfo.append([int(posinref - 1), hardlen])
        
        
        paddelarray = np.column_stack((tmppadnone, np.zeros(hardlen).reshape(hardlen, 1)))
          #print(paddelarray.shape)
        if(posinread == 0):
          readarray = np.row_stack((paddelarray, readarray[posinread:]))
          qualityarray = np.row_stack((np.zeros(hardlen).reshape(hardlen, 1), qualityarray[posinread: ]))
        elif(posinread == readarray.shape[0]):
          readarray = np.row_stack((readarray, paddelarray))
          qualityarray = np.row_stack((qualityarray, np.zeros(hardlen).reshape(hardlen, 1)))
        else:
          readarray = np.row_stack((np.row_stack((readarray[:posinread], paddelarray)), readarray[posinread:]))
          qualityarray = np.row_stack((qualityarray[: posinread], np.row_stack((np.zeros(hardlen).reshape(hardlen, 1), qualityarray[posinread: ]))))

        
        posinread = posinread + hardlen

        continue
        
      print(str(item[0]),' CIGAR TAG DONT SUPPORT ')
      return None


  #readarray = readarray[slicestart: posinread + end - posinref]
  
  #print(cigarinfodict)
  if(posinref < end): #Need padding.
    readrearpadding = np.column_stack((np.arange(posinref, end).reshape(end - posinref, 1), np.zeros(end - posinref).reshape(end - posinref, 1)))
    readarray = np.row_stack((readarray, readrearpadding))
    qualityarray = np.row_stack((qualityarray, np.zeros(end - posinref).reshape(end - posinref, 1)))
    cigararray = np.append(cigararray, np.ones(end - posinref) * 6)
  
  return np.array(padinsertinfo), cigararray[1:], readarray[slicestart:], qualityarray[slicestart:], [softclippresent, softclipcount / totalcount , softclipinfo, cliploc]

def paddel(AlignedSegment, start, end):

  grp = np.array(AlignedSegment.get_reference_positions(True))#[AlignedSegment.query_alignment_start: AlignedSegment.query_alignment_end]
  grp = grp.reshape(grp.size, 1)
  read = pd.DataFrame(list(AlignedSegment.query_sequence)).replace({'A': 10, 'T': 15, 'G': 20, 'C': 25}).values.reshape(grp.size,1)
  readarray = np.column_stack((grp, read))

  pos = 0
  i  = 0
  delpos = 0

  '''print(readarray.T[0].tolist())
  print(readarray.T[1].tolist())'''
  #print(grp[-10:].T)

  cigartuples = AlignedSegment.cigartuples
  #print(grp[-cigartuples[-1][1] - 10:].T)
  if(cigartuples == None):
    print('The alignment does not give CIGAR string')
    return None

  while(cigartuples[i][0] not in [0, 7, 8]): # Get TRUE reference start location
    if(cigartuples[i][0] != 5):
      pos = cigartuples[i][1] + pos
      if(cigartuples[i][0] == 2):
        delpos = cigartuples[i][1] + delpos

    i = i + 1

  '''if(grp[0] == None):#test for insertion or softclip on begin
    tmpbase = np.array([readarray[0][0] - 1, 0]).reshape(1,2)
    readarray = np.row_stack((tmpbase, readarray))'''
  #print('origin readarray is ', readarray.T.tolist())
  qqarray = np.array(AlignedSegment.query_qualities)
  if(qqarray.size == 1):
    qqarray = np.zeros((readarray.shape[0], 1))
  else:
    qqarray = qqarray.reshape((readarray.shape[0], 1))

  iarray, cigararray, readarray, qualityarray, softclipinfo = allocation(cigartuples, int(grp[pos] - delpos), readarray, qqarray, start, end)
  if(softclipinfo[0] == True):
    softclipinfo.append(read)
  else:
    softclipinfo.append([])

  return readarray, iarray, cigararray, qualityarray, softclipinfo


def dropselectcigartag(pcigararray, selectedtag, refseq, filter = False, preadarray = None, cutvalue = 1, keepfrontsoftclip = True):
  for tag in selectedtag:
    try:
      outputset = set(np.nonzero((pcigararray == tag).sum(axis = 0) == 0)[0].tolist()) & outputset
    except:
      outputset = set(np.nonzero((pcigararray == tag).sum(axis = 0) == 0)[0].tolist())

  if(filter == True):
    outputset = set(np.nonzero((preadarray > 0).sum(axis = 0, keepdims = True) > (cutvalue))[1].tolist()) & outputset

  if(keepfrontsoftclip):

    outputset = set(np.nonzero((pcigararray == 4.1).sum(axis = 0) != 0)[0].tolist()) & outputset

  outputset = set(np.nonzero(refseq > -1)[0].tolist()) | outputset
  return np.sort(np.array(list(outputset)))

def myshow(preadarray, pcigararray, qualityarray, refseq, seqbase, excludelist = [1, 4, 4.1], includecagtag = [2], filter = False, cutvalue = 1, maxdot = 200, softread = [], showpic = False, minrow = 18, maxrow = 18):
  #print(preadarray.shape, pcigararray.shape, qualityarray.shape)
  pltsoft = False
  if(len(softread) > 0):
    psoftarray = (np.array(softread[:,1]).reshape(preadarray.shape[0], 1).astype('float32') * (preadarray > 0).astype('float32'))
    pltsoft = True
  if(len(excludelist) > 0):

    columnkeeped = dropselectcigartag(pcigararray, excludelist, refseq, filter, preadarray, cutvalue)

    preadarray = preadarray[:, columnkeeped]
    qualityarray = qualityarray[:, columnkeeped]
    if(pltsoft == True):
      psoftarray = psoftarray[:, columnkeeped]

    slicedpcigararray = pcigararray[:, columnkeeped]
    for cigartag in includecagtag:
      try:
        spotarray = spotarray + (slicedpcigararray == cigartag) * cigartag
      except:
        spotarray = (slicedpcigararray == cigartag) * cigartag
  else:

    slicedpcigararray = pcigararray
    for cigartag in includecagtag:
      try:
        spotarray = spotarray + (slicedpcigararray == cigartag) * cigartag
      except:
        spotarray = (slicedpcigararray == cigartag) * cigartag

  cspotarray = spotarray[:,:200][np.argsort(spotarray[:200].sum(axis = 1))[::-1]]
  for loc in range(200, spotarray.shape[1], 200):
    tmp = spotarray[:,loc:loc+200][np.argsort(spotarray[:,loc:loc+200].sum(axis = 1))[::-1]]
    cspotarray = np.column_stack((cspotarray, tmp))
  spotarray=cspotarray 
  if(preadarray.shape[0] < minrow):
    pt = [preadarray, spotarray, psoftarray, qualityarray]
    padrownumber = minrow - preadarray.shape[0]
    column_number = preadarray.shape[1]
    for locinpt in range(4):
      
      pt[locinpt] = np.row_stack((pt[locinpt], np.zeros((padrownumber, column_number))))

    preadarray, spotarray, psoftarray, qualityarray = pt[0], pt[1], pt[2], pt[3]
  
  if(preadarray.shape[0] > maxrow and showpic == False):

    preadarray, spotarray, psoftarray, qualityarray = preadarray[:maxrow], spotarray[:maxrow], psoftarray[:maxrow], qualityarray[:maxrow]

    
  cliparray = np.ones((preadarray.shape[0], 1)) * seqbase

  if(showpic):
    loc = 0
    print('Read sequence')
    plt.matshow(preadarray)
    plt.show()
  else:
    timestep = 100
    fm = (spotarray>0).astype('float32').T
    fm = fm.reshape(fm.size//(200 * 18), 200, 18, 1)
    if(fm.shape[0]<timestep):
      return np.array(0), fm.reshape(1, fm.shape[0], 200, 18, 1)
    
    tail = fm.shape[0]%timestep
    if(tail == 0):
      return fm.reshape(fm.shape[0]//timestep, timestep, 200, 18, 1), np.array(0)
      
    topdata, taildata = fm[:-tail], fm[-tail:].reshape(1, tail, 200, 18, 1)
    
    
    return topdata.reshape(topdata.shape[0]//timestep, timestep, 200, 18, 1), taildata





def pileupf(bamfile, contig, start, end, droplq = False, dropvalue = 0.8):
  window_size = 200
  samplelocation = start + np.column_stack((np.arange(0, window_size * (int((end - start - 1) / window_size) + 1), window_size).reshape((int(( end - start - 1) / window_size) + 1), 1), np.arange(0, window_size * (int((end - start - 1) / window_size) + 1), window_size).reshape((int((end - start - 1) / window_size) + 1), 1) + window_size))
  end = samplelocation[-1, 1] 
  totalstarttime = time.time()
  locationlist = []
  readlist = []
  insertlist = []
  insertinfo = dict()
  keylist = []
  cigarlist = []
  qualitylist = []
  softcliplist = []

  debug = []
  
  depth = 0
  paddeltime = 0
  fetchtime = time.time()
  seqbase = np.zeros((1, end - start)).astype('float32')
  overlap = False
  
  for AlignedSegment in bamfile.fetch(contig, start, end):
    
    #debug.append(AlignedSegment)
    if(AlignedSegment.reference_start <= start):
        frontloc = 0
        whichstart = start
    else:
      frontloc = AlignedSegment.reference_start - start
      whichstart = AlignedSegment.reference_start
      
    if(AlignedSegment.reference_end >= (end - 1)):
      tailloc = end - start - 1
    else:
      tailloc = AlignedSegment.reference_end - start
    paddelstarttime = time.time()
    
    read, iarray, cigararray, qualityarray, softclipinfo = paddel(AlignedSegment, start, end)
    ratio = softclipinfo[1]
    tmp = seqbase[:,frontloc: tailloc + 1] + ratio
    for cclip in set(softclipinfo[3]):
      tmp[:,cclip - whichstart] = tmp[:,cclip - whichstart] + ratio
    fillblank = tmp.mean()
    seqbase[:,frontloc: tailloc + 1] = (np.column_stack((np.array([[fillblank]]), tmp))[:,:-1] + tmp + np.column_stack((tmp, np.array([[fillblank]])))[:,1:])/3
    
    
    if(droplq and softclipinfo[1] > dropvalue):
      continue
    overlap = True
    paddeltime = - paddelstarttime + time.time() + paddeltime
    locationlist.append(read[:,0])
    readlist.append(np.array(read[:,1]).astype('float32'))
    insertlist.append(iarray)
    cigarlist.append(cigararray.astype('float32'))
    qualitylist.append(qualityarray.flatten().astype('float32'))
    softcliplist.append(softclipinfo)
    depth = depth + 1
    '''print(read.shape)
    print(cigararray.shape)'''

    if(type(iarray) != np.ndarray):
      continue

    for item in iarray:
      cloc = item[0]
      if((item[0]) in insertinfo):
        if(cloc == lastloc):
          insertinfo[item[0]] = insertinfo[item[0]] + item[1]
        else:
          insertinfo[item[0]] = max(insertinfo[item[0]], int(item[1]))
      else:
        insertinfo[item[0]] = item[1]
        keylist.append(item[0])
      lastloc = cloc
  #print('fetch time = ', time.time() - fetchtime, time.time() - totalstarttime)
  
  keylist = np.sort(np.array(keylist))

  #print(keylist)
  '''print(readlist)
  print()
  print(insertinfo)'''

  #print(insertinfo)
  #readposlist = [0 for i in range(len(readlist))]

  refseq = np.arange(start, end)
  bias = 0
  for key in keylist:
    if(key == (end - 1)):
      print('end in insertioninfo')
      return 0
      
    insert = - np.ones(insertinfo[key])
    refseq = np.append(np.append(refseq[:key+1 + bias-start], insert), refseq[key+1+bias-start: ])
    bias = bias + insertinfo[key]
  
  if(overlap == False):
    return np.zeros((1, end - start)), np.ones((1, end - start)) * 6, np.zeros((1, end - start)), refseq, np.array([[False, 0]]), seqbase
  readcount = 0
  state = False
  readlisttime = time.time()
  
  cc = 0
  
  if(True):
    pallarray = 'None'
    for read in readlist:
    
      refloc = start
      locinread = 0
      insertcount = 0
      parray = 'None'


      for key in keylist:

        #print(insertlist[readcount][insertcount])
        while(True):
          slicelength = key + 1 - refloc
          refloc = refloc + slicelength
          insertsizeofrfortkey = 0
          insertpresentonkey = False
          if(insertlist[readcount].shape[0] > insertcount and key == insertlist[readcount][insertcount][0]):
            
            onreadinsert = insertlist[readcount][insertcount][1]

            insertpresentonkey = True
            while(insertlist[readcount].shape[0] > (insertcount + 1) and key == insertlist[readcount][insertcount + 1][0]):
              insertcount = insertcount + 1
              onreadinsert = onreadinsert + insertlist[readcount][insertcount][1]
            slicelength = slicelength + onreadinsert


          tmpparray = np.array([read[locinread: locinread + slicelength], cigarlist[readcount][locinread: locinread + slicelength], qualitylist[readcount][locinread: locinread + slicelength]])
          locinread = locinread + slicelength

          if(insertpresentonkey):

            remainlengh = insertinfo[key] - onreadinsert
            insertcount = insertcount + 1
            if(remainlengh > 0):
              
              tmpparray = np.column_stack((tmpparray, np.ones((3, remainlengh)) * np.array([[0.], [6.], [0.]])))
          
          else:

            tmpparray = np.column_stack((tmpparray, np.ones((3, insertinfo[key])) * np.array([[0.], [6.], [0.]])))


          try:

            parray = np.column_stack((parray, tmpparray))
          
          except:

            parray = tmpparray
          
  
          break

      
      if(locinread !=  len(read) and len(keylist) > 0):

        parray = np.column_stack((parray, np.array([read[locinread: ], cigarlist[readcount][locinread: ], qualitylist[readcount][locinread: ]])))
        
        #print()
      else:

        parray = np.array([read[locinread: ], cigarlist[readcount][locinread: ], qualitylist[readcount][locinread: ]])
      
      readcount = readcount + 1
        
      if(state):

        pallarray = np.column_stack((pallarray, parray))
        #print(parray.shape)


          
      else:

        pallarray = parray
        readlength = parray.shape[1]
        state = True
        
  
  #print('readlisttime',time.time() - readlisttime)
  #print(time.time() - totalstarttime, paddeltime)
  
  return pallarray[0].reshape(int(pallarray.shape[1] / readlength), readlength), pallarray[1].reshape(int(pallarray.shape[1] / readlength), readlength), pallarray[2].reshape(int(pallarray.shape[1] / readlength), readlength), refseq, np.array(softcliplist, dtype = 'object'), seqbase
def fx(alist, blist, clist, rowcount):
    for b in blist:
        alist.append(b)
        clist.append(rowcount)
def chioce_top18(tensor):
    batch_size, window_size, rowcount = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    tensor = tf.concat([tensor, tf.zeros([batch_size, window_size, 18])], axis = 2)
    return tf.reshape(tf.gather(tensor, tf.argsort(tf.reduce_sum(tensor, 1, keepdims = True), axis = 2), axis=2, batch_dims=1)[:,:,:,-18:], [tensor.shape[0], tensor.shape[1], 18, 1])
    
#position SIZE cluster
from numba.pycc import CC
from numba import jit, njit
from numba.typed import List
import numpy as np

cc = CC('mamnet')

@njit
def unpacklist(typed_result):
    return [i[0] for i in typed_result]
@njit
def unpacklist_a(typed_result):
    return [i for i in typed_result]
@njit
def combinelist(listoflist):#require same type
    combinelist = listoflist[0]
    for onelist in listoflist[1:]:
        for item in onelist:
            combinelist.append(item)
    return combinelist 
@njit('int64(unicode_type)')
def str_to_int(s):
    final_index, result = len(s) - 1, 0
    for i,v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result
@njit('ListType(int64)(unicode_type, unicode_type)')
def c_ssee(cigar, refstartstr):#tested

    zero = ord('0')
    INSsyb = ord('I') - zero
    SOFTsyb = ord('S') - zero
    HARDsyb = ord('H') - zero
    PADsyb = ord('P') - zero
    DELsyb = ord('D') - zero
    SKIPsyb = ord('N') - zero
    delsyb = ord('^') - zero
    
    readloc = 0
    refloc = str_to_int(refstartstr)

    
    
    meetstart = False

    typed_cigar = List()
    [typed_cigar.append(ord(item) - zero) for item in cigar] 
    number = 0
    for item in typed_cigar:
        if(item < 10):
            number = number * 10 + item

        else:
            if(item != INSsyb and item != SOFTsyb and item != HARDsyb and item != PADsyb):
                if(meetstart == False):
                    meetstart = True
                    refstart = refloc 
                    readstart = readloc
                if(item != DELsyb and item != SKIPsyb):
                    readloc += int(number)
                    readend = readloc
                refloc += int(number)
                number = 0

            else:

                if(item == INSsyb or item == SOFTsyb):

                    readloc += int(number)

                number = 0
    return List([refstart, refloc, readstart, readend])
@cc.export('g_d', 'float64[:,:](ListType(unicode_type), ListType(unicode_type), ListType(ListType(int64)), int64, int64[:], int64)')
def fast_info_P(mdtaglist, cigarlist, corposlist, end, qualityarray, maxcountread):
                             #      0              1              2               3               4             5             6            7         8
    info = np.zeros((end, 9))#MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH

    zero = ord('0')
    INSsyb = ord('I') - zero
    SOFTsyb = ord('S') - zero
    HARDsyb = ord('H') - zero
    PADsyb = ord('P') - zero
    DELsyb = ord('D') - zero
    delsyb = ord('^') - zero

    

    for readcount in qualityarray:
        
        cigar = cigarlist[readcount]
        loc = corposlist[readcount][0]
        weight = 1.
        typed_cigar = List()
        [typed_cigar.append(ord(item) - zero) for item in cigar] 
        number = 0.
        for item in typed_cigar:
            if(item < 10):
                number = number * 10. + item
  
            else:
                if(item != INSsyb and item != SOFTsyb and item != HARDsyb and item != PADsyb):
                    if(loc >= 0 and info[loc][8] < maxcountread and item == DELsyb and loc < end):
                        info[loc][6] += number
                        info[loc][7] = max(info[loc][7], number)
                        
                        
                        
                    loc += int(number)
                    number = 0.
                    if(loc >= end):
                        break
                else:

                    if(loc >= 0 and info[loc][8] < maxcountread):
                        if(item == INSsyb):
                            info[loc][4] += number
                            info[loc][5] = max(info[loc][5], number)
                            info[loc][3] += 1.
                            number = 0.
                            continue
                        if(item == SOFTsyb or item == HARDsyb):
                            info[loc][2] += 1.
                            number = 0.
                            continue
                        
                        number = 0.
                    else:
                        number = 0.
                        

    
        
        typed_mdtag = List()
        [typed_mdtag.append(ord(item) - zero) for item in mdtaglist[readcount]] 
        matchnumber = 0
        loc = corposlist[readcount][0]
        indeletion = False
        inmatch = False
        for item in typed_mdtag:

            if(item < 10):
                matchnumber = matchnumber * 10 + item 
                indeletion = False
                inmatch = True
                continue

            if(inmatch):
                loc += matchnumber
            #print(matchnumber)
                matchnumber = 0
                inmatch = False 

            if(item == delsyb):
                indeletion = True
                continue

            if(loc >= end):
                break

            if(loc >= 0 and info[loc][8] < maxcountread):
                if(indeletion == False):
                    info[loc][0] += 1.
                else:
                    info[loc][1] += 1.


            loc += 1
            if(inmatch):
                loc += matchnumber
                
        info[max(corposlist[readcount][0], 0): min(corposlist[readcount][1], end), 8:9] += 1.
    
    info[:,4:5] = info[:,4:5] / (info[:,3:4] + 1e-20)
    info[:,6:7] = info[:,6:7] / (info[:,1:2] + 1e-20)
    #info[:,0:4] = info[:,0:4] / (info[:,8:9]+ 1e-20)

    return info
@njit
def cluster_fn(num_total_indelinfo):
    thorhold = 200
    order = np.argsort(np.array(unpacklist(num_total_indelinfo)))
    
    

    numinfo = num_total_indelinfo[order[0]]

    cluster_result = List([numinfo[:2]])
    


    
    cluster_readcount = List([1])
    cluster_split_id = List()
    prestart = numinfo[0]
    pos_cache = List([numinfo[0]])
    size_cache = List([numinfo[1]])
    split_id_cache = List([numinfo[2]])

    cluster_count = 0
    C_skipedrec = List([numinfo])#transfer all data to P_skipedrec when initial new cluster
    P_skipedrec = List([numinfo])

    for locationinorder in order[1:]:
        numinfo = num_total_indelinfo[locationinorder]
        numinfoused = False
        
        while(True):
            if(numinfoused == False):
                numinfoused = True
            else:
                if(len(P_skipedrec) == 1):
                    break
                else:
                    numinfo = P_skipedrec.pop(1)
            
            if((prestart+thorhold)>= numinfo[0]): #add in current cluster
                if((numinfo[1] * size_cache[-1]) > 0):
                    if((min(abs(size_cache[-1]), abs(numinfo[1])) / max(abs(size_cache[-1]), abs(numinfo[1]))) > 0.7):
                        pos_cache.append(numinfo[0])
                        size_cache.append(numinfo[1])
                        cluster_readcount[-1] += 1
                        if(numinfo[2] != 0):
                            split_id_cache.append(numinfo[2])
                        prestart = numinfo[0]
                        continue
                C_skipedrec.append(numinfo)
            else:#create new cluster
                cluster_result[-1][0] = pos_cache[cluster_readcount[-1]//2]
                size_cache.sort()
                cluster_result[-1][1] = size_cache[cluster_readcount[-1]//2]
                cluster_split_id.append(split_id_cache)

                while(len(C_skipedrec) != 1):
                    P_skipedrec.append(C_skipedrec.pop(1))

                if(len(P_skipedrec) != 1):
                    P_skipedrec.append(numinfo)
                    numinfo = P_skipedrec.pop(1)

                pos_cache = List([numinfo[0]])
                size_cache = List([numinfo[1]])
                split_id_cache = List([numinfo[2]])



                cluster_result.append(numinfo[:2])

                cluster_count += 1
                cluster_readcount.append(1)
                prestart = numinfo[0]
    
    if(len(C_skipedrec) !=1):
        while(len(C_skipedrec) != 1):
            P_skipedrec.append(C_skipedrec.pop(1))
        while(True):
            numinfovaild = False
            if(len(P_skipedrec) != 1):
                numinfo = P_skipedrec.pop(1)
                numinfovaild = True
            else:
                if(len(C_skipedrec) == 1):
                    break       
            if(((prestart+thorhold)>= numinfo[0]) and (numinfovaild == True)): #add in current cluster
                if((numinfo[1] * size_cache[-1]) > 0):
                    if((min(abs(size_cache[-1]), abs(numinfo[1])) / max(abs(size_cache[-1]), abs(numinfo[1]))) > 0.7):
                        pos_cache.append(numinfo[0])
                        size_cache.append(numinfo[1])
                        cluster_readcount[-1] += 1
                        if(numinfo[2] != 0):
                            split_id_cache.append(numinfo[2])
                        prestart = numinfo[0]
                        continue
                C_skipedrec.append(numinfo)
            else:#create new cluster
                cluster_result[-1][0] = pos_cache[cluster_readcount[-1]//2]
                size_cache.sort()
                cluster_result[-1][1] = size_cache[cluster_readcount[-1]//2]

                while(len(C_skipedrec) != 1):
                    P_skipedrec.append(C_skipedrec.pop(1))

                if(len(P_skipedrec) != 1):
                    if(numinfovaild == True):
                        P_skipedrec.append(numinfo)
                    numinfo = P_skipedrec.pop(1)
                    numinfovaild = True
                if(numinfovaild == True):
                    pos_cache = List([numinfo[0]])
                    size_cache = List([numinfo[1]])
                    cluster_split_id.append(split_id_cache)
                    split_id_cache = List([numinfo[2]])
                    cluster_result.append(numinfo[:2])
                    cluster_count += 1
                    cluster_readcount.append(1)
                    prestart = numinfo[0]
            
            
    cluster_result[-1][0] = pos_cache[cluster_readcount[-1]//2]
    size_cache.sort()
    cluster_result[-1][1] = size_cache[cluster_readcount[-1]//2]
    cluster_split_id.append(split_id_cache)

    confi_order = np.argsort(np.array(unpacklist_a(cluster_readcount)))
    argloc = 1
    used_split_set = set()
    while(argloc<=len(confi_order)):
        loc = confi_order[-argloc]
        for splitid in cluster_split_id[loc]:
            if(splitid == 0):
                continue
            else:
                if(splitid in used_split_set):
                    cluster_readcount[loc] -= 1
                else:
                    used_split_set.add(splitid)
        argloc += 1
    return cluster_result, cluster_readcount


@cc.export('c_cw', 'Tuple((float64[:,:], ListType(ListType(int64)), ListType(int64)))(ListType(unicode_type), ListType(unicode_type), ListType(ListType(int64)), int64, int64, ListType(ListType(unicode_type)), ListType(ListType(int64)), unicode_type, int64[:], int64)')
def c_c_withsa(mdtaglist, cigarlist, corposlist, tstart, end, primaryreadidcontigandsa, primaryssee, svtype, qualityarray, maxcountread):
                             #      0              1              2               3               4             5             6            7         8
    #MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH

    zero = ord('0')
    INSsyb = ord('I') - zero
    SOFTsyb = ord('S') - zero
    HARDsyb = ord('H') - zero
    PADsyb = ord('P') - zero
    DELsyb = ord('D') - zero
    SKIPsyb = ord('N') - zero
    delsyb = ord('^') - zero
    
                             #      0              1              2               3               4             5             6            7         8
    info = np.zeros((end, 9))#MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH


    
    typed_pands_sese = List()
    typed_readidandcontig_sord_list = List()

    
    readcount = 0
    while(readcount<len(primaryreadidcontigandsa)):
        preadid = primaryreadidcontigandsa[readcount][0]
        pcontig = primaryreadidcontigandsa[readcount][1]
        pstrand = '+' if(primaryreadidcontigandsa[readcount][2] == '1') else '-'
        
        for onesa in primaryreadidcontigandsa[readcount][3].split(';')[:-1]:
            pssee = List([primaryssee[readcount][0], primaryssee[readcount][1], primaryssee[readcount][2], primaryssee[readcount][3]])
            listedonesa = List(onesa.split(','))
            samecontig = '1' if(listedonesa[0] == pcontig) else '0'
            samestrand = '1' if(listedonesa[2] == pstrand) else'0'
            typed_readidandcontig_sord_list.append(List([preadid, pcontig, listedonesa[0], samecontig, samestrand]))
            sssee = c_ssee(listedonesa[3], listedonesa[1])
            

            for item in sssee:
                pssee.append(item)
            typed_pands_sese.append(pssee)
            

            
            
            
        readcount += 1

    typed_delresult, typed_insresult = List(), List()
    splitcount = 1
    thorhold = 2000
    for loc in range(len(typed_readidandcontig_sord_list)):
        stringinfo, numinfo = typed_readidandcontig_sord_list[loc], typed_pands_sese[loc]
        if(stringinfo[3] == '0' or stringinfo[4] == '0'):
            continue
            
        refend_1, readend_1, refstart_2, readstart_2 =  numinfo[1] , numinfo[3], numinfo[4], numinfo[6]
        refgap = (refend_1 - refstart_2)
        readgap = (readend_1 - readstart_2)
        dif = -readgap + refgap
        if(dif<0 and (abs(readgap)<thorhold)):#deletion
            cstart = min(refend_1, refstart_2)
            typed_delresult.append(List([cstart-tstart, dif, splitcount]))
     
       
            splitcount += 1

            
            
        elif(dif>0 and (abs(refgap)<thorhold)):#insertion
            typed_insresult.append(List([refend_1-tstart, dif, splitcount]))

   
                
            typed_insresult.append(List([refstart_2-tstart, dif, splitcount]))
        
   
                
            splitcount += 1
        
        
    ################################

    delload, insload = False, False
    thorhold = 20
    delreadcount, insreadcount = -1, -1

    for readcount in qualityarray:
        
        cigar = cigarlist[readcount]
        loc = corposlist[readcount][0]
        weight = 1
        typed_cigar = List()
        [typed_cigar.append(ord(item) - zero) for item in cigar] 
        number = 0
        for item in typed_cigar:
            if(item < 10):
                number = number * 10 + item
  
            else:
                if(item != INSsyb and item != SOFTsyb and item != HARDsyb and item != PADsyb):
                    if(loc >= 0 and item == DELsyb and loc < end):
                        if(info[loc][8] < maxcountread):
                            info[loc][6] += number+0.
                            info[loc][7] = max(info[loc][7], number)+0.
                        if(number>thorhold):
                            if(delload and (delreadcount == readcount) and (((number-typed_delresult[-1][1])/(loc+number-typed_delresult[-1][0]))>0.5)):
                                typed_delresult[-1] = List([typed_delresult[-1][0], -(loc+number-typed_delresult[-1][0]), 0])
                                
                            else:
                                typed_delresult.append(List([loc, -number, 0]))
                                delload = True
                                delreadcount = readcount

                        
                        
                        
                    loc += int(number)
                    number = 0
                    if(loc >= end):
                        break
                else:

                    if(loc >= 0 ):
                        if(item == INSsyb):
                            if(info[loc][8] < maxcountread):
                                info[loc][4] += number+0.
                                info[loc][5] = max(info[loc][5], number)+0.
                                info[loc][3] += 1.
               
                            if(number>thorhold):
                                if(insload and (insreadcount == readcount) and (((number+typed_insresult[-1][1])/(loc+number+typed_insresult[-1][1]-typed_insresult[-1][0]))>0.5)):
                                    typed_insresult[-1] = List([typed_insresult[-1][0], typed_insresult[-1][1]+number, 0])
                                    
                                else:
                                    typed_insresult.append(List([loc, number, 0]))
                                    insload = True
                                    insreadcount = readcount

                            number = 0
                            continue
                        if(item == SOFTsyb or item == HARDsyb):
                            if(info[loc][8] < maxcountread):
                                info[loc][2] += 1.
                            number = 0
                            continue
                        
                        number = 0
                    else:
                        number = 0

            
    #########
        typed_mdtag = List()
        [typed_mdtag.append(ord(item) - zero) for item in mdtaglist[readcount]] 
        matchnumber = 0
        loc = corposlist[readcount][0]
        indeletion = False
        inmatch = False
        for item in typed_mdtag:

            if(item < 10):
                matchnumber = matchnumber * 10 + item 
                indeletion = False
                inmatch = True
                continue

            if(inmatch):
                loc += matchnumber
            #print(matchnumber)
                matchnumber = 0
                inmatch = False 

            if(item == delsyb):
                indeletion = True
                continue

            if(loc >= end):
                break

            if(loc >= 0 and info[loc][8] < maxcountread):
                if(indeletion == False):
                    info[loc][0] += 1.
                else:
                    info[loc][1] += 1.


            loc += 1
            if(inmatch):
                loc += matchnumber
                
        info[max(corposlist[readcount][0], 0): min(corposlist[readcount][1], end), 8:9] += 1.
    
    info[:,4:5] = info[:,4:5] / (info[:,3:4] + 1e-20)
    info[:,6:7] = info[:,6:7] / (info[:,1:2] + 1e-20)
    #########


    delfail = False
    if(len(typed_delresult) != 0):
        cluster_result, cluster_readcount = cluster_fn(typed_delresult)
    else:
        delfail = True


    insfail = False
    if(len(typed_insresult) != 0):
        if(delfail == False):
            ##################################
            tmpcluster_result, tmpcluster_readcount = cluster_fn(typed_insresult)
            cluster_result = combinelist(List([cluster_result, tmpcluster_result]))
            cluster_readcount = combinelist(List([cluster_readcount, tmpcluster_readcount]))

        else:
            cluster_result, cluster_readcount = cluster_fn(typed_insresult)
    else:
        insfail = True
    
    if(insfail == True and delfail == True):
        return info, List([List([0, 0])]), List([0])

    return info, cluster_result, cluster_readcount


@cc.export('c_cn', 'Tuple((float64[:,:], ListType(ListType(int64)), ListType(int64)))(ListType(unicode_type), ListType(unicode_type), ListType(ListType(int64)), int64, int64, unicode_type, int64[:], int64)')
def c_c_nosa(mdtaglist, cigarlist, corposlist, tstart, end, svtype, qualityarray, maxcountread):
                             #      0              1              2               3               4             5             6            7         8
    #MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH

    zero = ord('0')
    INSsyb = ord('I') - zero
    SOFTsyb = ord('S') - zero
    HARDsyb = ord('H') - zero
    PADsyb = ord('P') - zero
    DELsyb = ord('D') - zero
    SKIPsyb = ord('N') - zero
    delsyb = ord('^') - zero
    
                             #      0              1              2               3               4             5             6            7         8
    info = np.zeros((end, 9))#MISMATCHCOUNT, DELETIONCOUNT, SOFTHARDCOUNT, INSERTIONCOUNT, INSERTIONMEAN, INSERTIONMAX, DELETIONMEAN, DELETIONMAX, DEPTH


    
    

    typed_delresult, typed_insresult = List(), List()
    
        
    ################################

    delload, insload = False, False
    thorhold = 20
    delreadcount, insreadcount = -1, -1

    for readcount in qualityarray:
        
        cigar = cigarlist[readcount]
        loc = corposlist[readcount][0]
        weight = 1
        typed_cigar = List()
        [typed_cigar.append(ord(item) - zero) for item in cigar] 
        number = 0
        for item in typed_cigar:
            if(item < 10):
                number = number * 10 + item
  
            else:
                if(item != INSsyb and item != SOFTsyb and item != HARDsyb and item != PADsyb):
                    if(loc >= 0 and item == DELsyb and loc < end):
                        if(info[loc][8] < maxcountread):
                            info[loc][6] += number+0.
                            info[loc][7] = max(info[loc][7], number)+0.
                        if(number>thorhold):
                            if(delload and (delreadcount == readcount) and (((number-typed_delresult[-1][1])/(loc+number-typed_delresult[-1][0]))>0.5)):
                                typed_delresult[-1] = List([typed_delresult[-1][0], -(loc+number-typed_delresult[-1][0]), 0])
                                
                            else:
                                typed_delresult.append(List([loc, -number, 0]))
                                delload = True
                                delreadcount = readcount

                        
                        
                        
                    loc += int(number)
                    number = 0
                    if(loc >= end):
                        break
                else:

                    if(loc >= 0 ):
                        if(item == INSsyb):
                            if(info[loc][8] < maxcountread):
                                info[loc][4] += number+0.
                                info[loc][5] = max(info[loc][5], number)+0.
                                info[loc][3] += 1.
               
                            if(number>thorhold):
                                if(insload and (insreadcount == readcount) and (((number+typed_insresult[-1][1])/(loc+number+typed_insresult[-1][1]-typed_insresult[-1][0]))>0.5)):
                                    typed_insresult[-1] = List([typed_insresult[-1][0], typed_insresult[-1][1]+number, 0])
                                    
                                else:
                                    typed_insresult.append(List([loc, number, 0]))
                                    insload = True
                                    insreadcount = readcount

                            number = 0
                            continue
                        if(item == SOFTsyb or item == HARDsyb):
                            if(info[loc][8] < maxcountread):
                                info[loc][2] += 1.
                            number = 0
                            continue
                        
                        number = 0
                    else:
                        number = 0

            
    #########
        typed_mdtag = List()
        [typed_mdtag.append(ord(item) - zero) for item in mdtaglist[readcount]] 
        matchnumber = 0
        loc = corposlist[readcount][0]
        indeletion = False
        inmatch = False
        for item in typed_mdtag:

            if(item < 10):
                matchnumber = matchnumber * 10 + item 
                indeletion = False
                inmatch = True
                continue

            if(inmatch):
                loc += matchnumber
            #print(matchnumber)
                matchnumber = 0
                inmatch = False 

            if(item == delsyb):
                indeletion = True
                continue

            if(loc >= end):
                break

            if(loc >= 0 and info[loc][8] < maxcountread):
                if(indeletion == False):
                    info[loc][0] += 1.
                else:
                    info[loc][1] += 1.


            loc += 1
            if(inmatch):
                loc += matchnumber
                
        info[max(corposlist[readcount][0], 0): min(corposlist[readcount][1], end), 8:9] += 1.
    
    info[:,4:5] = info[:,4:5] / (info[:,3:4] + 1e-20)
    info[:,6:7] = info[:,6:7] / (info[:,1:2] + 1e-20)
    #########


    delfail = False
    if(len(typed_delresult) != 0):
        cluster_result, cluster_readcount = cluster_fn(typed_delresult)
    else:
        delfail = True


    insfail = False
    if(len(typed_insresult) != 0):
        if(delfail == False):
            ##################################
            tmpcluster_result, tmpcluster_readcount = cluster_fn(typed_insresult)
            cluster_result = combinelist(List([cluster_result, tmpcluster_result]))
            cluster_readcount = combinelist(List([cluster_readcount, tmpcluster_readcount]))

        else:
            cluster_result, cluster_readcount = cluster_fn(typed_insresult)
    else:
        insfail = True
    
    if(insfail == True and delfail == True):
        return info, List([List([0, 0])]), List([0])

    return info, cluster_result, cluster_readcount

try:
  import mamnet
except:
  cc.compile()
def labeldata(vcfpath, contig, start, end):
  goldl = []
  window_size = 200
  index = start + np.column_stack((np.arange(0, window_size * (int((end - start - 1) / window_size) + 1), window_size).reshape((int(( end - start - 1) / window_size) + 1), 1), np.arange(0, window_size * (int((end - start - 1) / window_size) + 1), window_size).reshape((int((end - start - 1) / window_size) + 1), 1) + window_size))
  if('chr' in contig):
    contig = contig[3:]
  for rec in pysam.VariantFile(vcfpath).fetch():

    if(rec.contig != contig):
      continue            
    if((rec.info['SVTYPE'] == 'DEL')):
      goldl.append([rec.start, rec.stop, rec.stop - rec.start, 1])
        
  
    
  goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')


  y = []
  for rec in index:
        
    if(((goldl[:,1:2] > rec[0]) & (goldl[:,:1] < rec[1])).sum() != 0):
      y.append((((goldl[:,1:2] > rec[0]) & (goldl[:,:1] < rec[1])) * goldl[:,3:]).sum())


    else:
      y.append(0)
  return (np.array(y)>0).astype('float32')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pysam
from scipy.sparse import coo_matrix
import time
import numpy as np
import tensorflow as tf
import pysam
import matplotlib.pyplot as plt
def fx(alist, blist, clist, rowcount):
  for b in blist:
    alist.append(b)
    clist.append(rowcount)
def chioce_top18(tensor):
  batch_size, window_size, rowcount = tensor.shape[0], tensor.shape[1], tensor.shape[2]
  tensor = tf.concat([tensor, tf.zeros([batch_size, window_size, 18])], axis = 2)
  return tf.reshape(tf.gather(tensor, tf.argsort(tf.reduce_sum(tensor, 1, keepdims = True), axis = 2), axis=2, batch_dims=1)[:,:,:,-18:], [tensor.shape[0], tensor.shape[1], 18, 1])

def myshow(bamfile, contig, window_size = 200):
  block = 10000000
  fmlist = []
  masklist = []
  indexlist = []
  for teststart in range(0, 500000000, block):
    collist = []
    rowlist = []
    rowend = []
    count = 0
    maxend = 0
    minstart = 999999999
    rowcount = 0
    overlap = False
    for AlignedSegment in bamfile.fetch(contig, teststart, teststart + block):
      overlap = True
      count += 1
      seqqosition =(AlignedSegment.get_reference_positions())
      cstart = seqqosition[0]
      cend = seqqosition[-1]
      seqqosition = set(range(cstart, cend+1)) - set(seqqosition)
      newrow = True
      loc = -1
      for oneend in rowend:
        loc += 1
        if(oneend<cstart):
          fx(collist, seqqosition, rowlist, loc)
          rowend[loc] = cend
          newrow = False
          break
      if(newrow == True and (rowcount < 100)):
        rowcount += 1
        rowend.append(cend)
        fx(collist, seqqosition, rowlist, len(rowend)-1)
      if(maxend<cend):
        maxend = cend
      if(minstart>cstart):
        minstart=cstart
    if(overlap == False):
      trueend = teststart
      continue
    minstart = min(teststart, minstart)
    maxend = max(teststart+block, maxend + 1)
    row  = np.array(rowlist)
    col  = np.array(collist)-minstart
    data = np.ones(col.size, dtype = np.float32)
    fm = (coo_matrix((data, (row, col)), shape=(len(rowend), (maxend-minstart))).toarray()[:,teststart - minstart:teststart - minstart + block]).T

    oshape = fm.shape

    fm = fm.reshape(fm.shape[0]//window_size, window_size, len(rowend))

    fm = chioce_top18(fm).numpy()
    mask = (fm.reshape(-1, window_size*18).sum(axis = 1) != 0)
    fm = fm.reshape(-1, window_size*18)[mask]
    fmlist.append(fm)
    indexlist.append(np.arange(teststart, teststart+block, window_size)[mask])
    masklist.append(mask)
  fm = np.concatenate(fmlist, axis = 0)  
  index = np.concatenate(indexlist, axis = 0) 

  timestep = 100
  if(fm.shape[0]<timestep):
    return np.array(0), fm.reshape(1, fm.shape[0], window_size, 18, 1), (minstart, maxend + 1), index
    
  tail = fm.shape[0]%timestep
  if(tail == 0):
    return fm.reshape(fm.shape[0]//timestep, timestep, window_size, 18, 1), np.array(0), (minstart, maxend + 1), index
      
  topdata, taildata = fm[:-tail], fm[-tail:].reshape(1, tail, window_size, 18, 1)
  return topdata.reshape(topdata.shape[0]//timestep, timestep, window_size, 18, 1), taildata, (0, trueend), index
def labeldata(vcfpath, contig, start, end, window_size, index):
  goldl = []
  if('chr' in contig):
    contig = contig[3:]
  for rec in pysam.VariantFile(vcfpath).fetch():

    if(rec.contig != contig):
      continue            
    if((rec.info['SVTYPE'] == 'DEL')):
      goldl.append([rec.start, rec.stop, rec.stop - rec.start, 1])
        
  
    
  goldl = (pd.DataFrame(goldl).sort_values([0, 1]).values).astype('float64')


  y = []
  for rec in index:
        
    if(((goldl[:,1:2] > rec) & (goldl[:,:1] < (rec+window_size))).sum() != 0):
      y.append((((goldl[:,1:2] > rec) & (goldl[:,:1] < (rec+window_size))) * goldl[:,3:]).sum())


    else:
      y.append(0)
  return (np.array(y)>0).astype('float32')
def one_fn(contig, bamfilepath, outputpath = '', vcfpath = '', window_size = 200):
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 64)
    data1, data2, startend, index = myshow(bamfile, contig, window_size)
    start, end = startend
    print(startend)
    print(data1.shape, data2.shape)
    print(index.shape)
    label, label1, label2 = np.array([0]), np.array([0]), np.array([0])
    if(vcfpath != ''):
        label = labeldata(vcfpath, contig, start, end, window_size, index)
        if(data1.size != 1 and data2.size != 1):
            label1, label2 = label[:data1.shape[0]*data1.shape[1]], label[data1.shape[0]*data1.shape[1]:]
    if(data1.size != 1 and data2.size != 1):
        index1, index2 = index[:data1.shape[0]*data1.shape[1]], index[data1.shape[0]*data1.shape[1]:]



    if(data1.size != 1):
        if(label1.size != 1):
            np.savez(outputpath+contig+','+str(start)+','+str(end)+',data1', data = data1, label = label1, index = index1)
        else:
            np.savez(outputpath+contig+','+str(start)+','+str(end)+',data1', data = data1, label = label, index = index)



    if(data2.size != 1):
        if(label1.size != 1):
            np.savez(outputpath+contig+','+str(start)+','+str(end)+',data2', data = data2, label = label2, index = index2)
        else:
            np.savez(outputpath+contig+','+str(start)+','+str(end)+',data2', data = data2, label = label, index = index)

def feature_matrix(bamfilepath, outputpath = '', vcfpath = '', window_size = 200):
    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)
    for contig in [rec.contig for rec in bamfile.get_index_statistics()]:
        one_fn(contig, bamfilepath, outputpath, vcfpath, window_size)



import time
from collections import Counter
import numpy as np
import pysam
import time
import time
import pysam
from pysam import VariantFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
def cigarinfo(cigararray, refstartarray, start, end, reccount, cigarweight):

    a = tf.reshape(tf.cast(~((cigararray[:,0] == 1) | (cigararray[:,0] == 4)), 'float32'), [cigararray.shape[0], 1]) * cigararray
    a1 = tf.reshape(a[:,1], [reccount, cigarweight])

    a = tf.concat([cigararray, tf.reshape(tf.matmul(a1, tf.linalg.band_part(tf.ones([a1.shape[1] , a1.shape[1]], tf.float32), 0, -1)) + refstartarray, [cigararray.shape[0], 1])], axis = 1)


    return tf.boolean_mask(a, (start <= a[:,-1]) & (a[:,-1] < end))
    return tf.boolean_mask(a, (start <= a[:,-2]) & (a[:,-2] < end) & (a[:,0] == 1))[:,1:]



def baseinfo(bamfile, contig, start, end):
    

    cigararray = []
    readstartandend = []
    refpositonlist = []
    refpositonweight = []
    substitionarray, deletionarray, substitionweight, deletionweight = [], [], [], []
    nooverlap = True
    qualityarray = []

    for AlignedSegment in bamfile.fetch(contig, start, end):
    



        cigararray.append(tf.keras.backend.flatten(tf.constant(AlignedSegment.cigartuples)))
        readstartandend.append([AlignedSegment.reference_start-start, AlignedSegment.reference_end-start, AlignedSegment.mapping_quality, (1 - (AlignedSegment.query_alignment_length / AlignedSegment.infer_read_length()))**2])

        nooverlap = False

    
    if(nooverlap):
        print(dsahjdaj)
    readstartandend = tf.constant(readstartandend, tf.float32)
    cigararray = tf.keras.preprocessing.sequence.pad_sequences(cigararray)
    reccount, cigarweight = cigararray.shape[0], int(cigararray.shape[1] / 2)
    cigararray = cigararray.reshape(int(cigararray.size / 2), 2)

    cigararray = cigarinfo(cigararray, readstartandend[:,:1], 0, end - start, reccount, cigarweight).numpy().astype('int64')
    a = cigararray[(cigararray[:,0] == 2) & (cigararray[:,1] > 20)]
    if(a.size == 0):
        return []
    a[:,-1] = a[:,-1] - a[:,-2] 
    delsig = np.column_stack((a[:,-1:], a[:,-2:-1]))



    loc = np.array(list(delsig))[:,0]
    binnum = 20
    binresult = (loc//binnum)
    mc = Counter(binresult).most_common(1)[0]
    sp = mc[1]
    minv, maxv = mc[0]-1, mc[0]+1
    tmp = np.median(np.array(list(delsig))[(minv<=binresult) *  (maxv>= binresult)], axis = 0).astype('int64')
    tmp[0] = tmp[0]+start
    return tmp.tolist()+[sp]
    
def baseinfo_main_binsaver(bamfilepath, delloc):




    bamfile = pysam.AlignmentFile(bamfilepath, 'rb', threads = 20)


    delsig = []
    for rec in delloc:
        contig, start, end = str(rec[1]), int(rec[2]), int(rec[3])

        delsig.append(baseinfo(bamfile, contig, start, end))
    return delsig
