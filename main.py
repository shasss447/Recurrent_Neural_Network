from data import train_data,test_data
import numpy as np
from rnn import RNN

vocab=list(set([w for i in train_data.keys() for w in i.split(' ')]))
vc_sz=len(vocab)

w_i={w:i for i,w in enumerate(vocab)}
i_w={i:w for i,w in enumerate(vocab)}

def createInputs(text):
    inputs=[]
    for w in text.split(' '):
        v=np.zeros((vc_sz,1))
        v[w_i[w]]=1
        inputs.append(v)

    return inputs

def softmax(xs):
        return np.exp(xs)/sum(np.exp(xs))

rnn=RNN(vc_sz,2)
i=createInputs('i am very good')
o,h=rnn.forward(i)
pr=softmax(o)
print(pr)