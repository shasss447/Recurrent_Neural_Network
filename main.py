from data import train_data,test_data
import numpy as np
from rnn import RNN
import random


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
def processData(data,backprop):
     itm=list(data.items())
     random.shuffle(itm)

     loss=0
     num_c=0
     for x,y in itm:
       inp=createInputs(x)
       tg=int(y)

       out,_=rnn.forward(inp)
       prbs=softmax(out)

       loss-=np.log(prbs[tg])
       num_c+=int(np.argmax(prbs)==tg)

       if backprop:
         dl_dy=prbs
         dl_dy[tg]-=1

         rnn.backprop(dl_dy)

     return loss/len(data),num_c/len(data)

for epoch in range(1000):
    train_loss,train_acc=processData(train_data,True)

    if epoch%100==99:
        print('Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss.item(), train_acc))
        test_loss,test_acc=processData(test_data,False)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss.item(), test_acc))