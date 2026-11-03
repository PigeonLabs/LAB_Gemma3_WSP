"""Bounded FP64 primitive calculator with source-referenced replay."""
import math
import numpy as np

class Calculator:
    def __init__(self, table, limit=32, batch_limit=4):
        data=np.asarray(table,dtype=np.float64)
        if data.ndim!=2 or data.shape[1]!=5 or not np.isfinite(data).all():
            raise ValueError('Invalid coefficient table')
        self.n=len(data); self.limit=limit; self.batch_limit=batch_limit
        self.values={k:data[:,i].copy() for i,k in enumerate(['gX','gY','uX','uY','d'])}
        self.trace=[]; self.attempts=0

    def value(self, arg):
        if isinstance(arg,str):
            if arg not in self.values: raise ValueError('Unknown handle '+arg)
            return self.values[arg]
        if type(arg) in (int,float) and arg in [-1,0,.5,1,2]:
            return np.float64(arg)
        raise ValueError('Use coefficient/result handles or constants -1,0,0.5,1,2')

    def apply(self, calls):
        if not isinstance(calls,list) or not 1<=len(calls)<=self.batch_limit:
            raise ValueError('Expected 1 to 4 operations')
        results=[]
        for call in calls:
            if self.attempts>=self.limit: raise ValueError('Operation budget exhausted')
            self.attempts+=1
            if not isinstance(call,dict) or set(call)!={'id','op','args'}: raise ValueError('Operation schema')
            name,op,args=call['id'],call['op'],call['args']
            if not isinstance(name,str) or not name.isidentifier() or name in self.values:
                raise ValueError('Result id must be new identifier')
            if not isinstance(args,list): raise ValueError('args must be a list')
            arities={'add':2,'sub':2,'mul':2,'div':2,'abs':1,'sum':1,'gelu':1,'take':2}
            if op not in arities or len(args)!=arities[op]: raise ValueError('Unknown operation or arity')
            with np.errstate(all='raise'):
                a=self.value(args[0])
                if op=='take':
                    indices=args[1]
                    if not isinstance(indices,list) or not indices or any(type(i)!=int or not 0<=i<self.n for i in indices):
                        raise ValueError('take indices outside vector')
                    if np.ndim(a)!=1: raise ValueError('take needs vector')
                    result=a[indices]
                elif op=='abs': result=np.abs(a)
                elif op=='sum': result=np.sum(a,dtype=np.float64)
                elif op=='gelu': result=.5*a*(1+np.tanh(math.sqrt(2/math.pi)*(a+.044715*a**3)))
                else:
                    b=self.value(args[1])
                    result={'add':np.add,'sub':np.subtract,'mul':np.multiply,'div':np.divide}[op](a,b)
            result=np.asarray(result,dtype=np.float64)
            if result.ndim>1 or result.size>self.n or not np.isfinite(result).all(): raise ValueError('Invalid numeric result')
            self.values[name]=result
            item={'id':name,'op':op,'args':args,'value':result.tolist()}
            self.trace.append(item); results.append(item)
        return results

    def replay(self, trace):
        for item in trace:
            self.apply([{k:item[k] for k in ['id','op','args']}])
        return self.trace

    def scalar(self, name):
        if not isinstance(name,str): raise ValueError('Final influence must reference a result handle')
        value=self.value(name)
        if np.ndim(value)!=0: raise ValueError('Final influence is not scalar')
        return float(value)
