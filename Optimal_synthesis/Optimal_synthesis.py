
# coding: utf-8



import pandas as pd
import numpy as np
###get_ipython().magic('matplotlib inline')
pd.options.mode.chained_assignment = None


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from catboost import CatBoostRegressor
import scipy


model_area=CatBoostRegressor()
model_size=CatBoostRegressor()
model_defects=CatBoostRegressor()




model_area.load_model('area_model.bin')
model_size.load_model('size_model.bin')
model_defects.load_model('defects_model.bin')




Data=pd.read_csv('Syntetic_dataset.csv')




Area=Data['Area'].to_numpy()
Size=Data['Size'].to_numpy()
Defects=Data['Defects'].to_numpy()



def best(x):
    
    result=0
    minim=1000
    for i in range(0,len(Data)):
        
        dataset=[Area[i],Size[i],Defects[i]]
        razn=mean_absolute_percentage_error(x,dataset)
        
        if razn<=minim:
            minim=razn
            result=i
            
    return result, minim



def day_sintes(x):
    result,_=best(x)
    return Data[result:result+1]


def x_into_asd(x, Md,ls,zs):
    
    Zr=x[0]
    BDC=x[1]
    
    mod=pd.Series(Md)
    
    mod_area=mod.map({'None':1105.0042574257425,'AcOH':1320.229446605375,'Form':1448.2931368136813,  
                                'TFA':1402.6343131188119,'HCl':1296.5647502250224,'NH3':1163.0186262376237,
                                'HF':1162.8317281728173,'BzOH':1335.2527227722774})
    
    mod_size=mod.map({'None':108.86559923906148,'AcOH':241.420618322346,'Form':311.6533132530121, 
                         'TFA':272.86463855421687, 'HCl':261.75611015490534,'NH3':109.53659638554217,
                         'HF':416.0487951807229,'BzOH':342.44948364888126})
    
    mod_defects=mod.map({'None':4.7632812208034885,'AcOH':4.573278247393633,
                          'Form':4.5246952662721895,'TFA':4.5260803043110736,'HCl':5.006167246780369,
                          'NH3':4.225335798816568,'HF':4.472562130177515,'BzOH':4.82824091293322})
    
    
    
    last=pd.Series(ls)
    
    last_area=last.map({'None':1286.1490099099,'MeOH':1224.8794334433444,
                         'Water':1263.2298019801979,'Acetone':1370.0343543729375,
                         'Ethanol':1211.734313118812,'DMF':1364.628279276203,'DCM':1430.8457508250824,
                         'iPrOH':1286.14900990099,'CHCl3':1286.14900990099,'THF':1211.734313118812})
    
        
    last_size=last.map({'None':236.29277108433735,'MeOH':229.5665249232223,'Water':280.8495983935743,
                        'Acetone':169.5081325301205,'Ethanol':151.39025191675793,'DMF':328.97561279601166,
                        'DCM':326.85773092369476,                            
                            'iPrOH':236.29277108433735,'CHl3':236.29277108433735,'THF':189.23519793459553})
        
    last_defects=last.map({'None':4.6776863905325445,'MeOH':4.754125889717863,
                               'Water':4.6776863905325445,'Acetone':4.3539228796844185,
                               'Ethanol':4.729187376725838,'DMF':4.891649641856121,
                               'DCM':4.2756405325443785,                            'iPrOH':4.6776863905325445,
                               'CHCl3':4.6776863905325445,'THF':4.672187376725838})
    
    mod_conc=x[2]
    water=x[3]
    aging=x[4]
    temp=x[5]
    num_of_DMF=x[6]
    total_wash=x[7]
    activation_T=x[8]
    activation_time=x[9]
    synt_time=x[10]
    Zr_source=zs
    
    object_area=pd.DataFrame({'[Zr], M':[Zr],'[BDC], M':[BDC],'Modulator':[mod_area],
                               '[Mod],M':[mod_conc],'[H$_2$O]:[Zr] ratio':[water],'Aging, h':[aging],
                               'T, $^o$C':[temp],'Number of DMF washes':[num_of_DMF],
                               'Total washes':[total_wash],'Last  solvent in pores':[last_area],
                               'Activation T, $^o$C':[activation_T],'Activation time, h':[activation_time],
                               'Time of synthesis, h':[synt_time],'Zr source':[Zr_source]})
    
    object_size=pd.DataFrame({'[Zr], M':[Zr],'[BDC], M':[BDC],'Modulator':[mod_size],
                               '[Mod], M':[mod_conc],'[H$_2$O]:[Zr] ratio':[water],'Aging, h':[aging],
                               'T, $^o$C':[temp],'Number of DMF washes':[num_of_DMF],
                               'Total washes':[total_wash],'Last  solvent in pores':[last_size],
                               'Activation T, $^o$C':[activation_T],'Activation time, h':[activation_time],
                               'Time of synthesis, h':[synt_time],'Zr source':[Zr_source]})
    
    object_defects=pd.DataFrame({'[Zr], M':[Zr],'[BDC], M':[BDC],'Modulator':[mod_defects],
                               '[Mod], M':[mod_conc],'[H$_2$O]:[Zr] ratio':[water],'Aging, h':[aging],
                               'T, $^o$C':[temp],'Number of DMF washes':[num_of_DMF],
                               'Total washes':[total_wash],'Last  solvent in pores':[last_defects],
                               'Activation T, $^o$C':[activation_T],'Activation time, h':[activation_time],
                               'Time of synthesis, h':[synt_time],'Zr source':[Zr_source]})
    
    return [model_area.predict(object_area)[0],
            model_size.predict(object_size)[0],
            model_defects.predict(object_defects)[0]]
    
    
    
    




def dif_check(x):
    
    flag=False
    
    if x[0]>75:
        flag=True
    if x[1]>69:
        flag=True
    if x[2]>0.24:
        flag=True
        
    return flag




class founder():
    
    def __init__(self,y):
        
        self.y=y
        self.zero=day_sintes(y)
        self.md=self.zero['Modulator']
        self.ls=self.zero['Last  solvent in pores']
        
        if (self.zero['Zr source']).to_numpy()[0]=='ZrCl4':
            
            self.zs=0
        else:
            self.zs=1
        
        self.dif=np.zeros(3)
        self.dif[0]=abs(self.zero['Area']-y[0])
        self.dif[1]=abs(self.zero['Size']-y[1])
        self.dif[2]=abs(self.zero['Defects']-y[2])
        
        
        
       
        
        
        
        self.start=[(max([0.01,self.zero['[Zr], M'].to_numpy()[0]*0.7]),min([self.zero['[Zr], M'].to_numpy()[0]*1.3,0.2])),
                    (max([self.zero['[BDC], M'].to_numpy()[0]*0.7,0.01]),min([self.zero['[BDC], M'].to_numpy()[0]*1.3,0.3])),
                    (self.zero['[Mod],M'].to_numpy()[0]*0.7,min([self.zero['[Mod],M'].to_numpy()[0]*1.3+0.1,700])),
                    (self.zero['[H$_2$O]:[Zr] ratio'].to_numpy()[0]*0.7,self.zero['[H$_2$O]:[Zr] ratio'].to_numpy()[0]+10),
                    (self.zero['Aging, h'].to_numpy()[0]*0.7,self.zero['Aging, h'].to_numpy()[0]*1.3),
                    (max([self.zero['T, $^o$C'].to_numpy()[0]*0.7,50]),min([self.zero['T, $^o$C'].to_numpy()[0]*1.3,220])),
                    (0,6),
                    (0,6),
                    (self.zero['Activation T, $^o$C'].to_numpy()[0]*0.7,min([200.0,self.zero['Activation T, $^o$C'].to_numpy()[0]*1.3])),
                    (self.zero['Activation time, h'].to_numpy()[0]*0.7,self.zero['Activation time, h'].to_numpy()[0]*1.3),
                    (max([1,self.zero['Time of synthesis, h'].to_numpy()[0]*0.7]),min([72,self.zero['Time of synthesis, h'].to_numpy()[0]*1.3]))]
        
        
        
    
            
    def zero_visual(self):
        
        if self.zs:
            source='ZrOCl2*8H2O'
        else:
            source='ZrCl4'
                
        ###print(self.start)
        print( 'Optimal synthesis from literature: ')
        print('')
        print ('Zr source: ', source)
    
        print ('[Zr], M: ', self.zero['[Zr], M'].to_numpy()[0])
        print('[BDC], M:', self.zero['[BDC], M'].to_numpy()[0])
    
        print('Modulator: ', self.md.to_numpy()[0])
        print('[Mod]:[Zr] ratio:', round(self.zero['[Mod],M'].to_numpy()[0]))
        print('[H2O]:[Zr] ratio:', self.zero['[H$_2$O]:[Zr] ratio'].to_numpy()[0])
    
        print('Aging, h:', self.zero['Aging, h'].to_numpy()[0])
        print('Temperature:', self.zero['T, $^o$C'].to_numpy()[0])
        print('Time: ', self.zero['Time of synthesis, h'].to_numpy()[0])
    
        print('Number of DMF washes: ', self.zero['Number of DMF washes'].to_numpy()[0])
        print('Total number of washes: ', self.zero['Total washes'].to_numpy()[0])
        print('Activation temperature: ', self.zero['Activation T, $^o$C'].to_numpy()[0])
        print('Activation time: ', self.zero['Activation time, h'].to_numpy()[0])
        
        print('')
        print('Properties:')
        print('Area: ',round(self.zero['Area'].to_numpy()[0]), 'm2/g')
        print('Size: ', round(self.zero['Size'].to_numpy()[0]), 'nm')
        print('BDC to Zr6 ratio: ', round(self.zero['Defects'].to_numpy()[0],2))
    
        
            
    def method(self,x):
        return mean_absolute_percentage_error(self.y,x_into_asd(x,self.md,self.ls,self.zs))
    
    
    def poisk(self):
        
        def method(x):
            result=mean_absolute_percentage_error(self.y,x_into_asd(x,self.md,self.ls,self.zs))

            return result
        
        result=scipy.optimize.differential_evolution(method,self.start,maxiter=20, polish=False).x
        
        self.result=result
        
            
    


def Found_synthesis(x):
    
    S=founder(x)
    
    S.zero_visual()
    
    if dif_check(S.dif):
        print('')
        print('Try to optimize:')
        print('')
        
        
        S.poisk()
        result=S.result
        
        if S.zs:
            source='ZrOCl2*8H2O'
        else:
            source='ZrCl4'
    
        print ('Zr source: ', source)
        print ('[Zr], M: ', result[0])
        print('[BDC], M:', result[1])
    
        print('Modulator: ', S.md.to_numpy()[0])
        print('[Mod]:[Zr] ratio:', result[2])
        print('[H2O]:[Zr] ratio:', result[3])
    
        print('Aging, h:', result[4])
        print('Temperature:', round(result[5]))
        print('Time: ', round(result[9]))
    
        print('Number of DMF washes: ', round(result[6]))
        print('Total number of washes: ', round(result[7]))
        print('Activation temperature: ', round(result[8]))
        print('Activation time: ', round(result[9]))
        
        
        asd=x_into_asd(result,S.md,S.ls,S.zs)
        print('')
        print('Oprimized properties:')
        print('Area: ',round(asd[0]), 'm2/g')
        print('Size: ', round(asd[1]), 'nm')
        print('BDC to Zr6 ratio: ', round(asd[2],2))
    



