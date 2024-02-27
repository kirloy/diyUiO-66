
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


# In[317]:


import sklearn 

from sklearn.model_selection import train_test_split


# In[318]:





# In[319]:

import catboost as catboost
from catboost import CatBoostRegressor


class Synthesis():
    
    def __init__(self):
        self.Zr_source=True
        self.Zr_mass=32200
        self.BDC_mass=16000
        self.DMF_volume=500.0
        self.modulator='HCl'
        self.mod_volume=16.0
        self.n_mod=0.192
        self.water=11.93
        self.aging=0
        self.temp=150.0
        self.num_of_DMF=2.0
        self.total_wash=4.0
        self.last_solvent='MeOH'
        self.activation_T=100.0
        self.activation_time=12.0
        self.synt_time=24.0

        self.mod_map_area={'None':1105.0042574257425,'AcOH':1320.229446605375,'HCOOH':1448.2931368136813,                           'TFA':1402.6343131188119,'HCl':1296.5647502250224,'NH3':1163.0186262376237,'HF':1162.8317281728173,'BzOH':1335.2527227722774}
        
        self.mod_map_size={'None':108.86559923906148,'AcOH':241.420618322346,'HCOOH':311.6533132530121,                           'TFA':272.86463855421687,'HCl':261.75611015490534,'NH3':109.53659638554217,'HF':416.0487951807229,'BzOH':342.44948364888126}
        
        self.mod_map_defects={'None':4.7632812208034885,'AcOH':4.573278247393633,'HCOOH':4.5246952662721895,                              'TFA':4.5260803043110736,'HCl':5.006167246780369,'NH3':4.225335798816568,'HF':4.472562130177515,'BzOH':4.82824091293322}
        
        self.last_area_map={'None':1286.1490099099,'MeOH':1224.8794334433444,'Water':1263.2298019801979,'Acetone':1370.0343543729375,                            'Ethanol':1211.734313118812,'DMF':1364.628279276203,'DCM':1430.8457508250824,'iPrOH':1286.14900990099
,'CHCl3':1286.14900990099,'THF':1211.734313118812}
        
        self.last_size_map={'None':236.29277108433735,'MeOH':229.5665249232223,'Water':280.8495983935743,'Acetone':169.5081325301205,                            'Ethanol':151.39025191675793,'DMF':328.97561279601166,'DCM':326.85773092369476,                            'iPrOH':236.29277108433735,'CHl3':236.29277108433735,'THF':189.23519793459553}
        
        self.last_defects_map={'None':4.6776863905325445,'MeOH':4.754125889717863,'Water':4.6776863905325445,'Acetone':4.3539228796844185,                            'Ethanol':4.729187376725838,'DMF':4.891649641856121,'DCM':4.2756405325443785,                            'iPrOH':4.6776863905325445,'CHCl3':4.6776863905325445,'THF':4.672187376725838}
        
        


        
            
        TGA=pd.read_csv('Defects.csv')
        Area=pd.read_csv('Area.csv')
        Size=pd.read_csv('Size.csv')
    
        ###F=Size.drop('Unnamed: 0', axis=1)
        ###W=Area.drop('Unnamed: 0',axis=1)
        ###G=TGA.drop('Unnamed: 0', axis=1)

        F=Size
        W=Area
        G=TGA

        y1=W['Area']
        X1=W.drop('Area',axis=1)
    
    
        y2=F['Size']
        X2=F.drop('Size',axis=1)
    
        y3=G['TGA']
        X3=G.drop('TGA',axis=1)
    
    
        X_train, X_valid, y_train, y_valid=train_test_split(X1,y1, test_size=0.1, random_state=50, stratify=X1['Modulator'])
        model_area = CatBoostRegressor(iterations=200, depth=8, l2_leaf_reg=0.5)
        model_area.fit(X_train,y_train, logging_level='Silent')
    
        X_train, X_valid, y_train, y_valid = train_test_split(X2, y2, test_size=0.1, random_state=44,stratify=X2['Modulator'])
        model_size = CatBoostRegressor(iterations=200, depth=8, l2_leaf_reg=0.5)
        model_size.fit(X_train,y_train, logging_level='Silent')
        
        X_train, X_valid, y_train, y_valid = train_test_split(X3, y3, test_size=0.1, random_state=84,stratify=X3['Modulator'])
        model_defects = CatBoostRegressor(iterations=200, depth=7, l2_leaf_reg=0.3)
        model_defects.fit(X_train,y_train, logging_level='Silent')
        
        self.model_area=model_area
        self.model_size=model_size
        self.model_defects=model_defects
        
        
    def calculate(self):
        
        if self.modulator=='HCl':
            self.tot_vol=self.DMF_volume+self.mod_volume
        else:
            self.tot_vol=self.DMF_volume+self.mod_volume+self.water
            
        if self.Zr_source:
            self.Zr=self.Zr_mass/322.2/self.tot_vol
        else:
            self.Zr=self.Zr_mass/233.2/self.tot_vol
            
        self.BDC=self.BDC_mass/166.13/self.tot_vol
        
        self.mod_conc=self.n_mod/self.tot_vol*1000
        self.mod_conc=self.mod_conc/self.Zr
        
        self.water=self.water/18/self.tot_vol*1000
        self.water=self.water/self.Zr
        
        self.mod=pd.Series(self.modulator)
        
        self.mod_area=self.mod.map(self.mod_map_area)
        self.mod_size=self.mod.map(self.mod_map_size)
        self.mod_defects=self.mod.map(self.mod_map_defects)
        
        self.mod_area=self.mod_area[0]
        self.mod_size=self.mod_size[0]
        self.mod_defects=self.mod_defects[0]
        
        
        self.last_solvent=pd.Series(self.last_solvent)
        
        self.last_area=self.last_solvent.map(self.last_area_map)
        self.last_size=self.last_solvent.map(self.last_size_map)
        self.last_defects=self.last_solvent.map(self.last_defects_map)
        
        self.last_area=self.last_area[0]
        self.last_size=self.last_size[0]
        self.last_defects=self.last_defects[0]
        
        
       
        
        
        self.object_area=pd.DataFrame({'[Zr], M':[self.Zr],                                       '[BDC], M':[self.BDC],                                       'Modulator':[self.mod_area],                                       '[Mod],M':[self.mod_conc],                                       '[H$_2$O]:[Zr] ratio':[self.water],                                       'Aging, h':[self.aging],                                       'T, $^o$C':[self.temp],                                       'Number of DMF washes':[self.num_of_DMF],                                       'Total washes':[self.total_wash],                                       'Last  solvent in pores':[self.last_area],                                       'Activation T, $^o$C':[self.activation_T],                                       'Activation time, h':[self.activation_time],                                       'Time of synthesis, h':[self.synt_time],                                       'Zr source':[self.Zr_source]})
        
        self.object_size=pd.DataFrame({'[Zr], M':[self.Zr],                                       '[BDC], M':[self.BDC],                                       'Modulator':[self.mod_size],                                       '[Mod], M':[self.mod_conc],                                       '[H$_2$O]:[Zr] ratio':[self.water],                                       'Aging, h':[self.aging],                                       'T, $^o$C':[self.temp],                                       'Number of DMF washes':[self.num_of_DMF],                                       'Total washes':[self.total_wash],                                       'Last  solvent in pores':[self.last_size],                                       'Activation T, $^o$C':[self.activation_T],                                       'Activation time, h':[self.activation_time],                                       'Time of synthesis, h':[self.synt_time],                                       'Zr source':[self.Zr_source]})
        
        self.object_defects=pd.DataFrame({'[Zr], M':[self.Zr],                                       '[BDC], M':[self.BDC],                                       'Modulator':[self.mod_defects],                                       '[Mod], M':[self.mod_conc],                                       '[H$_2$O]:[Zr] ratio':[self.water],                                       'Aging, h':[self.aging],                                       'T, $^o$C':[self.temp],                                       'Number of DMF washes':[self.num_of_DMF],                                       'Total washes':[self.total_wash],                                       'Last  solvent in pores':[self.last_defects],                                       'Activation T, $^o$C':[self.activation_T],                                       'Activation time, h':[self.activation_time],                                       'Time of synthesis, h':[self.synt_time],                                       'Zr source':[self.Zr_source]})
        
      
        
    def predict(self):
       
        print('BET area:','%.1f' % self.model_area.predict(self.object_area)[0],'m*m/g')
        print('Particle size:','%.1f' % self.model_size.predict(self.object_size)[0],'nm')
        print('Number of linkers per Zr-cluster:','%.1f'% self.model_defects.predict(self.object_defects)[0])
        
        
        




S=Synthesis()
print('\n')
S.Zr_source=float(input('Zr source (enter 0 for ZrCl\u2084 and 1 for ZrOCl\u2082*8H\u2082O): '))

S.Zr_mass=float(input('Enter mass of Zr source (mg): '))
S.BDC_mass=float(input('Enter mass of H\u2082BDC (mg): '))
S.DMF_volume=float(input('Enter DMF volume (ml): '))
S.temp=float(input('Enter temperature: '))
S.synt_time=float(input('Enter time of synthesis (h): '))

print('\n')
print('Types of modulators: None, AcOH, HCOOH, HCl (37%), BzOH, TFA')

S.modulator=input('Enter modulator type: ')
S.mod_volume=float(input('Enter modulator volume (ml for liquid and g for BzOH): '))

ml_in_num=1


if (S.modulator=='None'):
    ml_in_num=0

if (S.modulator=='AcOH'):
    ml_in_num=1.05/60

if (S.modulator=='HCOOH'):
    ml_in_num=1.22/45

if (S.modulator=='HCl'):
    ml_in_num=1.19*0.37/(1+35.5)

if (S.modulator=='BzOH'):
    ml_in_num=1/122.12

if (S.modulator=='TFA'):
    ml_in_num=1.53/114.03




S.n_mod=ml_in_num*S.mod_volume


###S.n_mod=float(input('Enter number of modulator (mol): '))
S.water=float(input('Enter water volume (ml): '))
S.aging=float(input('Enter time of aging (h): '))

print('\n')
S.num_of_DMF=float(input('Enter number of DMF washes: ' ))
S.total_wash=float(input('Enter total number of washes: '))
print('Types of last solvents: MeOH, Ethanol,Acetone, Water, DMF, iPrOH, DMC, CHCl3')
S.last_solvent=input('Enter last solvent in pores: ')
S.activation_T=float(input('Enter activation temperature, degrees Celsius: '))
S.activation_time=float(input('Enter time of activation (h): '))

flag=1

if S.modulator in ['None', 'AcOH', 'HCOOH', 'HCl', 'HF', 'BzOH', 'TFA', 'NH3']:
	None
else:
	flag=2


if S.last_solvent in ['MeOH', 'Ethanol', 'Acetone', 'Water', 'DMF', 'iPrOH', 'DMC', 'CHCl3']:
	None
else:
	flag=3

if (flag==1):
	S.calculate()
	print('\n')
	print('Result: ')
	S.predict()

if (flag==2):
	print('Incorrect modulator')

if (flag==3):
	print('Incorrect last solvent')


