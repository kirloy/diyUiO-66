
# coding: utf-8

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

pd.options.mode.chained_assignment = None


# In[317]:




from sklearn.model_selection import train_test_split



# In[318]:




# In[319]:


from catboost import CatBoostRegressor


import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from ipywidgets import Accordion, IntSlider, Text
from ipywidgets import Dropdown

from IPython.display import display
from ipywidgets import Tab, IntSlider, Text
from ipywidgets import Button

from ipywidgets import IntProgress
from IPython.display import display
import time
# In[320]:




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
        
        
        


# In[324]:



# In[43]:


Zr_source = Dropdown(
    options=[('ZrCl\u2084', 1), ('ZrOCl\u2082*8H\u2082O', 2)],
    value=1,
)


# In[304]:


Zr_mass=widgets.BoundedFloatText(min=1,max=32000, step=0.1, description='Mass of Zr source, mg: ')
BDC_mass=widgets.BoundedFloatText(min=11,max=32000, step=0.1,description='Mass of H\u2082BDC, mg: ')
DMF_volume=widgets.BoundedFloatText(min=1,max=1000, step=0.1,description='DMF volume, ml: ')
temp=widgets.BoundedIntText(min=50,max=220, description='Temperature: ')
synt_time=widgets.BoundedFloatText(min=12,max=72,step=0.5, description='Synthesis time: ')


# In[305]:


Modulator = Dropdown(
    options=[('None',0),('AcOH', 1), ('HCOOH', 2),('HCl',3),('HF',4),('BzOH',5),('TFA',6),('NH3',7)],
    value=0,
)


l_s= Dropdown(
    options=[('MeOH', 1), ('Ethanol', 2),('Acetone',3),('Water',4),('DMF',5),('iPrOH',6),('DMC',7),('CHCl3',8)],
    value=3,
)

# In[306]:


Mod_volume=widgets.BoundedFloatText(min=0,max=500, step=0.1, description='Modulator volume, ml: ')


# In[307]:


Mod_mol=widgets.BoundedFloatText(min=0,max=500, step=0.1, description='Number of modulator, mol: ')


# In[308]:


Water=widgets.BoundedFloatText(min=0,max=500, step=0.1, description='Water volume, ml: ')


# In[309]:


Aging=widgets.BoundedIntText(min=0, max=72, description='Aging, h: ')


# In[310]:


Num_of_DMF=widgets.BoundedIntText(min=0, max=10, description='Number of DMF washes: ')
Total_washes=widgets.BoundedIntText(min=0, max=10, description='Total number of washes: ')
Activation_T=widgets.BoundedIntText(min=0, max=300, description='Activation temperature: ')
Activation_time=widgets.BoundedIntText(min=0, max=72, description='Activation time, h: ')


# In[311]:


accordion1 = Accordion(children=[Zr_source,Zr_mass,BDC_mass,DMF_volume,temp,synt_time])

accordion1.set_title(0, 'Choose a source of zirconium:')


accordion1.set_title(1, 'Enter mass of Zr source:')
accordion1.set_title(2, 'Enter mass of H\u2082BDC:')
accordion1.set_title(3, 'Enter DMF volume:')
accordion1.set_title(4, 'Enter temperature')
accordion1.set_title(5, 'Enter time of synthesis:')




# In[312]:


accordion2 = Accordion(children=[Modulator, Mod_volume,Mod_mol])

accordion2.set_title(0, 'Choose a modulator:')
accordion2.set_title(1, 'Enter volume of modulator, ml:')
accordion2.set_title(2, 'Enter number of modulator, mol:')



# In[313]:


accordion3=Accordion(children=[Water,Aging])
accordion3.set_title(0, 'Enter volume of water:')
accordion3.set_title(1,'Enter time of aging:')



# In[314]:


# In[315]:


accordion4=Accordion(children=[Num_of_DMF,Total_washes,l_s,Activation_T,Activation_time])

accordion4.set_title(0,'Enter number of DMF washes:')
accordion4.set_title(1,'Enter total number of washes:')
accordion4.set_title(2,'Choose last solvent in pores:')
accordion4.set_title(3,'Enter activation temperature:')
accordion4.set_title(4,'Enter time of activation:')



# In[316]:


def diyUIO():
    
    
  
    
    prgBar = IntProgress(min = 0, max = 50,description='Training models...')
    display(prgBar)

    while prgBar.value < prgBar.max:
        prgBar.value = prgBar.value + 1 
        time.sleep(0.01)
        
        


    tab = Tab()
    tab.children = [accordion1, accordion2,accordion3,accordion4]
    tab.set_title(0, 'General conditions')
    tab.set_title(1, 'Modulator')
    tab.set_title(2,'Additional conditions')
    tab.set_title(3, 'Washes/activation')

     

    button1 = Button(description="Predict!", 
                        button_style='success'
                        )
    
    def on_button_clicked(b): 
       
        S=Synthesis()
        
        if Zr_source.value==1:
            S.Zr_source=False
        if Zr_source.value==2:
            S.Zr_source=True

    
        S.Zr_mass=Zr_mass.value
        S.BDC_mass=BDC_mass.value
        S.DMF_volume=DMF_volume.value
        S.temp=temp.value
        S.synt_time=synt_time.value
        
        S.modulator=Modulator.value

        if S.modulator==0:
            S.modulator='None'
        if S.modulator==1:
            S.modulator='AcOH'
        if S.modulator==2:
            S.modulator='HCOOH'
        if S.modulator==3:
            S.modulator='HCl'
        if S.modulator==4:
            S.modulator='HF'
        if S.modulator==5:
            S.modulator='BzOH'
        if S.modulator==6:
            S.modulator='TFA'
        if S.modulator==7:
            S.modulator='NH3'
        
        
        S.mod_volume=Mod_volume.value
        S.n_mod=Mod_mol.value
        
        
        S.water=Water.value
        S.aging=Aging.value
        
        S.num_of_DMF=Num_of_DMF.value
        S.total_wash=Total_washes.value

        if l_s.value==1:
            S.last_solvent='MeOH'
            
        if l_s.value==2:
            S.last_solvent='Ethanol'
            
        if l_s.value==3:
            
            S.last_solvent='Acetone'
    
        if l_s.value==4:
            S.last_solvent='Water'
        if l_s.value==5:
            S.last_solvent='DMF'
        if l_s.value==6:
            S.last_solvent='iPrOH'
        if l_s.value==7:
            S.last_solvent='DMC'
        if l_s.value==8:
            S.last_solvent='CHCl3'
            

        S.activation_T=Activation_T.value
        S.activation_time=Activation_time.value

        S.calculate()

        S.predict()
        
    button1.on_click(on_button_clicked)
    
    display(tab)
    
    display(button1)





