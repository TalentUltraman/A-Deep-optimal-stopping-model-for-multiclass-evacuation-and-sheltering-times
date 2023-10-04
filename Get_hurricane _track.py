#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 13:07:36 2023

@author: hanwen
"""

import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd
from geopandas.tools import sjoin
from shapely import speedups
import shapely.geometry as geom
from shapely.geometry import LineString
from shapely import geometry,ops
import os
import glob
from scipy.stats import norm
import copy
speedups.enable
#%%
SC_county=gpd.read_file(r'/Users/hanwen/Desktop/GIS project/SC_countyb/south-carolina-county-boundaries.shp')
SC_county.plot()
#%%
#load wind probability data for 34knt level at 2018-10-07-00am(12 pm)
countyname=SC_county.iloc[:,4]
countyname=pd.concat([pd.Series(['Time']),countyname])
countyname=countyname.transpose()
SC_Wind_p34=pd.DataFrame(columns=range(47))
SC_Wind_p34.columns=countyname
SC_Wind_p50=pd.DataFrame(columns=range(47))
SC_Wind_p50.columns=countyname
SC_Wind_p64=pd.DataFrame(columns=range(47))
SC_Wind_p64.columns=countyname
#%%
def windpro(wind_track,SC_Wind_p,SC_county,Time):
    #Update SC_wind_p
    level=[0.025,0.075,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    nrow=len(SC_Wind_p)
    SC_Wind_p.loc[nrow,'Time']=Time
    for i in range(len(wind_track)):
        L=wind_track.iloc[i,1]
        Lp=SC_county.intersects(L)
        SC_Wind_p.iloc[nrow,Lp[Lp].index.values+1]=level[i] 
    #return SC_Wind_p
    return(SC_Wind_p)
#%%
def scan_folder34(parent,SC_Wind_p,SC_county):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith('34knt120hr_5km.shp'):
            # if it's a txt file, print its name (or do whatever you want)
            current_path = "".join((parent, "/", file_name))
            daytime=file_name[0:10]
            wind_p=gpd.read_file(current_path)
            SC_Wind_p=windpro(wind_p,SC_Wind_p,SC_county,daytime)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder34(current_path,SC_Wind_p,SC_county)
               
    return(SC_Wind_p)
def scan_folder50(parent,SC_Wind_p,SC_county):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith('50knt120hr_5km.shp'):
            # if it's a txt file, print its name (or do whatever you want)
            current_path = "".join((parent, "/", file_name))
            daytime=file_name[0:10]
            wind_p=gpd.read_file(current_path)
            SC_Wind_p=windpro(wind_p,SC_Wind_p,SC_county,daytime)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder50(current_path,SC_Wind_p,SC_county)
               
    return(SC_Wind_p)
def scan_folder64(parent,SC_Wind_p,SC_county):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith('64knt120hr_5km.shp'):
            # if it's a txt file, print its name (or do whatever you want)
            current_path = "".join((parent, "/", file_name))
            daytime=file_name[0:10]
            wind_p=gpd.read_file(current_path)
            SC_Wind_p=windpro(wind_p,SC_Wind_p,SC_county,daytime)
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recursively call this method
                scan_folder64(current_path,SC_Wind_p,SC_county)            
    return(SC_Wind_p)

def Format_Wind_p(SC_Wind_p):
    SC_Wind_p.Time=pd.to_datetime(SC_Wind_p.Time,format='%Y%m%d%H')
    SC_Wind_p=SC_Wind_p.sort_values(by='Time')
    SC_Wind_p=SC_Wind_p.reset_index(drop=True)
    SC_Wind_p=SC_Wind_p.fillna(0)
    return(SC_Wind_p)

#%%
SC_Wind_p34=scan_folder34(r'/Users/hanwen/Downloads/Florence',SC_Wind_p34,SC_county)
SC_Wind_p34=Format_Wind_p(SC_Wind_p34)

SC_Wind_p50=scan_folder50(r'/Users/hanwen/Downloads/Florence',SC_Wind_p50,SC_county)
SC_Wind_p50=Format_Wind_p(SC_Wind_p50)

SC_Wind_p64=scan_folder64(r'/Users/hanwen/Downloads/Florence',SC_Wind_p64,SC_county)
SC_Wind_p64=Format_Wind_p(SC_Wind_p64)
#%%
def finddis(p1,p2):
    if p2==0:
        p2=0.00000000000001
    p1=1-p1
    p2=1-p2
    z1=norm.ppf(p1)
    z2=norm.ppf(p2)
    u=(64*z1-34*z2)/(z1-z2)
    v=(34-u)/z1
    return([u,v])
#%%
def windpro_c(wind_track,SC_Wind_p,SC_county,Time):
    #Update SC_wind_p
    T=np.zeros(46)
    T=pd.Series(T)
    level=[0.025,0.075,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    nrow=len(SC_Wind_p)
    SC_Wind_p.loc[nrow,'Time']=Time
    for i in range(len(wind_track)):
        L=wind_track.iloc[i,1]
        Lp=SC_county.intersects(L)
        SC_Wind_p.iloc[nrow,Lp[Lp].index.values+1]=level[i] 
        T[SC_county.intersects(L)]=level[i] 
    #return SC_Wind_p
    return(SC_Wind_p)
#%%
SC_Wind_p34_c=SC_Wind_p34.copy()
SC_Wind_p50_c=SC_Wind_p50.copy()
SC_Wind_p64_c=SC_Wind_p64.copy()
u=np.zeros(25)
v=np.zeros(25)
SC_dummy_wind=SC_Wind_p34_c
from scipy.stats import norm
for j in range(1,len(SC_Wind_p34_c.columns)):
    for i in range(len(SC_Wind_p34_c)):
        p1=SC_Wind_p34_c.iloc[i,j]
        p2=SC_Wind_p64_c.iloc[i,j]
        u[i],v[i]=finddis(p1, p2)
        SC_dummy_wind.iloc[i,j]=u[i]
#%%
numeric_cols = SC_dummy_wind.select_dtypes(include=np.number)

# Set negative values in these columns to 0
SC_dummy_wind[numeric_cols.columns] = numeric_cols.clip(lower=0)

SC_dummy_wind=SC_dummy_wind.fillna(0)    
#%%
SC_dummy=SC_dummy_wind

SC_dummy.to_csv(r'/Users/hanwen/Desktop/GIS project/Code/Final1/Project1/dummy_wind_Florence.csv')