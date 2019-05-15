#!/usr/bin/env python
#
"""
This script filters a 2D field of data with Lanczos windowing method and writes the filtered signal in a new file that has the same structure than the input file.
It is possible to filter one or multiple files (same variable) and maps of total and filtered signal can also be produced.
External module needed : 
  - GriddedData.py : https://github.com/lesommer/codes/blob/master/GriddedData.py
	- that needs basemap : conda install basemap 
  - oocgcm filtering module : https://github.com/lesommer/oocgcm
"""

## path for mdules

import sys
sys.path.insert(0,"/home/albert7a/lib/python")

## imports

import numpy as np
import xarray as xr
import GriddedData
#- Other modules
import numpy.ma as ma
from netCDF4 import Dataset

### quick plot
import matplotlib.pyplot as plt

### palette
import matplotlib.cm as mplcm
seq_cmap = mplcm.Blues
div_cmap = mplcm.seismic


### local/specific imports
import oocgcm
import oocgcm.filtering
import oocgcm.filtering.linearfilters as tf

from datetime import date
import time

## read the data

def read(dirname,filename,varname):
   """Return navlon,navlat,data.
   """
   filenamet=dirname+filename
   navlon = xr.open_dataset(filenamet)['nav_lon']
   navlat = xr.open_dataset(filenamet)['nav_lat']
   data = xr.open_dataset(filenamet)[varname]
   return navlon,navlat,data


## filter the data

def filt(data,nwin,fcut):
    """ Filter the data with Lanczos window of size nwin and fcut cut-off frequency
        Return signal_LS[Large scale] and signal_SS[Small scale]
    """
    win_box2D = data.win
    win_box2D.set(window_name='lanczos', n=[nwin, nwin], dims=['x', 'y'], fc=fcut)
    bw = win_box2D.boundary_weights(drop_dims=[])
    signal_LS = win_box2D.apply(weights=bw)
    signal_SS=data-signal_LS
    return signal_LS,signal_SS


## write output file
def write(dirin,filein,signal_SS,nwin,fcut,dirout,varname,suffix):
    """Write the output file with the same structure than the input file
       In the variable varname_filt, the fine scale signal is written
    """
    
    outname=filein[0:len(filein)-3]+'_filt-n'+str(nwin)+'-f'+str(fcut)+'_'+str(suffix)+'.nc'
    outnamet=dirout+outname
    print('output file is '+outnamet)
    fileint=dirin+filein
    dstin=Dataset(fileint,'r')
    dstout=Dataset(outnamet,'w')

    today=date.today()
    dstout.description = "Data filtered with Lanczos filter with window size of "+str(nwin)+" and cut-off frequency of "+str(fcut)+" obtained with Lanczos2DHighPassFilter.py script "+str(today.day)+"/"+str(today.month)+"/"+str(today.year)

    #Copy the structure of the input file
    for dname, the_dim in dstin.dimensions.iteritems():
      dstout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)

    for v_name, varin in dstin.variables.iteritems(): 
      if v_name == varname: 
        continue
      outVar = dstout.createVariable(v_name, varin.datatype, varin.dimensions)
      outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
      outVar[:] = varin[:]

    #Create output variable
    datain=dstin[varname]
    dataout=dstout.createVariable(varname+'_small_scales',datain.datatype,datain.dimensions)
    dataout.setncatts({k: datain.getncattr(k) for k in datain.ncattrs()})

    #Because time is a unlimited dimension we have to fill the file with an temporary array
    sh=signal_SS.shape
    xdim=sh[0]
    ydim=sh[1]  
    ztemp=np.zeros((1,xdim,ydim))
    ztemp[0,:,:]=signal_SS[:,:]
    dataout[0,:,:] = ztemp[0,:,:]

    dstout.close()    


## parser and main
def script_parser():
    """Customized parser.
    """
    from optparse import OptionParser
    usage = "usage: %prog dirin file*.nc varname n[size of Lanczos window] f[cut-off frequency] dirout"
    parser = OptionParser(usage=usage)
    return parser


def main():
    parser = script_parser()
    (options, args) = parser.parse_args()
    if len(args) < 5: # print the help message if number of args is not 3.
        parser.print_help()
        sys.exit()
    ## One file in input
    if len(args) == 6:
      print time.strftime('%d/%m/%y %H:%M',time.localtime())
      print('One file to process')
      dirin = args[0]
      filein = args[1]
      varname = args[2]
      nwin = int(args[3])
      fcut = float(args[4])
      dirout = args[5]
      navlon,navlat,data = read(dirin,filein,varname)
      signal_LS,signal_SS = filt(data.squeeze(),nwin,fcut)
      write(dirin,filein,signal_SS.squeeze(),nwin,fcut,dirout,varname,'small_scales')
      write(dirin,filein,signal_LS.squeeze(),nwin,fcut,dirout,varname,'large_scales')
      print time.strftime('%d/%m/%y %H:%M',time.localtime())
    ## Multiple files in input => loop
    if len(args) > 6:
      print('Multiple files to process')
      dirin=args[0]
      varname = args[len(args)-4]
      nwin=int(args[len(args)-3])
      fcut=float(args[len(args)-2])
      dirout=args[len(args)-1]
      for t in np.arange(1,len(args)-4):
        print time.strftime('%d/%m/%y %H:%M',time.localtime())
        filein = args[t]
        navlon,navlat,data = read(dirin,filein,varname)
        signal_LS,signal_SS = filt(data,nwin,fcut)
        write(dirin,filein,signal_SS.squeeze(),signal_LS.squeeze(),nwin,fcut,dirout,varname,'small_scales')
        write(dirin,filein,signal_SS.squeeze(),signal_LS.squeeze(),nwin,fcut,dirout,varname,'large_scales')
        print time.strftime('%d/%m/%y %H:%M',time.localtime())

if __name__ == '__main__':
    sys.exit(main() or 0)
