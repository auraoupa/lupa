#!/usr/bin/env python
#
"""
This script filters a 2D field of data with Lanczos windowing method and writes the filtered signal in a new file that has the same structure than the input file.
It is possible to filter one or multiple files (same variable) and maps of total and filtered signal can also be produced.
External module needed : 
  - WavenumberSpectrum : https://github.com/lesommer/codes/blob/master/WavenumberSpectrum.py
  - oocgcm filtering module : https://github.com/lesommer/oocgcm
"""

## path for mdules

import sys
sys.path.insert(0,"/home/albert/lib/python")

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

## read the data

def read(filename,varname,level=None,time=0,**kwargs):
   """Return navlon,navlat,data.
   """
   navlon = xr.open_dataset(filename)['nav_lon']
   navlat = xr.open_dataset(filename)['nav_lat']
   if level is None:
     data = xr.open_dataset(filename)[varname]
   else:
     data = xr.open_dataset(filename)[varname][time,level,:,:]
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

## plot total, large scale and fine scale signals

def plot(filein,navlon,navlat,data,signal_LS,signal_SS,nwin,fcut,plotname=None,varname='socurloverf',**kwargs):
    """Create a png file with the plot.
    """
    if plotname is None:
      plotname=filein[0:len(filein)-3]+'_filt-n'+str(nwin)+'-f'+str(fcut)+'_plots.png'
    print('plotname is '+plotname)
    cont=np.isnan(data)
    plt.figure(figsize=(15,33))
    ax = plt.subplot(311)
    ax.autoscale(tight=True)
    pcolor = ax.pcolormesh(navlon,navlat,
      ma.masked_invalid(data),cmap=div_cmap,vmin=-1,vmax=1,alpha=1)
    ax.tick_params(labelsize=25)
    ax.contour(navlon,navlat,cont,alpha=0.5,linewidth=0.000001,antialiased=True)
    cbar = plt.colorbar(pcolor,orientation='horizontal',pad=0.1)
    cbar.ax.tick_params(labelsize=35)
    ax.set_xlabel('Longitude (in degree)',fontsize=20)
    ax.set_ylabel('Latitude (in degree)',fontsize=20)
    cbar.ax.tick_params(labelsize=25)
    plt.title('Total signal '+varname,fontsize=25)
    cont=np.isnan(signal_LS)
    ax = plt.subplot(312)
    ax.autoscale(tight=True)
    pcolor = ax.pcolormesh(navlon,navlat,
      ma.masked_invalid(signal_LS),cmap=div_cmap,vmin=-1,vmax=1,alpha=1)
    ax.tick_params(labelsize=25)
    ax.contour(navlon,navlat,cont,alpha=0.5,linewidth=0.000001,antialiased=True)
    cbar = plt.colorbar(pcolor,orientation='horizontal',pad=0.1)
    cbar.ax.tick_params(labelsize=35)
    ax.set_xlabel('Longitude (in degree)',fontsize=20)
    ax.set_ylabel('Latitude (in degree)',fontsize=20)
    cbar.ax.tick_params(labelsize=25)
    plt.title('Large scale '+varname,fontsize=25)
    signal_SS=data-signal_LS
    cont=np.isnan(signal_SS)
    ax = plt.subplot(313)
    ax.autoscale(tight=True)
    pcolor = ax.pcolormesh(navlon,navlat,
      ma.masked_invalid(signal_SS),cmap=div_cmap,vmin=-1,vmax=1,alpha=1)
    ax.contour(navlon,navlat,cont,alpha=0.5,linewidth=0.000001,antialiased=True)
    cbar = plt.colorbar(pcolor,orientation='horizontal',pad=0.1)
    cbar.ax.tick_params(labelsize=35)
    ax.set_xlabel('Longitude (in degree)',fontsize=20)
    ax.set_ylabel('Latitude (in degree)',fontsize=20)
    cbar.ax.tick_params(labelsize=25)
    plt.title('Small scale '+varname,fontsize=25)
    plt.savefig(plotname)

## write output file
def write(filein,signal_SS,nwin,fcut,varname,**kwargs):
    """Write the output file with the same structure than the input file
       In the variable varname_filt, the fine scale signal is written
    """
    
    outname=filein[0:len(filein)-3]+'_filt-n'+str(nwin)+'-f'+str(fcut)+'.nc'
    print('output file is '+outname)
    dstin=Dataset(filein,'r')
    dstout=Dataset(outname,'w')

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
    dataout=dstout.createVariable(varname+'_filt',datain.datatype,datain.dimensions)
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
    usage = "usage: %prog [options] file*.nc n[size of Lanczos window] f[cut-off frequency]"
    parser = OptionParser(usage=usage)
    parser.add_option("-v", "--varname", dest="varname",\
                    help="name of the variable to process", default='socurloverf')
    parser.add_option("-l","--level", dest="level",type="int",\
                      help="level of the variable to process, default value for 2D fields", default=None)
    parser.add_option("-s", action="store_true", dest="showmap", default=False,\
                    help="plot 2D maps of total, large and fine scale signal")
    parser.add_option("-p", "--plotname", dest="plotname",\
                    help="name of the plot (if one file)", default=None)
    return parser


def main():
    parser = script_parser()
    (options, args) = parser.parse_args()
    if len(args) < 3: # print the help message if number of args is not 3.
        parser.print_help()
        sys.exit()
    optdic = vars(options)
    ## One file in input
    if len(args) == 3:
      print('One file to process')
      filein = args[0]
      nwin=int(args[1])
      fcut=float(args[2])
      navlon,navlat,data = read(filein,**optdic)
      signal_LS,signal_SS = filt(data,nwin,fcut)
      if optdic['showmap'] is True:
        plot(filein,navlon,navlat,data.squeeze(),signal_LS.squeeze(),signal_SS.squeeze(),nwin,fcut,**optdic)
      write(filein,signal_SS.squeeze(),nwin,fcut,**optdic)
    ## Multiple files in input => loop
    if len(args) > 3:
      print('Multiple files to process')
      nwin=int(args[len(args)-2])
      fcut=float(args[len(args)-1])
      for t in np.arange(0,len(args)-2):
        filein = args[t]
        navlon,navlat,data = read(filein,**optdic)
        signal_LS,signal_SS = filt(data,nwin,fcut)
        if optdic['showmap'] is True:
          plot(filein,navlon,navlat,data.squeeze(),signal_LS.squeeze(),signal_SS.squeeze(),nwin,fcut,**optdic)
        write(filein,signal_SS.squeeze(),nwin,fcut,**optdic)

if __name__ == '__main__':
    sys.exit(main() or 0)
