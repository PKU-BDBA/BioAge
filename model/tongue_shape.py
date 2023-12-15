import glob
import numpy as np
import math
import os
import csv

from scipy.integrate import simps
from scipy.signal import butter, filtfilt


def procrustes(a, b):

  #translate the points to be centred at 0, 0
  a1 = a - a.mean(axis=0)
  b1 = b - b.mean(axis=0)

  #scale the points to have unit variance.
  a1 /= np.sqrt((a1**2.0).sum(axis=1).mean())
  b1 /= np.sqrt((b1**2.0).sum(axis=1).mean())

  #find the optimum rotation angle
  num   = (b1[:,0]*a1[:,1] - b1[:,1]*a1[:,0]).sum()
  denom = (b1[:,0]*a1[:,0] + b1[:,1]*a1[:,1]).sum()
  theta = math.atan2(num, denom)

  #rotate the b points onto a
  r_matrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
  b2 = np.dot(r_matrix, b1.transpose()).transpose()

  #compute the error metric
  return math.sqrt(((a1 - b2)**2.0).sum())


def curvature_index(data):

  # compute signed curvature
  dx = np.gradient(data[:,0])
  dy = np.gradient(data[:,1])
  ddx = np.gradient(dx)
  ddy = np.gradient(dy)
  cur = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5;

  s = np.cumsum(np.sqrt(np.sum(np.diff(data,axis=0)**2,axis=1)))
  s = np.insert(s,0,0)

  b, a = butter(5,1./4.)
  n = len(data)
  r = cur[::-1]
  fcur = filtfilt(b, a, np.concatenate((r,cur,r)))
  fcur = fcur[n:-n]

  fcurA = np.abs(fcur)
  mci = simps(fcurA,s)

  return mci


def fourier_analysis(data):

  ta = np.arctan2(np.gradient(data[:,1]),np.gradient(data[:,0]))

  ntfm = np.fft.rfft(ta)

  rl = np.real(ntfm)
  im = np.imag(ntfm)
  mod = np.absolute(ntfm)

  return rl, im, mod


def doIt():

  #name for the data output file
  output_file = "shape_analysis_data_out.csv"
  #number of lines to skip for header information in csv files
  n_header_lines = 0

  #make list of csv files in the working directory
  file_list = glob.glob("*.csv")

  #remove output file from file list
  if output_file in file_list: file_list.remove(output_file)

  #extract ID and symbol info from filename by splitting at last underscore
  file_list = [(f, os.path.splitext(f)[0].rsplit("_",1)) for f in file_list]

  #find the unique IDs among all files
  ids = set(i[1][0] for i in file_list)

  print "Got data for", len(ids), "unique id(s).", ids

  #open csv file for data output and write header information
  with open(output_file, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "symbol", "repetition", "MCI", "procrustes", "real_1", "imag_1", "mod_1", "real_2", "imag_2", "mod_2", "real_3", "imag_3", "mod_3"])

    for current_id in ids:

      print "Processing data for", current_id

      #filter to get the files relevant to the current id
      current_files = [(i[0],i[1][1]) for i in file_list if i[1][0] == current_id]

      #filter again to find the resting shape file, which should be ID_rest.csv
      rest_file = [i[0] for i in current_files if i[1] == "rest"]

      #check that there is either 0 or 1 resting shape file
      if len(rest_file) == 0:
        print "No resting shape found for", current_id, ", Procrustes analysis not available"
        doProcrustes = False
      elif len(rest_file) == 1:
        doProcrustes = True
        rdata = np.genfromtxt(rest_file[0], delimiter=",", skip_header=n_header_lines)
        #there should only be one shape in the resting shape file
        if rdata.shape[1] != 2:
          raise IOError("There should be one and only one shape in the resting shape file")
        print "Found resting shape"
      else:
        assert False, "This can't happen"

      #loop over all the files for the current id
      for file_name, symbol in current_files:

        #skip the resting shape file
        if symbol == "rest":
          continue

        data = np.genfromtxt(file_name, delimiter=",", skip_header=n_header_lines)

        if data.shape[1]%2 != 0: raise IOError("Number of data columns not a multiple of 2 in "+str(file_name))
        num_reps = data.shape[1]/2
        print "Found", num_reps, "shapes for", symbol

        for rep in range(0, num_reps):

          j = 2*rep

          #check for NaNs
          if (np.isnan(np.sum(data[:,j:j+2]))):
            print "NaN in shape", r, "ignoring..."
            continue

          if doProcrustes:
            proc = procrustes(rdata, data[:,j:j+2])
          else:
            proc = 0

          mci = curvature_index(data[:,j:j+2])

          rl, im, mod = fourier_analysis(data[:,j:j+2]) 

          writer.writerow([current_id, symbol, rep, mci, proc, rl[1], im[1], mod[1], rl[2], im[2], mod[2], rl[3], im[3], mod[3]])

if  __name__ == '__main__':
  doIt()
