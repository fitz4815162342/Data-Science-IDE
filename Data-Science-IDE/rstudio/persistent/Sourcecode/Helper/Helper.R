options(warn=-1)

#try(
#  expr = {
#    rstudioapi::removeTheme("matrix_glow")    
#  }
#)
#try(
#  expr = {
#    rstudioapi::addTheme("https://raw.githubusercontent.com/AlessioMR/matrix_glow/master/matrix_glow.rstheme", apply = TRUE ) 
#  }
#)

cat("\014")  
print(R.version.string)
list.of.packages <- c("reticulate", "future", "stringr", "docstring", "imager")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(imager)

library(stringr)
library(future)
plan(multicore)

library(reticulate)
use_virtualenv("/home/rstudio/venv")
characteristics <- import_from_path("Helper.AudioCharacteristics", 
                                    path = file.path(file.path(getwd(), "Helper"), "AudioCharacteristics.py"), 
                                    convert = TRUE, delay_load = FALSE)
helper <- import_from_path("Helper.Helper", 
                           path = file.path(file.path(getwd(), "Helper"), "Helper.py"), 
                           convert = TRUE, delay_load = FALSE)
np <- import("numpy")
plt <- import("matplotlib.pyplot")
librosa <- import("librosa")
librosa_display <- import("librosa.display")
pathlib <- import("pathlib")
cv2 <- import("cv2")

samples <- list()


get_filename <- function(path, verbose=FALSE){
  
  #' Extract the file name out of a file path
  #' @param path string, the input file path
  #' @param verbose bool, sets how much logging output we see
  
  file_name <- pathlib$Path(path)$stem
  if (verbose){
    print(paste("[get_filename] ", file_name))    
  }
  return(file_name)
}


get_file_extension <- function(path, verbose){
  
  #' Extract the file type out of a file path
  #' @param path string, the input file path
  #' @param verbose bool, sets how much logging output we see
  
  extension <- substr(path, nchar(path)-3+1, nchar(path))
  if (verbose){
    print(paste("[get_file_extension] ", extension))    
  }
  return(extension)
}


list_files <- function(path, verbose){
  
  #' Populate a list with all file paths inside a given directory
  #' @param path string, the input path to the directory
  #' @param verbose bool, sets how much logging output we see
  
  files <- list.files(path=path, pattern=NULL, all.files=FALSE, full.names=TRUE)
  if (verbose){
    print(paste("[list_files] ", files))    
  }
  return(files)
}


list_directories <- function(path, verbose=FALSE){
  
  #' Populate a list with all directories inside a given directory, not recursive
  #' @param path string, the input path to the directory
  #' @param verbose bool, sets how much logging output we see
  
  dirs <- list.dirs(path=path, full.names=TRUE, recursive=TRUE)
  if (verbose){
    print(paste("[list_directories] ", dirs))    
  }
  return(dirs)
}


import_job <- function(data_dir, verbose){
  
  #' Populate a list of type AudioCharacteristics for all sound files inside a given directory
  #' @param data_dir string, the input path to the directory
  #' @param verbose bool, sets how much logging output we see
  
  sub_dir <- list_files(data_dir, verbose=FALSE)
  imported_samples <- list()
  for (file in sub_dir){
    allowed_extensions <- c("wav", "mp3", "m4a")
    if (str_detect(allowed_extensions, get_file_extension(file, FALSE))){
      try(
        expr = {
          sample_name <- get_filename(file, FALSE)
          sampling_rate <- helper$librosa$get_samplerate(file)
          audio_import <- librosa$load(path=file, sr=sampling_rate, duration=duration, offset=offset)
          time_series <- audio_import[[1]]
          sampling_rate <- audio_import[[2]]
          imported_samples <- append(imported_samples, characteristics$AudioCharacteristics(sample_name, 
                                                                                            time_series, 
                                                                                            sampling_rate))
        }
      )
    }  
  }  
  samples <- grow_data(samples, imported_samples, verbose)
  return(samples)
}


grow_data <- function(samples, subsample, verbose){
  
  #' Append new imports from the import_job to global samples list
  #' For multi processing this function could be used as data callback
  #' @param samples list, imported audio in total
  #' @param subsample list, imported audio files for specific directory
  #' @param verbose bool, sets how much logging output we see
  
  for (subsubsample in subsample){
    samples <- append(samples, subsubsample)
  }
  if (verbose){
    print(paste("[grow_data] total length of imported samples: ", length(samples)))    
  }
  return(samples)
}


show_voice_components <- function(timeseries, sampling_rate, sample_name) {
  
  #' Populate a list of type AudioCharacteristics for all sound files inside a given directory
  #' @param timeseries numpy array, input data of the sound signal
  #' @param sampling_rate int, careful with sampling rates, hashtag #nyquist 
  #' @param sample_name string, the image should get a title
  
  sampling_rate <- as.integer(sampling_rate)
  plt$figure(figsize=tuple(10, 3))
  plt$title(paste("Wave plots: ", sample_name))
  wave_decomposition <- librosa$effects$hpss(timeseries)
  y_harmonic = wave_decomposition[[1]]
  y_percussive = wave_decomposition[[2]]
  librosa_display$waveplot(y_harmonic, sr=sampling_rate, alpha=0.25)
  librosa_display$waveplot(y_percussive, sr=sampling_rate, color='red', alpha=0.5)
  plt$show()
}


show_mel_pitches <- function(timeseries, sampling_rate, sample_name) {
  
  #' Populate a list of type AudioCharacteristics for all sound files inside a given directory
  #' @param timeseries numpy array, input data of the sound signal
  #' @param sampling_rate int, careful with sampling rates, hashtag #nyquist 
  #' @param sample_name string, the image should get a title
  
  sampling_rate <- as.integer(sampling_rate)
  plt$figure(figsize=tuple(10, 3))
  plt$title(paste("Mel pitches: ", sample_name))
  spec = librosa$feature$melspectrogram(y=timeseries, sr=sampling_rate)
  db_spec = librosa$power_to_db(spec, ref=np$max)
  librosa_display$specshow(db_spec, y_axis='mel', x_axis='s', sr=sampling_rate)
  plt$colorbar()
  plt$show()
}