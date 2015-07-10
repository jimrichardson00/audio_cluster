# ------------------------------------------------------
# extract data and perform PCA

setwd(audio_dir)

audio_files <- sort(list.files(audio_dir, pattern = "\\.wav$"))
audio_files

# pulls out spectrogram at 10 second mark
require(parallel)
require(tuneR)
AudioFeatures_l <- mclapply(X = audio_files, FUN = function(audio_file) AudioFeatures(audio_file)
	, mc.cores = n_cores
	, mc.silent = FALSE
	, mc.preschedule = FALSE
	)

# audio data
AudioFeatures_m <- matrix(unlist(AudioFeatures_l), nrow = length(audio_files), byrow = TRUE)
AudioFeatures_m

require(stats)
prcomp <- prcomp(AudioFeatures_m, scale. = TRUE)

components_ <- prcomp$rotation
prx <- prcomp$x

# update ccipca list with new values
write.table(AudioFeatures_m, file = paste("AudioFeatures_m", year, ".txt", sep = ""))  
write.table(audio_files, file = paste("audio_files", year, ".txt", sep = ""))  
write.table(components_, file = paste("components_", year, ".txt", sep = ""))  
write.table(prx, file = paste("prx", year, ".txt", sep = ""))
