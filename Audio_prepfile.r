# ------------------------------------------------------
# extracts audio from video files and copies it to audio folder

setwd(video_dir)
for(filenameMP4 in list.files(video_dir)) {

	require(stringr)
	filename <- str_match(filenameMP4, "(.+)\\.MP4")[, 2]

  if(!(paste(filename, ".wav", sep = "") %in% list.files(audio_dir))) {
    system(command = paste("ffmpeg -i ", filename, ".MP4", " ", filename, ".wav", sep = ""))
    system(command = paste("mv ", video_dir, "/", filename, ".wav", " ", audio_dir, "/", filename, ".wav", sep = ""))
  }
}
