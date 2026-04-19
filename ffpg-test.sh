export SRC=/Users/anpoltavets/anna-apps/denoising/test-src-videos
ffmpeg -i $SRC/5_2.mp4 -vf hqdn3d=4:3:6:4 $SRC/5_2_output_ffmpeg.mp4