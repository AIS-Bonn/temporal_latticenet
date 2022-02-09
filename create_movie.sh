# usage: call this script from the folder, where your images are. Then all the numbered images are added in the correct order.
ffmpeg -r 25 -framerate 25 -start_number 0 -i %d.png  -vcodec libx264 -pix_fmt yuv420p -crf 24 movie.mp4
