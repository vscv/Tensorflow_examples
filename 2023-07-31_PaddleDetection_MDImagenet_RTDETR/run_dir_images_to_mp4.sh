ls MD_test_534_out_bigCht/*.JPG | sort -V | xargs -I {} echo "file '{}'" > MD_test_534_out_bigCht_jpg_list.txt  
ffmpeg -r 20 -f concat -i jpg_list.txt -c:v libx264 -r 20 -pix_fmt yuv420p out.mp4


file name should be \(1\) not the (1) for shell path!!!
