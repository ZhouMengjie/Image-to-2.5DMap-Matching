import os
import cv2
import random
import pandas as pd
from PIL import Image


if __name__ == '__main__':
    dataset_path = 'datasets'
    query_filename = 'unionsquare5kU'
    # sequences
    seq_filepath = os.path.join(dataset_path, 'csv', query_filename+'_sq.csv')
    sequences = (pd.read_csv(seq_filepath, sep=',', header=None)).values
    # pano meta
    meta_filepath = os.path.join(dataset_path, 'csv', query_filename+'_nb.csv')
    pano_meta = (pd.read_csv(meta_filepath, sep=',', header=None)).values
    # map meta
    set_filepath = os.path.join(dataset_path, 'csv', query_filename+ '_set.csv')
    map_meta = (pd.read_csv(set_filepath, sep=',', header=None)).values
           
    # Define the folder where to save the images
    output_folder = 'sequence_images'
    os.makedirs(output_folder, exist_ok=True)

    # Define the folder where to save the videos
    video_folder = 'sequence_videos'
    os.makedirs(video_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Random choose a sequence to visualize
    # ndx = random.randint(1, len(sequences))
    ndx = 4104
    seq_indices = sequences[ndx]
    videowrite_pano = cv2.VideoWriter(os.path.join(video_folder,str(ndx)+'pano.avi'),fourcc,1,(1664,832))
    videowrite_map = cv2.VideoWriter(os.path.join(video_folder,str(ndx)+'map.avi'),fourcc,1,(256,256))
    for i, idx in enumerate(seq_indices):
        panoid = pano_meta[idx][0]
        yaw = float(pano_meta[idx][3])
        city = pano_meta[idx][4]
        pano_pathname = os.path.join(dataset_path, 'jpegs_'+city+'_2019', panoid+'.jpg')
        global_idx = map_meta[idx][1]
        tile_pathname = os.path.join(dataset_path, 'tiles_'+city+'_2019', 'z18', str(global_idx).zfill(5) + '.png')
        # Read the images
        pano_image = Image.open(pano_pathname)
        tile_image = Image.open(tile_pathname)
        # Construct the new filenames
        pano_filename = f'{ndx}_pano_{i}_{yaw}.png'
        tile_filename = f'{ndx}_map_{i}_{yaw}.png'
        # Save the images to the output folder with corrected names
        pano_image.save(os.path.join(output_folder, pano_filename))
        tile_image.save(os.path.join(output_folder, tile_filename))
        # Close the images to free up resources
        pano_image.close()
        tile_image.close()

        # Make videos
        frame_pano = cv2.imread(pano_pathname)
        videowrite_pano.write(frame_pano)
        frame_tile = cv2.imread(tile_pathname)
        videowrite_map.write(frame_tile)
    
    videowrite_pano.release()
    videowrite_map.release()



