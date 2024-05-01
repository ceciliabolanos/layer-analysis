import os
import json
import argparse
from utils_alignments import parse_textgrid, match_words_to_frames, remove_first_directory, read_line_by_identifier
import numpy as np

def main():
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--sample_rate', type=int, required=False, help='kHz', default=24) 
    parser.add_argument('--stride', type=int, required=False, help='milliseconds', default=1000/75)
    parser.add_argument('--alignments', type=str, required=False, help='Directory containing the alignment TextGrid files', default='../datasets/alignments')      
    args = parser.parse_args()

    stride = args.stride
    alignments_dir = args.alignments
    sample_rate = args.sample_rate
    audio_dict = {}
    
    if sample_rate == 24: 
        path_to_sample = '../datasets/librispeech-clean-24k'
    elif sample_rate == 16:
        path_to_sample = '../datasets/librispeech-clean-16k'    

    # Process each Textgrid file
    for root, dirs, files in os.walk(alignments_dir):
        for file in files :
            if (file.endswith(".TextGrid")) & ('test-clean' not in root):
                textgrid_path = os.path.join(root, file)
                alignments = parse_textgrid(textgrid_path)
                root1 = remove_first_directory(root)
                path = os.path.splitext(file)[0]
                path_to_transcript = os.path.join(path_to_sample, root1)
                line_content = read_line_by_identifier(path_to_transcript, path) # get the audio transcript 
                words = match_words_to_frames(alignments, stride, line_content)
                audio_dict[os.path.join(path_to_transcript, path)] = words


    with open(f'alignments/audio_alignments_{np.round(stride, 2)}_{sample_rate}.json', 'w') as json_file:
        json.dump(audio_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
