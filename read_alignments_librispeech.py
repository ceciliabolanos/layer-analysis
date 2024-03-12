import os
import json
import argparse
from utils_alignments import parse_textgrid, match_frames_to_intervals, remove_first_directory

def main():
    
    parser = argparse.ArgumentParser(description='')

   # parser.add_argument('--model', type=str, required=False, help='', default="wav2vec") 
    parser.add_argument('--frame_length', type=int, required=False, help='milliseconds', default=25) 
    parser.add_argument('--stride', type=int, required=False, help='milliseconds', default=20)
    parser.add_argument('--sample_rate', type=int, required=False, help='Hz', default=16000)  
    parser.add_argument('--alignments', type=str, required=False, help='Directory containing the alignment TextGrid files', default='./alignments')      
    args = parser.parse_args()


    frame_length = args.frame_length
    stride = args.stride
    sample_rate = args.sample_rate
    alignments_dir = args.alignments


    # Dictionary to hold the results
    words_dict = {}
    phones_dict = {}
    words_counts_dict = {}
    phones_counts_dict = {}

    # Walk through the directory, and process each TextGrid file
    for root, dirs, files in os.walk(alignments_dir):
        for file in files:
            if file.endswith(".TextGrid"):
                # Construct the path to the TextGrid file
                textgrid_path = os.path.join(root, file)
                # Parse the TextGrid file to get alignments
                alignments = parse_textgrid(textgrid_path)
                root1 = remove_first_directory(root)
                # Construct the final path with the modified root (root1) and replacing extension with '.flac'
                path = os.path.join(root1, os.path.splitext(file)[0] + '.flac')
                
                words, phones = match_frames_to_intervals(alignments, frame_length, stride, path, words_dict, phones_dict,
                                                           words_counts_dict, phones_counts_dict)


    # Saving the results to a JSON file
    with open('words.json', 'w') as json_file:
        json.dump(words_dict, json_file, ensure_ascii=False, indent=4)

    with open('phones.json', 'w') as json_file:
        json.dump(phones_dict, json_file, ensure_ascii=False, indent=4)

    with open('words_counts.json', 'w') as json_file:
        json.dump(words_counts_dict, json_file, ensure_ascii=False, indent=4)

    with open('phones_counts.json', 'w') as json_file:
        json.dump(phones_counts_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
