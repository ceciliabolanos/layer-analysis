import os
import json
import argparse
from utils_alignments import parse_textgrid, match_words_to_frames, remove_first_directory, read_line_by_identifier

def main():
    
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--frame_length', type=int, required=False, help='milliseconds', default=25) 
    parser.add_argument('--stride', type=int, required=False, help='milliseconds', default=1000/75)
    parser.add_argument('--sample_rate', type=int, required=False, help='Hz', default=16000)  
    parser.add_argument('--alignments', type=str, required=False, help='Directory containing the alignment TextGrid files', default='../datasets/alignments')      
    args = parser.parse_args()


    frame_length = args.frame_length
    stride = args.stride
    sample_rate = args.sample_rate
    alignments_dir = args.alignments

    audio_dict = {}
    
    # Process each Textgrid file
    for root, dirs, files in os.walk(alignments_dir):
        for file in files :
            if (file.endswith(".TextGrid")) & ('test-clean' not in root):
                textgrid_path = os.path.join(root, file)
                alignments = parse_textgrid(textgrid_path)
                root1 = remove_first_directory(root)
                path = os.path.splitext(file)[0]
                path_to_transcript = os.path.join('../datasets/librispeech-clean-24k', root1)
                line_content = read_line_by_identifier(path_to_transcript, path) # get the audio transcript 
                words = match_words_to_frames(alignments, frame_length, stride, line_content)
                audio_dict[os.path.join(path_to_transcript, path)] = words


    with open('alignments/audio_alignments_encodec.json', 'w') as json_file:
        json.dump(audio_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
