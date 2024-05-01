import os

def parse_textgrid(textgrid_path):
    """
    Parses a TextGrid file and returns alignments for words and phones.
    """
    alignments = {'words': [], 'phones': []}
    current_section = None
    
    with open(textgrid_path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'name = "words"' in line:
                current_section = 'words'
            elif 'name = "phones"' in line:
                current_section = 'phones'
            elif line.startswith('intervals ['):
                if current_section:
                    xmin, xmax, text = None, None, None
            elif line.startswith('xmin ='):
                xmin = float(line.split('=')[1].strip())
            elif line.startswith('xmax ='):
                xmax = float(line.split('=')[1].strip())
            elif line.startswith('text ='):
                text = line.split('=')[1].strip().strip('"')
                if text:  # Ignore empty intervals
                    alignments[current_section].append((xmin, xmax, text))
    
    return alignments

def remove_first_directory(path):
    parts = path.split(os.path.sep)  
    if len(parts) > 1:  
        return os.path.join(*parts[3:])  
    return path


def read_line_by_identifier(directory_path, identifier):
    # Find the .txt file in the given directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(directory_path, file_name)
            break
    else:
        return "Text file not found in the directory."

    # Read the specified line from the found text file
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith(identifier):
                return line.strip().replace(identifier, '').strip()


def match_words_to_frames(alignments, stride, line_content):
    """
    Matches frame indices to word from alignments, correctly calculating
    start and end frames based on the provided alignments.
    """
    # Convert frame_length and stride from milliseconds to seconds for consistency
    stride_sec = stride / 1000.0
    words = {'transcript_audio': line_content}
    for key in alignments:
        for xmin, xmax, text in alignments[key]:
            # Calculate the frame index for the start and end of the interval
            # Start frame is calculated by dividing xmin by stride_sec, because each new frame starts every stride_sec seconds
            start_frame = int(xmin / stride_sec)
            end_frame = int(xmax / stride_sec) if xmax > 0 else 0
            if key == 'words':
                if text not in words:
                   words[text] = [] 
                words[text].append((start_frame, end_frame))
    return words

