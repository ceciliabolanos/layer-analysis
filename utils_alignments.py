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


def match_frames_to_intervals(alignments, frame_length, stride, path, words, phones, words_counts, phones_counts):
    """
    Matches frame indices to the intervals from alignments for words and phones, correctly calculating
    start and end frames based on the provided alignments.
    """
    # Convert frame_length and stride from milliseconds to seconds for consistency
    frame_length_sec = frame_length / 1000.0
    stride_sec = stride / 1000.0

    for key in alignments:
        for xmin, xmax, text in alignments[key]:
            # Calculate the frame index for the start and end of the interval
            # Start frame is calculated by dividing xmin by stride_sec, because each new frame starts every stride_sec seconds
            start_frame = int(xmin / stride_sec)
            # End frame calculation considers the entire duration of the interval minus the frame length because
            # the last frame that fully fits within xmax should also be included.
            # Adjusting by frame_length_sec ensures we account for the case where xmax is exactly at the edge of a frame.
            end_frame = int((xmax - frame_length_sec) / stride_sec) if (xmax - frame_length_sec) > 0 else 0
            if key == 'phones':
                if text not in phones:
                   phones[text] = []
                   phones_counts[text] = 0
                phones[text].append((start_frame, end_frame, path))
                phones_counts[text] += 1
            if key == 'words':
                if text not in words:
                   words[text] = []
                   words_counts[text] = 0
                words[text].append((start_frame, end_frame, path))
                words_counts[text] += 1
    return words, phones

def remove_first_directory(path):
    parts = path.split(os.path.sep)  # Split the path into parts
    if len(parts) > 1:  # Check if there are enough parts to remove one
        return os.path.join(*parts[2:])  # Join the parts back together, skipping the first one
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


def match_words_to_frames(alignments, frame_length, stride, line_content):
    """
    Matches frame indices to word from alignments, correctly calculating
    start and end frames based on the provided alignments.
    """
    # Convert frame_length and stride from milliseconds to seconds for consistency
    frame_length_sec = frame_length / 1000.0
    stride_sec = stride / 1000.0
    words = {'transcript': line_content}
    for key in alignments:
        for xmin, xmax, text in alignments[key]:
            # Calculate the frame index for the start and end of the interval
            # Start frame is calculated by dividing xmin by stride_sec, because each new frame starts every stride_sec seconds
            start_frame = int(xmin / stride_sec)
            # End frame calculation considers the entire duration of the interval minus the frame length because
            # the last frame that fully fits within xmax should also be included.
            # Adjusting by frame_length_sec ensures we account for the case where xmax is exactly at the edge of a frame.
            end_frame = int((xmax - frame_length_sec) / stride_sec) if (xmax - frame_length_sec) > 0 else 0
            if key == 'words':
                if text not in words:
                   words[text] = []
                words[text].append((start_frame, end_frame))
    return words