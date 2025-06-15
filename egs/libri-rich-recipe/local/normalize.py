import re

# Open the input and output files
input_file_path = 'data/lang_bpe_500/mux-rich-transcript_words.txt'
output_file_path = 'data/lang_bpe_500/mux-common-transcript_words.txt'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate through each line in the input file
    for line in input_file:
        # Remove punctuation (except apostrophes) and convert to lowercase
        cleaned_line = re.sub(r'(?<=[^\w\s])\'|\'(?=[^\w\s])|[^\w\s\']', '', line).lower()
        
        # Write the cleaned line to the output file
        output_file.write(cleaned_line)

print("File cleaned and saved to", output_file_path)
