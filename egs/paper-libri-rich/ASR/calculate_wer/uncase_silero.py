import re
import torch
import string
from tqdm import tqdm


from rpunct import RestorePuncts

rpunct = RestorePuncts(use_cuda=False)


def restore_case(output_filepath, new_output, apply_te):
    # Open the file and read its contents
    with open(output_filepath, "r") as file:
        lines = file.readlines()
    # Initialize an empty dictionary
    with open(new_output, "w") as out_file:
        # Process the lines and construct the dictionary
        for line in tqdm(lines):
            sentence_id, sentence = line.split(",", 1)
            cleaned_text = re.sub(
                r"(?<=[^\w\s])\'|\'(?=[^\w\s])|[^\w\s\']", "", sentence
            ).lower()
            # print(cleaned_text)
            # if cleaned_text=="'":
            #     mapped_sentence=""
            try:
                lm_out = apply_te(cleaned_text)
            except:
                lm_out = ""
            lm_out_noPunc = re.sub(
                r"(?<=[^\w\s])\'|\'(?=[^\w\s])|[^\w\s\']", "", lm_out
            )
            original_tokens = sentence.split()
            cleaned_tokens = lm_out_noPunc.split()
            # Map punctuation from the original text to the cleaned text
            mapped_tokens = []

            for orig_token, cleaned_token in zip(original_tokens, cleaned_tokens):
                if orig_token[-1] in string.punctuation:
                    cleaned_token += orig_token[-1]
                mapped_tokens.append(cleaned_token)  # Use cleaned token

            # Join mapped tokens back into a sentence
            mapped_sentence = " ".join(mapped_tokens)
            out_file.write(sentence_id + "," + mapped_sentence + "\n")


def main_uncase():
    model_silero_te, example_texts, languages, punct, apply_te = torch.hub.load(
        repo_or_dir="snakers4/silero-models", model="silero_te"
    )
    file_list = [
        "greedy_com_unc_rich/modeluncase.txt",
        "greedy_com_unc_rich/modeluncaseuncase.txt",
    ]
    # file_list =
    for file in file_list:
        print("processing ", file)
        output = file.split(".")[0] + "_case.txt"
        restore_case(file, output, apply_te)


def restore_case_punc(output_filepath, new_output):
    # Open the file and read its contents
    with open(output_filepath, "r") as file:
        lines = file.readlines()
    # Initialize an empty dictionary
    with open(new_output, "w") as out_file:
        # Process the lines and construct the dictionary
        for line in tqdm(lines):
            sentence_id, sentence = line.split(",", 1)
            try:
                lm_out = rpunct.punctuate(sentence.lower(), lang="en")
                # lm_out = apply_te(sentence.lower())
            except:
                lm_out = ""
            out_file.write(sentence_id + "," + lm_out + "\n")


def main_uncase_punc():
    # model_silero_te, example_texts, languages, punct, apply_te = torch.hub.load(
    #     repo_or_dir="snakers4/silero-models", model="silero_te"
    # )
    # file_list = ["common/clean.txt", "common/other.txt", "common/ihm.txt"]
    file_list = [
        "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/paper-libri-rich/ASR/calculate_wer/common/cv-40.txt"
    ]
    for file in file_list:
        print("processing ", file)
        # output = file.split(".")[0] + "_casePunc.txt"
        output = "/srv/storage/talc2@talc-data2.nancy.grid5000.fr/multispeech/calcul/users/ccui/icefall/egs/paper-libri-rich/ASR/calculate_wer/common/cv_casePunc.txt"
        restore_case_punc(file, output)


if __name__ == "__main__":
    main_uncase_punc()
