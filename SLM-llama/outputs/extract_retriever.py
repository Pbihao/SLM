import argparse
import torch



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("src_file",
                        type=str,
                        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")
    parser.add_argument(
        "output_file",
        type=str,
        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)")
    args = parser.parse_args()

    data = torch.load(args.src_file)

    keys = list(data.keys())
    for key in keys:
        if not "retriever" in key:
            data.pop(key)
    
    keys = list(data.keys())
    for key in keys:
        if "retriever.bert" in key:
            data.pop(key)
    
    torch.save(data, args.output_file)