import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    # input related
    parser.add_argument("--exp_file", type=str, default="7")#
    parser.add_argument("--model", type=str, default='mosaicml/mpt-7b-instruct')#
    parser.add_argument("--baseline", type=str, default='BERTScore_baseline')  #
    parser.add_argument("--unk_token", type=str, default='...')#
    parser.add_argument("--stop_token", type=str, default='<|endoftext|>')#
    parser.add_argument("--unk_token_id", type=int, default=3346)#
    parser.add_argument("--forward_search_length", type=int, default=200)
    parser.add_argument("--backward_search_size", type=int, default=30)
    parser.add_argument("--backward_search_length", type=int, default=10)

    return parser
