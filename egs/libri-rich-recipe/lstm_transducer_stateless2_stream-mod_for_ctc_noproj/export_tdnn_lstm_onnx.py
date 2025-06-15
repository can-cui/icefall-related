#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import onnx
import torch
import torch.nn as nn
from icefall.lexicon import UniqLexicon
from tdnn_lstm import RNN
from decoder import Decoder
from scaling_converter import convert_scaled_to_non_scaled
from train_mux_lex_tdnn_lstm import add_model_arguments, get_params, get_transducer_model

from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import setup_logger

from onnxruntime.quantization import quantize_dynamic, QuantType

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for averaging.
        Note: Epoch counts from 0.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless3/exp",
        help="""It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_phone",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--g2p-tokens",
        type=str,
        default="data/lang_phone/phones.txt",
        help="G2P tokens.txt",
    )

    parser.add_argument(
        "--asr-tokens",
        type=str,
        default="data/lang_phone/tokens.txt",
        help="ASR tokens.txt",
    )

    add_model_arguments(parser)

    return parser


def add_meta_data(filename: str, meta_data: Dict[str, str]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    onnx.save(model, filename)


class OnnxEncoder(nn.Module):
    """A wrapper for LSTM and the ctc_model """

    def __init__(self, encoder: RNN, ctc_model: nn.Module):
        """
        Args:
          encoder:
            LSTM encoder.
          ctc_model:
            The CTC projection layer for encoder.
        """
        super().__init__()
        from tdnn7_lstm2_online import RNNOnline 
        online_encoder = RNNOnline(encoder.num_features, num_encoder_layers=encoder.num_encoder_layers)
        online_encoder.to("cpu")
        online_encoder.eval()
        #convert_scaled_to_non_scaled(online_encoder, inplace=True, is_onnx=True) # Do not do this. Do it after copying the parameters from the scaled encoder !!!
        j = 0
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_1[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_2[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_3[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_4[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_5[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_6[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        for i in [0,1,2]:
            online_encoder.encoder_embed.tdnn_7[i].load_state_dict(encoder.encoder_embed.conv[j].state_dict())
            j += 1
        online_encoder.encoder_embed.out.load_state_dict(encoder.encoder_embed.out.state_dict())
        online_encoder.encoder_embed.out_norm.load_state_dict(encoder.encoder_embed.out_norm.state_dict())
        online_encoder.encoder_embed.out_balancer.load_state_dict(encoder.encoder_embed.out_balancer.state_dict())
        online_encoder.encoder.load_state_dict(encoder.encoder.state_dict())
        convert_scaled_to_non_scaled(online_encoder, inplace=True, is_onnx=True) # Do this here else LSTMs are not correctly exported to the ONNX model
        self.online_encoder = online_encoder
        self.ctc_model = ctc_model

    def forward(
        self,
        x: torch.Tensor,
        f_cache: torch.Tensor,
        tdnn_cache: torch.Tensor,
        lstm_cntxts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Please see the help information of RNN.forward

        Args:
          x:
            A 3-D tensor of shape (N, T, C)
        """
        x, tdnn_out, lstm_cntxts = self.online_encoder(x, f_cache, tdnn_cache, lstm_cntxts)

        ctc_posts = self.ctc_model(x)

        return ctc_posts, tdnn_out, lstm_cntxts


def export_encoder_model_onnx(
    online_encoder_model: OnnxEncoder,
    onnx_filename: str,
    g2p_token_dict: Dict,
    asr_token_list: List,
    opset_version: int = 11,
) -> None:
    """Export the given encoder model to ONNX format.
    The exported model has two inputs:

        - x, a tensor of shape (N, T, C); dtype is torch.float32
        - x_lens, a tensor of shape (N,); dtype is torch.int64

    and it has two outputs:

        - encoder_out, a tensor of shape (N, T', joiner_dim)
        - encoder_out_lens, a tensor of shape (N,)

    Args:
      encoder_model:
        The input encoder model
      encoder_filename:
        The filename to save the exported ONNX model.
      opset_version:
        The opset version to use.
    """
     ### reorder output layer
    assert(len(asr_token_list)==online_encoder_model.ctc_model[1].weight.data.shape[0])
    if True:
        new_linear = torch.nn.Linear(in_features=online_encoder_model.ctc_model[1].weight.data.shape[1], out_features=online_encoder_model.ctc_model[1].weight.data.shape[0])
        for i in range(len(asr_token_list)):
            if asr_token_list[i] == "<eps>":
                assert(i==0)
                assert(g2p_token_dict["<eos>"]==i)
                new_linear.weight.data[i,:] = online_encoder_model.ctc_model[1].weight.data[i,:]
                new_linear.bias.data[i] = online_encoder_model.ctc_model[1].bias.data[i]
            elif asr_token_list[i] == "SIL":
                new_linear.weight.data[-1,:] = online_encoder_model.ctc_model[1].weight.data[i,:]
                new_linear.bias.data[-1] = online_encoder_model.ctc_model[1].bias.data[i]
            elif asr_token_list[i] == "SPN":
                new_linear.weight.data[-2,:] = online_encoder_model.ctc_model[1].weight.data[i,:]
                new_linear.bias.data[-2] = online_encoder_model.ctc_model[1].bias.data[i]
            else:
                j = g2p_token_dict[asr_token_list[i]]
                new_linear.weight.data[j,:] = online_encoder_model.ctc_model[1].weight.data[i,:]
                new_linear.bias.data[j] = online_encoder_model.ctc_model[1].bias.data[i]
        online_encoder_model.ctc_model[1].weight.data = new_linear.weight.data
        online_encoder_model.ctc_model[1].bias.data = new_linear.bias.data
    
    logging.info("Torch Scripting started")
    with torch.no_grad():
        scripted_encoder_model = torch.jit.script(online_encoder_model)
        FEATURE_DIM = 80
        dummy_feats = torch.randn(1, FEATURE_DIM, 15) # [N, C, T]
        dummy_feat_cache = torch.randn(1, FEATURE_DIM, 2)
        dummy_tdnn_cache = torch.randn(5, 1, 500, 2) # 7-2=5 tdnn layer outputs cached
        dummy_in_lstm_cntxts = torch.rand(online_encoder_model.online_encoder.num_encoder_layers, 2, 1, 1, 500) # 2 lstm, 2 states (cell+hidden) cached
        dummy_post_online, dummy_tdnn_out, dummy_lstm_cntxts_out = online_encoder_model(dummy_feats, dummy_feat_cache, dummy_tdnn_cache, dummy_in_lstm_cntxts)

### Note: This model does not handle first and last chunk correctly. But Overall WER is not affected.
        torch.onnx.export(scripted_encoder_model, 
            (dummy_feats, dummy_feat_cache, dummy_tdnn_cache, dummy_in_lstm_cntxts), 
            onnx_filename, 
            verbose=False, 
            opset_version=opset_version, 
            input_names=["feats", "feat_cache", "tdnn_cache", "in_lstm_cntxts"], 
            output_names=["posts", "tdnn_out", "out_lstm_cntxts"], 
            dynamic_axes={"feats": {2: "feats_T"}, "posts": {1: "posts_T"}},
            #example_outputs=(dummy_post_online, dummy_tdnn_out, dummy_lstm_cntxts_out) # deprecated in recent versions of pytorch
        )
        logging.info(f"Saved ONNX model to {onnx_filename}")
    
        quantize_dynamic(onnx_filename, onnx_filename.replace('.onnx', '.uint8-quant.onnx'),  weight_type=QuantType.QUInt8, extra_options = {'EnableSubgraph': True}, optimize_model=True)


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    setup_logger(f"{params.exp_dir}/log-export/log-export-onnx")

    logging.info(f"device: {device}")

    logging.info(f"Using phone lexion")
    pl = UniqLexicon(params.lang_dir)
    params.blank_id = 0
    params.vocab_size = max(pl.tokens) + 1
    params.grad_norm_threshold = 25.0
	
    g2p_token_dict = {}
    with open(params.g2p_tokens) as fp:
        for line in fp:
            token, id = line.strip().split()
            g2p_token_dict[token] = int(id)

    asr_token_list = []
    with open(params.asr_tokens) as fp:
        for line in fp:
            token, id = line.strip().split()
            if token.startswith('#'):   # skip disamb tokens towards the end
                break
            asr_token_list.append(token)
	
    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    model.to(device)

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
            )
        elif len(filenames) < params.avg:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {params.iter}, --avg {params.avg}"
            )
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )
    elif params.avg == 1:
        load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
    else:
        start = params.epoch - params.avg + 1
        filenames = []
        for i in range(start, params.epoch + 1):
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(
            average_checkpoints(filenames, device=device), strict=False
        )

    model.to("cpu")
    model.eval()

    print("model.parameters()", sum([p.numel() for p in model.parameters()]))
    print("model.encoder.parameters()", sum([p.numel() for p in model.encoder.parameters()]))
    ### Warning: Do not do the following for TDNN-LSTM. It causes problems with the ONNX export.Checkout the init func of class OnnxEncoder!!!
    #convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
    #print("non-scaled model.parameters()", sum([p.numel() for p in model.parameters()]))
    #print("non-scaled model.encoder.parameters()", sum([p.numel() for p in model.encoder.parameters()]))

    encoder = OnnxEncoder(
        encoder=model.encoder,
        ctc_model=model.ctc_model
    )

    encoder.to("cpu")
    encoder.eval()

    encoder_num_param = sum([p.numel() for p in encoder.parameters()])
    logging.info(f"encoder parameters: {encoder_num_param}")

    if params.iter > 0:
        suffix = f"iter-{params.iter}"
    else:
        suffix = f"epoch-{params.epoch}"

    suffix += f"-avg-{params.avg}"

    opset_version = 11#13

    logging.info("Exporting encoder")
    encoder_filename = params.exp_dir / f"tdnn7-lstm{params.num_encoder_layers}-{suffix}.onnx"
    export_encoder_model_onnx(
        encoder,
        str(encoder_filename),
        g2p_token_dict=g2p_token_dict,
        asr_token_list=asr_token_list,
        opset_version=opset_version,
    )
    logging.info(f"Exported encoder to {encoder_filename}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    main()
