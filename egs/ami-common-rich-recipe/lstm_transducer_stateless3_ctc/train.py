import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import AsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse import CutSet
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from lstm import RNN
from model import Transducer
from optim import Eden, Eve
from scaling import ScaledLinear
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    display_and_save_batch,
    setup_logger,
    str2bool,
)
from ami import AMI

LRSchedulerType = Union[
    torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler
]


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=int,
        default=12,
        help="Number of RNN encoder layers..",
    )

    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=512,
        help="Encoder output dimesion.",
    )

    parser.add_argument(
        "--rnn-hidden-size",
        type=int,
        default=1024,
        help="Hidden dim for LSTM layers.",
    )

    parser.add_argument(
        "--aux-layer-period",
        type=int,
        default=0,
        help="""Peroid of auxiliary layers used for randomly combined during training.
        If set to 0, will not use the random combiner (Default).
        You can set a positive integer to use the random combiner, e.g., 3.
        """,
    )

    parser.add_argument(
        "--grad-norm-threshold",
        type=float,
        default=25.0,
        help="""For each sequence element in batch, its gradient will be
        filtered out if the gradient norm is larger than
        `grad_norm_threshold * median`, where `median` is the median
        value of gradient norms of all elememts in batch.""",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=40,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="lstm_transducer_stateless/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--initial-lr",
        type=float,
        default=0.003,
        help="""The initial learning rate. This value should not need to be
        changed.""",
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate decreases.
        We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; "
        "2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network)"
        "part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.3,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=20,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=100,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warm_step for Noam optimizer.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # "valid_interval": 700,
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "dim_feedforward": 2048, #1024
            # parameters for decoder
            "decoder_dim": 512,
            # parameters for joiner
            "joiner_dim": 512,
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            # "model_warm_step": 700,
            "env_info": get_env_info(),
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = RNN(
        num_features=params.feature_dim,
        subsampling_factor=params.subsampling_factor,
        d_model=params.encoder_dim,
        rnn_hidden_size=params.rnn_hidden_size,
        grad_norm_threshold=params.grad_norm_threshold,
        dim_feedforward=params.dim_feedforward,
        num_encoder_layers=params.num_encoder_layers,
        aux_layer_period=params.aux_layer_period,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_ctc_model(params: AttributeDict) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=0.1),
        ScaledLinear(params.encoder_dim, params.vocab_size),
        nn.LogSoftmax(dim=-1),
    )
    

def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    ctc_model = get_ctc_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        ctc_model=ctc_model,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    warmup: float = 1.0,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute RNN-T loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Conformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = (
        model.device
        if isinstance(model, DDP)
        else next(model.parameters()).device
    )
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            warmup=warmup,
            reduction="none",
        )
        simple_loss_is_finite = torch.isfinite(simple_loss)
        pruned_loss_is_finite = torch.isfinite(pruned_loss)
        ctc_loss_is_finite = torch.isfinite(ctc_loss)
        is_finite = simple_loss_is_finite & pruned_loss_is_finite & ctc_loss_is_finite
        # is_finite = simple_loss_is_finite & pruned_loss_is_finite
        if not torch.all(is_finite):
            logging.info(
                "Not all losses are finite!\n"
                f"simple_loss: {simple_loss}\n"
                f"pruned_loss: {pruned_loss}\n"
                f"ctc_loss: {ctc_loss}"
            )
            display_and_save_batch(batch, params=params, sp=sp)
            simple_loss = simple_loss[simple_loss_is_finite]
            pruned_loss = pruned_loss[pruned_loss_is_finite]
            ctc_loss = ctc_loss[ctc_loss_is_finite]

            # If either all simple_loss or pruned_loss is inf or nan,
            # we stop the training process by raising an exception

            if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite) or torch.all(~ctc_loss_is_finite):
            # if torch.all(~simple_loss_is_finite) or torch.all(~pruned_loss_is_finite):
                # print(batch["supervisions"]["text"])
                raise ValueError(
                    "There are too many utterances in this batch "
                    "leading to inf or nan losses."
                )
                # print("There are too many utterances in this batch leading to inf or nan losses.")
                # # continue
                # pass

        simple_loss = simple_loss.sum()
        pruned_loss = pruned_loss.sum()
        ctc_loss = ctc_loss.sum()
        # after the main warmup step, we keep pruned_loss_scale small
        # for the same amount of time (model_warm_step), to avoid
        # overwhelming the simple_loss and causing it to diverge,
        # in case it had not fully learned the alignment yet.
        pruned_loss_scale = (
            0.0
            if warmup < 1.0
            else (0.1 if warmup > 1.0 and warmup < 2.0 else 1.0)
        )
        ctc_loss_scale = params.ctc_loss_scale if warmup >= 2.0 else 0
        loss = (
            params.simple_loss_scale * simple_loss
            + pruned_loss_scale * pruned_loss
            + ctc_loss_scale * ctc_loss
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # info["frames"] is an approximate number for two reasons:
        # (1) The acutal subsampling factor is ((lens - 1) // 2 - 1) // 2
        # (2) If some utterances in the batch lead to inf/nan loss, they
        #     are filtered out.
        info["frames"] = (
            (feature_lens // params.subsampling_factor).sum().item()
        )

    # `utt_duration` and `utt_pad_proportion` would be normalized by `utterances`  # noqa
    info["utterances"] = feature.size(0)
    # averaged input duration in frames over utterances
    info["utt_duration"] = feature_lens.sum().item()
    # averaged padding proportion over utterances
    info["utt_pad_proportion"] = (
        ((feature.size(1) - feature_lens) / feature.size(1)).sum().item()
    )

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()
    info["ctc_loss"] = ctc_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    warmup=(params.batch_idx_train / params.model_warm_step),
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()

            scheduler.step_batch(params.batch_idx_train)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 30:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if (
            batch_idx % params.log_interval == 0
            and not params.print_diagnostics
        ):
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(
                    tb_writer, "train/tot_", params.batch_idx_train
                )

        if (
            batch_idx > 0
            and batch_idx % params.valid_interval == 0
            and not params.print_diagnostics
        ):
            logging.info("Computing validation loss")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

def filter_short_and_long_utterances(cuts: CutSet, low: float = 1.0, high: float = 20.0) -> CutSet:
    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 17 seconds
        return low <= c.duration <= high

    cuts = cuts.filter(remove_short_and_long_utt)

    return cuts

def filter_utterances_on_max_chars_per_sec(cuts: CutSet, max_chars_per_sec: float) -> CutSet:
    def remove_utt(c: Cut):
        return (len(c.supervisions[0].text) / c.supervisions[0].duration) < max_chars_per_sec

    cuts = cuts.filter(remove_utt)

    return cuts

def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank])

    optimizer = Eve(model.parameters(), lr=params.initial_lr)

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    # # overwrite it
    # scheduler.base_lrs = [params.initial_lr for _ in scheduler.base_lrs]
    # print(scheduler.base_lrs)

    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)

    ami = AMI(manifest_dir=args.manifest_dir)
    ami_train_cuts = ami.train_cuts()
    print("len(ami_train_cuts.to_eager()")
    print(len(ami_train_cuts.to_eager()))
    ami_train_cuts = filter_short_and_long_utterances(ami_train_cuts, 1.0, 12.0)
    # ami_train_cuts = filter_utterances_on_max_chars_per_sec(ami_train_cuts, 18.0)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    asr_datamodule = AsrDataModule(args)
    # train_dl = asr_datamodule(
    #     ami_train_cuts, sampler_state_dict=sampler_state_dict
    # )
    train_dl = asr_datamodule.train_dataloaders(
        ami_train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = filter_short_and_long_utterances(ami.dev_cuts(), 1.0, 18.0)
    valid_dl = asr_datamodule.valid_dataloaders(valid_cuts)

    # if not params.print_diagnostics:
        # scan_pessimistic_batches_for_oom(
        #     model=model,
        #     train_dl=train_dl,
        #     optimizer=optimizer,
        #     sp=sp,
        #     params=params,
        #     warmup=0.0 if params.start_epoch == 1 else 1.0,
        # )

    scaler = GradScaler(enabled=params.use_fp16)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
    warmup: float,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    warmup=warmup,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            raise


def main():
    parser = get_parser()
    AsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
