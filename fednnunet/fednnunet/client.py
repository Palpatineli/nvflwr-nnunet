from collections.abc import Mapping
from dataclasses import dataclass
import logging
import os
from io import BytesIO
from typing import Literal, cast

import flwr as fl
from flwr.client import Client, ClientApp
from flwr.common.typing import GetParametersIns
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import NonDetMultiThreadedAugmenter
import numpy as np
import torch
from flwr.common import Code, Context, EvaluateRes, FitRes, GetParametersRes, Parameters, Status

from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    DatasetFingerprintExtractor,
    extract_fingerprint_dataset,
    plan_experiments,
    preprocess,
)
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import (
    convert_dataset_name_to_id,
    maybe_convert_to_dataset_name,
)

from fednnunet.run_training import run_training, nnUNetTrainer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

@dataclass
class Args:
    device: Literal['cpu'] | Literal['cuda'] | Literal['mps']
    dataset_id: str
    configuration: Literal['3d_fullres'] | Literal['3d_lowres'] | Literal['2d']
    fold: int | Literal['all']
    trainer: str
    plan: str = "resEncM25"
    planner: str = "nnUNetPlanner"
    is_continue: bool = False
    gpu_memory_target: int | None = None
    num_process: int | None = None
    overwrite_target_spacing: tuple[float, float, float] | None = None

    def __post_init__(self):
        if self.overwrite_target_spacing is not None:
            self.overwrite_target_spacing = [float(x) for x in self.overwrite_target_spacing.split()]  # pyright: ignore[reportAttributeAccessIssue]


def get_device(args: Args) -> torch.device:
    if args.device == "cuda": # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return torch.device("cuda")
    elif args.device == "mps":
        return torch.device("mps")
    else:
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        return torch.device("cpu")

StateDict = Mapping[str, torch.Tensor | list[int] | list[float] | dict[str, int | float | bool | None]\
        | list[list[int]] | list[list[float]]]

def state_dict_to_bytes(state_dict: StateDict) -> bytes:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    bytes_io = BytesIO()
    torch.save(state_dict, bytes_io)
    return bytes_io.getvalue()


def state_dict_to_parameters(state_dict: StateDict) -> Parameters:
    """Carlos: flexibility, dont want to deal with anoying converisons."""
    tensors = state_dict_to_bytes(state_dict)
    return Parameters(tensors=[tensors], tensor_type="whatever")


def bytes_to_state_dict(bytes_data: bytes) -> StateDict:
    """Converts bytes back to a PyTorch state_dict."""
    bytes_io = BytesIO(bytes_data)
    state_dict = cast(StateDict, torch.load(bytes_io, weights_only=False))
    return state_dict


def parameters_to_state_dict(parameters: Parameters) -> StateDict:
    """Converts Flower Parameters back to a PyTorch state_dict."""
    bytes_data = parameters.tensors[0]
    return bytes_to_state_dict(bytes_data)


Task = Literal["train"] | Literal['plan_and_preprocess'] | Literal['extract_fingerprint']
Fingerprint = dict[str, list[list[int]] | list[list[float]] | list[float]]

class FlowerClient(fl.client.Client):
    num_samples: int | None = None
    fingerprint: Fingerprint | None = None
    local_fingerprint: Fingerprint | None = None

    def __init__(self, task: Task, args: Args):
        self.task: str = task
        self.dataset_name: str = maybe_convert_to_dataset_name(args.dataset_id)
        self.dataset_id: str = convert_dataset_name_to_id(self.dataset_name)
        self.extract_fingerprint: bool = False
        self.plan_experiment: bool = False
        self.preprocess_dataset: bool = False
        self.train: bool = False
        self.args: Args = args

        if self.task == "train":
            self.train = True
            # this calls run_training but is not running any training, I did not change the name of the method for compatibility with regular nnUnet.
            self.trainer: nnUNetTrainer = run_training(
                self.dataset_name,
                self.args.configuration,
                self.args.fold,
                args.trainer,
                args.plan,
                continue_training=args.is_continue,
                device=get_device(args),
                return_trainer=True,
            )

            self.trainer.initialize()
            self.model: torch.nn.Module = cast(torch.nn.Module, self.trainer.network)
            self.trainer.on_train_start()

        if self.task == "plan_and_preprocess":
            self.extract_fingerprint = True
            self.plan_experiment = True
            self.preprocess_dataset = True
            self.gpu_memory_target_in_gb = args.gpu_memory_target

            if args.num_process is None:
                self.num_process = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}.get(args.configuration, 4)
            else:
                self.num_process = args.num_process

        if self.task == "extract_fingerprint" or self.extract_fingerprint:
            self.extract_fingerprint = True

        self.preprocessed_output_folder: str = join(cast(str, nnUNet_preprocessed), self.dataset_name)

    def get_overlapping_keys(self, state_dict1, state_dict2) -> list[str]:
        """Find keys that are present in both state_dicts and have the same shape."""
        overlapping_keys = set(state_dict1.keys()).intersection(state_dict2.keys())
        compatible_keys = [
            key
            for key in overlapping_keys
            if state_dict1[key].shape == state_dict2[key].shape
        ]
        return compatible_keys

    def replace_overlapping_keys(self, target_state_dict, source_state_dict):
        """Replace keys in the target_state_dict with the values from source_state_dict for overlapping keys."""
        overlapping_keys = self.get_overlapping_keys(
            target_state_dict, source_state_dict
        )

        for key in overlapping_keys:
            target_state_dict[key] = source_state_dict[key]

        return target_state_dict

    def get_fingerprint(self) -> Fingerprint:
        if not self.local_fingerprint or not self.fingerprint:
            self.local_fingerprint = extract_fingerprint_dataset(
                self.dataset_id,
                fingerprint_extractor_class=DatasetFingerprintExtractor,
                num_processes=self.args.num_process,  # pyright: ignore[reportArgumentType]
                check_dataset_integrity=True,
                clean=True,
                verbose=False,
            )
            self.num_samples = len(self.local_fingerprint["shapes_after_crop"])
            save_json(
                self.local_fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint_local.json"),
            )
            self.fingerprint = self.local_fingerprint

        return self.fingerprint

    def get_parameters(self, ins) -> GetParametersRes:
        if self.extract_fingerprint:
            parameters = self.get_fingerprint()
            # print(f'Fingerprint with mean: {self.fingerprint["median_relative_size_after_cropping"]}')
        else:
            parameters = self.model.state_dict()

        parm = GetParametersRes(
            parameters=state_dict_to_parameters(parameters),
            status=Status(code=Code(0), message="caguento"),
        )
        return parm

    def set_parameters(self, parameters: Parameters):
        common_state_dict = parameters_to_state_dict(parameters)

        if self.extract_fingerprint:
            self.fingerprint = cast(Fingerprint, common_state_dict)
        else:
            # torch.save(common_state_dict,os.path.join(os.path.dirname(os.path.abspath(__file__)),'common_state_dict.arch'))
            # torch.save(self.model.state_dict(),os.path.join(os.path.dirname(os.path.abspath(__file__)),'local_state_dict.arch'))

            onset_subset_keys_dict = self.replace_overlapping_keys(
                self.model.state_dict(), common_state_dict
            )
            self.model.load_state_dict(onset_subset_keys_dict, strict=True)

    def fit(self, ins):
        self.set_parameters(ins.parameters)

        if self.extract_fingerprint:
            return FitRes(
                parameters=self.get_parameters(GetParametersIns({})).parameters,
                status=Status(code=Code(0), message="Fingerprint extracted"),
                num_examples=0,
                metrics={},
            )
        else:
            # adding try catch errors
            try:
                self.trainer.run_federated_train_round()
            except ValueError as e:
                logging.error(f"ValueError occurred: {e}")
            except RuntimeError as e:
                logging.error(f"RuntimeError occurred: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                raise

            print('----====---===---====----')
            print(self.trainer.logger.my_fantastic_logging)
            print('----====---===---====----')
            if len(tls := self.trainer.logger.my_fantastic_logging["train_losses"]) > 0:
                tl = np.round(tls[-1], decimals=4)
            else:
                tl = 0.0000
            fr = FitRes(
                parameters=self.get_parameters(GetParametersIns({})).parameters,
                status=Status(code=Code(0), message=""),
                num_examples=len(cast(NonDetMultiThreadedAugmenter, self.trainer.dataloader_train).generator._data.identifiers),
                metrics={"loss": float(tl)},
            )
            return fr

    def evaluate(self, ins):
        # We need to update to the aggregated parameters, otherwise the model will be evaluated on local weights
        self.set_parameters(ins.parameters)

        if self.extract_fingerprint:
            save_json(
                self.fingerprint,
                join(self.preprocessed_output_folder, "dataset_fingerprint.json"),
            )
            logging.info(
                f"Federated dataset fingerprint saved to {join(self.preprocessed_output_folder, 'dataset_fingerprint.json')}"
            )
            if self.plan_experiment:
                plans_identifier = cast(str, plan_experiments(
                    [self.dataset_id],
                    experiment_planner_class_name=self.args.planner,
                    gpu_memory_target_in_gb=self.args.gpu_memory_target,  # pyright: ignore[reportArgumentType]
                    preprocess_class_name="DefaultPreprocessor",
                    overwrite_target_spacing=self.args.overwrite_target_spacing,
                    overwrite_plans_name=self.args.plan,
                ))
                logging.info(f"Experiment plan created for {self.dataset_name}")
                if self.preprocess_dataset:
                    preprocess(
                        [self.dataset_id],
                        plans_identifier=plans_identifier,
                        configurations=[self.args.configuration],
                        num_processes=self.num_process,
                        verbose=False,
                    )
                    logging.info(f"Dataset {self.dataset_name} preprocessed")

            return EvaluateRes(
                status=Status(code=Code(0), message="Federated fingerprint saved"),
                loss=0.0,
                num_examples=1,
                metrics={},
            )

        print('----====---===---====----')
        print(self.trainer.logger.my_fantastic_logging)
        print('----====---===---====----')
        if len(vls := self.trainer.logger.my_fantastic_logging["val_losses"]) > 0:
            vl = np.round(vls[-1], decimals=4)
        else:
            vl = 0.0000
        if len(dcs := self.trainer.logger.my_fantastic_logging["dice_per_class_or_region"]) > 0:
            dc = [np.round(i, decimals=4) for i in dcs[-1]]
        else:
            dc = 0.0000

        er = EvaluateRes(
            status=Status(code=Code(0), message="yacasi"),
            loss=float(vl),
            num_examples=len(cast(NonDetMultiThreadedAugmenter, self.trainer.dataloader_val).generator._data.identifiers),
            metrics={"fg_dice": float(np.nanmean(dc))},
        )

        return er

def client_fn(ctx: Context) -> Client:
    task = cast(Task, ctx.run_config['task'])
    args: Args = Args(**{key[5:]: val for key, val in ctx.run_config.items() if key.startswith('args.')})  # pyright: ignore[reportArgumentType]
    client = FlowerClient(task, args)
    return client.to_client()

app = ClientApp(client_fn=client_fn)
