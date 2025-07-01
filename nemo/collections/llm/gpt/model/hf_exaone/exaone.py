# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from nemo.collections.llm.gpt.model.base import GPTConfig, GPTModel, torch_dtype_from_mcore_config
from nemo.collections.llm.gpt.model.llama4_utils import get_llama4_layer_spec
from nemo.collections.llm.utils import Config
from nemo.export.trt_llm.nemo_ckpt_loader.nemo_file import load_distributed_model_weights
from nemo.lightning import OptimizerModule, io, teardown
from nemo.lightning.ckpt_utils import ADAPTER_META_FILENAME
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.lightning.io.state import TransformCTX, TransformFns, _ModelState
from nemo.lightning.pytorch.utils import dtype_from_hf
from nemo.utils import logging

try:
    from megatron.core.transformer.spec_utils import ModuleSpec

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if TYPE_CHECKING:
    from megatron.core.models.gpt.gpt_model import GPTModel as MCoreGPTModel
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from .configuration_exaone import ExaoneConfig as HFExaoneConfig
    from .modeling_exaone import ExaoneForCausalLM

    from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
    from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


@dataclass
class ExaoneConfig(GPTConfig):
    """Configuration class for Llama models.

    Extends GPTConfig with specific settings optimized for Llama architectures.
    Includes configurations for normalization, activation functions, and various
    architecture-specific options.
    """

    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 4096
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    # Fusions
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    use_transformer_engine_op_fuser: Optional[bool] = None


    scale_factor: float = 8.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    old_context_len: int = 8192
    init_method_std: float = 0.02

    def configure_model(self, tokenizer, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Llama 3.1 model.

        Extends the base configuration with Llama 3.1 specific RoPE scaling.

        Args:
            tokenizer: Tokenizer used with the model
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        model = super().configure_model(tokenizer, pre_process, post_process, vp_stage)
        # Apply rope scaling for Llama3.1 model
        model.rotary_pos_emb.inv_freq = apply_rope_scaling(
            model.rotary_pos_emb.inv_freq,
            factor=self.scale_factor,
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )
        return model


@dataclass
class Exaone35Config8B(ExaoneConfig):
    """Configuration for a 7.8B parameter Exaone 3.5 model.
    """

    rotary_base: int = 500_000
    seq_length: int = 32768
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32



class ExaoneModel(GPTModel):
    """Llama model implementation based on the GPT model architecture.

    This class provides a high-level interface for Llama models,
    implementing the specific architecture and settings needed for Llama models.
    """

    def __init__(
        self,
        config: Annotated[Optional[ExaoneConfig], Config[ExaoneConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
        model_context_managers: Optional[List] = [],
    ):
        super().__init__(
            config or ExaoneConfig(),
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
            model_context_managers=model_context_managers,
        )


class MLPerfLoRAExaoneModel(ExaoneModel):
    """Memory-optimized Llama model implementation for MLPerf LoRA fine-tuning.

    This class wraps LlamaModel and adds context managers around configure_model
    to reduce memory consumption during initialization. It applies techniques like
    avoiding unnecessary gradients and using FP8 parameter initialization.

    Changes made here are experimental, proceed with caution.
    """

    def __init__(
        self,
        config: Annotated[Optional[ExaoneConfig], Config[ExaoneConfig]] = None,
        optim: Optional[OptimizerModule] = None,
        tokenizer: Optional["TokenizerSpec"] = None,
        model_transform: Optional[Callable[[nn.Module], nn.Module]] = None,
    ):
        # Apply context manager to reduce memory by avoiding unnecessary gradients
        model_context_managers = [torch.no_grad()]
        super().__init__(
            config or ExaoneConfig(),
            optim=optim,
            tokenizer=tokenizer,
            model_transform=model_transform,
            model_context_managers=model_context_managers,
        )


@io.model_importer(ExaoneModel, "hf")
class HFExaoneImporter(io.ModelConnector["ExaoneForCausalLM", ExaoneModel]):
    """Importer for converting Hugging Face Llama models to NeMo format.

    This class handles the conversion of Hugging Face's LlamaForCausalLM models
    to NeMo's LlamaModel format, including weight mapping and configuration translation.
    """

    def init(self) -> ExaoneModel:
        """Initialize a NeMo LlamaModel instance.

        Returns:
            LlamaModel: Initialized NeMo Llama model with the appropriate configuration
                        and tokenizer.
        """
        return ExaoneModel(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path, hf_cache_dir=None) -> Path:
        """Apply the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
        """
        from transformers import AutoConfig, AutoModelForCausalLM
    
        source = AutoModelForCausalLM.from_pretrained(str(self), torch_dtype='auto', trust_remote_code=True, cache_dir=hf_cache_dir)

        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted Exaone Model to Nemo, model saved to {output_path} in {source.dtype}.")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from HF format to NeMo format.

        Maps the weights from the HF model to the NeMo model according to
        the appropriate mapping scheme.

        Args:
            source: Source HF model
            target: Target NeMo model

        Returns:
            The result of applying the transforms
        """
        mapping = {
            "transformer.wte.weight": "embedding.word_embeddings.weight",
            "transformer.h.*.attn.attention.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "transformer.h.*.ln_1.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "transformer.ln_f.weight": "decoder.final_layernorm.weight",
            "lm_head.weight": "output_layer.weight",
        }
        if getattr(source.config, "tie_word_embeddings", False):
            # llama 3.2 1B and 3B models have no shared input output embeddings
            del mapping["lm_head.weight"]

        transforms = [
            io.state_transform(
                source_key=(
                    "transformer.h.*.attn.attention.q_proj.weight",
                    "transformer.h.*.attn.attention.k_proj.weight",
                    "transformer.h.*.attn.attention.v_proj.weight",
                ),
                target_key="decoder.layers.*.self_attention.linear_qkv.weight",
                fn=TransformFns.merge_qkv,
            )
        ]
        
        
        # Dense Mapping
        mapping.update(
            {
                "transformer.h.*.ln_2.weight": (
                    "decoder.layers.*.mlp.linear_fc1.layer_norm_weight"
                ),
                "transformer.h.*.mlp.c_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            }
        )
        transforms.append(
            io.state_transform(
                source_key=("transformer.layers.*.mlp.c_fc_0.weight", "transformer.layers.*.mlp.c_fc_1.weight"),
                target_key="decoder.layers.*.mlp.linear_fc1.weight",
                fn=TransformFns.merge_fc1,
            )
        )

        return io.apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self) -> "AutoTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            AutoTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """
        from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))


    @property
    def config(self) -> ExaoneConfig:
        """Create a NeMo LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self), trust_remote_code=True)
        try:
            generation_config = GenerationConfig.from_pretrained(str(self), trust_remote_code=True)
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        # if getattr(source, 'rope_scaling', None) is not None and source.rope_scaling.get('rope_type') == 'llama3':
        #     # Apply Llama3.1 customize rope scaling
        #     cls = partial(Exaone35Config8B, scale_factor=source.rope_scaling.get("factor", 8.0))
        # else:
        cls = ExaoneConfig

        args = {}
        

        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=(
                source.intermediate_size
                if not getattr(source, 'intermediate_size_mlp', None)
                else source.intermediate_size_mlp
            ),
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.layer_norm_epsilon,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=source.vocab_size,
            kv_channels=getattr(source, "head_dim", None),
            **args,
        )

        return output


@io.model_exporter(ExaoneModel, "hf")
class HFExaoneExporter(io.ModelConnector[ExaoneModel, "ExaoneForCausalLM"]):
    """Exporter for converting NeMo Llama models to Hugging Face format.

    This class handles the conversion of NeMo's LlamaModel to Hugging Face's
    LlamaForCausalLM format, including weight mapping and configuration translation.
    """

    def init(self, dtype=torch.bfloat16) -> "ExaoneForCausalLM":
        """Initialize a HF LlamaForCausalLM instance.

        Args:
            dtype: Data type for model parameters

        Returns:
            LlamaForCausalLM: Initialized HF Llama model
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF model
        """
        
        source, _ = self.nemo_load(str(self))
        source_config = source.config
        target = self.init(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target, source_config)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(output_path, state_dict=state_dict)
        else:
            target.save_pretrained(output_path)

        try:
            self.tokenizer.tokenizer.save_pretrained(output_path)
        except Exception:
            logging.warning("Failed to save tokenizer")

        return output_path

    def convert_state(self, source, target, source_config=None):
        # pylint: disable=C0301
        """Convert state dict from NeMo format to HF format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme.

        Args:
            source: Source NeMo model
            target: Target HF model
            source_config: Source NeMo config (optional, used for Llama4)

        Returns:
            The target model with weights transferred from source
        """
        is_llama4 = self.is_llama4()
        if is_llama4:
            assert source_config is not None
            source = self._modify_llama4_source_state(source, source_config)

        mapping = {
            "decoder.layers.*.self_attention.linear_proj.weight": "transformer.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "transformer.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "transformer.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "transformer.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "transformer.norm.weight",
        }

        transforms = [
            io.state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "transformer.layers.*.self_attn.q_proj.weight",
                    "transformer.layers.*.self_attn.k_proj.weight",
                    "transformer.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            io.state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("transformer.layers.*.mlp.gate_proj.weight", "transformer.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
            io.state_transform(
                source_key="embedding.word_embeddings.weight",
                target_key="transformer.embed_tokens.weight",
                fn=TransformFns.prune_padding,
            ),
        ]
        if not self.config.tie_word_embeddings:
            transforms.append(
                io.state_transform(
                    source_key="output_layer.weight",
                    target_key="lm_head.weight",
                    fn=TransformFns.prune_padding,
                )
            )

        

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def tokenizer(self) -> "TokenizerSpec":
        """Get the tokenizer from the NeMo model.

        Returns:
            TokenizerSpec: Tokenizer from the NeMo model
        """
        return io.load_context(str(self), subpath="model").tokenizer

    @property
    def config(self) -> "HFExaoneConfig":
        """Create a HF LlamaConfig from the NeMo model config.

        Translates the NeMo configuration parameters to the equivalent HF
        configuration.

        Returns:
            HFLlamaConfig: HF configuration for Llama models
        """
        source: ExaoneConfig = io.load_context(str(self), subpath="model.config")
        
        rope_scaling = None
        # For Llama 3.1 and Llama 3.2, rope_scaling is used and thus needed to parsed to the config
        
        rope_scaling = {
            'factor': source.scale_factor,
            'low_freq_factor': source.low_freq_factor,
            'high_freq_factor': source.high_freq_factor,
            'original_max_position_embeddings': source.old_context_len,
            'rope_type': 'llama3',
        
        }

        return HFExaoneConfig(
            architectures=["ExaoneForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            rope_theta=source.rotary_base,
            vocab_size=self.tokenizer.vocab_size,
            tie_word_embeddings=source.share_embeddings_and_output_weights,
            rope_scaling=rope_scaling,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id,
        )

    def ckpt_load(self, path: Path) -> Tuple[Dict, Any]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            Tuple[Dict, Any]: The loaded state dict and the yaml config object.
        """
        model_yaml = path / "context" / "model.yaml"
        if not model_yaml.exists():
            raise FileNotFoundError("model.yaml is not found in the context folder of the checkpoint.")
        with open(model_yaml, 'r') as stream:
            config = yaml.safe_load(stream)

        dist_ckpt_folder = path / "weights"
        state_dict = {}

        dict_to_obj = lambda d: (
            type('Config', (), {kk: dict_to_obj(vv) for kk, vv in d.items()}) if isinstance(d, dict) else d
        )
        config_obj = dict_to_obj(config['config'])
        langauge_layers = config_obj.num_layers
        distributed_model_weights = load_distributed_model_weights(dist_ckpt_folder, True).items()
        for k, v in distributed_model_weights:
            if '_extra_state' in k:
                continue
            new_k = k.replace("module.", "")
            if 'layers' in new_k and v.size(0) == langauge_layers:
                # Only split layers
                for i in range(v.size(0)):
                    state_dict[new_k.replace('layers', f'layers.{str(i)}')] = v[i]
            state_dict[new_k] = v

        return state_dict, config_obj


@io.model_exporter(ExaoneModel, "hf-peft")
class HFLlamaPEFTExporter(HFExaoneExporter):
    """Exporter for converting NeMo Llama models with PEFT adapters to Hugging Face format.

    This class extends HFLlamaExporter to handle Parameter-Efficient Fine-Tuning (PEFT)
    adapters, specifically LoRA and DoRA adapters.
    """

    def init(self, dtype=torch.bfloat16) -> "AutoPeftModelForCausalLM":
        """Initialize a HF PEFT model.

        Args:
            dtype: Data type for model parameters

        Returns:
            AutoPeftModelForCausalLM: Initialized HF PEFT model
        """
        from peft import get_peft_model

        model = super().init(dtype=dtype)

        # Infer base model checkpoint from checkpoint metadata file
        adapter_meta_path = ckpt_to_weights_subdir(str(self), is_saving=False) / ADAPTER_META_FILENAME
        with open(adapter_meta_path, "r") as f:
            model_ckpt_path = json.load(f)['model_ckpt_path']
        model.name_or_path = '/'.join(model_ckpt_path.split("/")[-2:])

        return get_peft_model(model, self.peft_config, autocast_adapter_dtype=False)

    def apply(self, output_path: Path) -> Path:
        """Apply the conversion from NeMo PEFT model to HF format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved HF PEFT model
        """
        from nemo.collections.llm.peft import CanonicalLoRA, DoRA, LoRA

        self.peft_obj: Union[LoRA, DoRA, CanonicalLoRA] = io.load_context(str(self), subpath="model.model_transform")

        source, _ = self.nemo_load(str(self))
        target = self.init(torch_dtype_from_mcore_config(source.config))
        target = self.convert_state(source, target)
        target = target.cpu()
        target.save_pretrained(output_path, save_embedding_layers=False)

        return output_path

    def convert_state(self, source, target):
        """Convert state dict from NeMo PEFT model to HF PEFT format.

        Maps the weights from the NeMo model to the HF model according to
        the appropriate mapping scheme for PEFT adapters.

        Args:
            source: Source NeMo model with PEFT adapters
            target: Target HF model

        Returns:
            The target model with weights transferred from source
        """
        from nemo.collections.llm.peft import CanonicalLoRA

        # nemo and HF prefixes
        pn = "decoder.layers."
        ph = "base_model.model.model.layers."

        # linear_proj and linear_fc2 prefixes
        p_proj = "self_attention.linear_proj.adapter"
        p_fc2 = "mlp.linear_fc2.adapter"

        # linear_qkv and linear_fc1 prefixes
        p_qkv = "self_attention.linear_qkv.adapter"
        p_fc1 = "mlp.linear_fc1.adapter"

        mapping = {
            # linear_proj for both canonical and performant lora
            f"{pn}*.{p_proj}.linear_in.weight": f"{ph}*.self_attn.o_proj.lora_A.default.weight",
            f"{pn}*.{p_proj}.linear_out.weight": f"{ph}*.self_attn.o_proj.lora_B.default.weight",
            # linear_fc2 for both canonical and performant lora
            f"{pn}*.{p_fc2}.linear_in.weight": f"{ph}*.mlp.down_proj.lora_A.default.weight",
            f"{pn}*.{p_fc2}.linear_out.weight": f"{ph}*.mlp.down_proj.lora_B.default.weight",
        }
        transforms = []

        if isinstance(self.peft_obj, CanonicalLoRA):
            mapping.update(
                {
                    # linear_qkv for canonical lora
                    f"{pn}*.{p_qkv}.adapter_q.linear_in.weight": f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_q.linear_out.weight": f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_in.weight": f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_k.linear_out.weight": f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_in.weight": f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                    f"{pn}*.{p_qkv}.adapter_v.linear_out.weight": f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                    # linear_fc1 for canonical lora
                    f"{pn}*.{p_fc1}.adapter_up.linear_in.weight": f"{ph}*.mlp.up_proj.lora_A.default.weight",
                    f"{pn}*.{p_fc1}.adapter_up.linear_out.weight": f"{ph}*.mlp.up_proj.lora_B.default.weight",
                    f"{pn}*.{p_fc1}.adapter_gate.linear_in.weight": f"{ph}*.mlp.gate_proj.lora_A.default.weight",
                    f"{pn}*.{p_fc1}.adapter_gate.linear_out.weight": f"{ph}*.mlp.gate_proj.lora_B.default.weight",
                }
            )
        else:
            transforms.extend(
                [
                    # linear_qkv for performant lora
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_in.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_A.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_A.default.weight",
                        ),
                        fn=TransformFns.duplicate3,
                    ),
                    io.state_transform(
                        source_key=f"{pn}*.self_attention.linear_qkv.adapter.linear_out.weight",
                        target_key=(
                            f"{ph}*.self_attn.q_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.k_proj.lora_B.default.weight",
                            f"{ph}*.self_attn.v_proj.lora_B.default.weight",
                        ),
                        fn=TransformFns.split_qkv,
                    ),
                    # linear_fc1 for performant lora
                    io.state_transform(
                        source_key=f"{pn}*.mlp.linear_fc1.adapter.linear_in.weight",
                        target_key=(
                            f"{ph}*.mlp.gate_proj.lora_A.default.weight",
                            f"{ph}*.mlp.up_proj.lora_A.default.weight",
                        ),
                        fn=TransformFns.duplicate2,
                    ),
                    io.state_transform(
                        source_key=f"{pn}*.mlp.linear_fc1.adapter.linear_out.weight",
                        target_key=(
                            f"{ph}*.mlp.gate_proj.lora_B.default.weight",
                            f"{ph}*.mlp.up_proj.lora_B.default.weight",
                        ),
                        fn=TransformFns.split_fc1,
                    ),
                ]
            )

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=transforms,
        )

    @property
    def peft_config(self) -> "PeftConfig":
        """Create a PEFT config for the HF model.

        Translates the NeMo PEFT configuration to the equivalent HF PEFT
        configuration.

        Returns:
            PeftConfig: HF PEFT configuration
        """
        from peft import LoraConfig

        from nemo.collections.llm.peft import DoRA

        assert (
            not self.peft_obj.dropout or self.peft_obj.dropout_position == 'pre'
        ), "LoRA dropout_position must be 'pre' to convert to HF."

        NEMO2HF = {
            'linear_q': ['q_proj'],
            'linear_k': ['k_proj'],
            'linear_v': ['v_proj'],
            'linear_qkv': ['q_proj', 'k_proj', 'v_proj'],
            'linear_proj': ['o_proj'],
            'linear_fc1_up': ['up_proj'],
            'linear_fc1_gate': ['gate_proj'],
            'linear_fc1': ['up_proj', 'gate_proj'],
            'linear_fc2': ['down_proj'],
        }

        # Infer HF target modules from NeMo target modules
        hf_target_modules = []
        for tm in self.peft_obj.target_modules:
            hf_target_modules.extend(NEMO2HF[tm])

        return LoraConfig(
            r=self.peft_obj.dim,
            target_modules=hf_target_modules,
            lora_alpha=self.peft_obj.alpha,
            lora_dropout=self.peft_obj.dropout,
            use_dora=isinstance(self.peft_obj, DoRA),
        )


def apply_rope_scaling(
    inv_freq,
    factor: float = 8.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
):
    """Apply RoPE scaling for extending context length in Llama models.

    This implements the NTK-aware RoPE scaling method used in Llama 3.1 models to
    extend context length beyond the original training length.

    Args:
        inv_freq: Original inverse frequency tensor
        factor: Scaling factor for context length extension
        low_freq_factor: Factor for low frequency components
        high_freq_factor: Factor for high frequency components
        old_context_len: Original context length

    Returns:
        torch.Tensor: Modified inverse frequency tensor for extended context
    """
    logging.info(
        f"Apply rope scaling with factor={factor}, low_freq_factor={low_freq_factor}, "
        f"high_freq_factor={high_freq_factor}, old_context_len={old_context_len}."
    )

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


@staticmethod
def split_moe(ctx: TransformCTX, tensor: torch.Tensor):
    """
    Split interleave-concatenated MoE expert weights.

    Args:
        ctx: Transformation context containing model configuration.
        tensor: The tensor containing concatenated expert weights.

    Returns:
        A list of tensors, each corresponding to an expert's weights.
    """
    megatron_config = ctx.source.config

    num_experts = megatron_config.num_local_experts
    expert_tensors = torch.chunk(tensor, num_experts, dim=1)

    return expert_tensors


__all__ = [
    "ExaoneConfig",
    "Exaone35Config8B",
    "ExaoneModel",
    "MLPerfLoRAExaoneModel",
]


if __name__ == '__main__':
    model = HFExaoneImporter('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')
    breakpoint()