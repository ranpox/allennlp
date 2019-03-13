from typing import List, Dict, Union
import torch
import logging
import json
from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.selfattnlbl_encoder import UnicodeBilmEmbeddings,\
    UnicodeBilmCharacterEncoder
from allennlp.data import Vocabulary
from allennlp.nn.util import add_sentence_boundary_token_ids, remove_sentence_boundaries

logger = logging.getLogger(__name__)


class _UnicodeElmoBilm(torch.nn.Module):
    def __init__(self,
                 options_file: str,
                 token_embedder_file: str,
                 encoder_file: str,
                 char_vocab_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None) -> None:
        super(_UnicodeElmoBilm, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)

        self._token_embedder_file = token_embedder_file
        self._encoder_file = encoder_file

        logger.info("Initializing UnicodeElmoBilm")
        c = self._options['token_embedder']

        if c.get('char_dim', 0) > 0 or c.get('wordpiece_dim', 0) > 0:
            assert char_vocab_file is not None
            dim = c.get('char_dim', None) or c.get('wordpiece_dim')
            mapping = {}
            with open(char_vocab_file, 'r') as fin:
                for line in fin:
                    fields = line.strip().split('\t')
                    assert len(fields) == 2
                    token, i = fields
                    mapping[token] = int(i)
            char_embedder = UnicodeBilmEmbeddings(dim, mapping)
        else:
            char_embedder = None

        self._token_embedder = UnicodeBilmCharacterEncoder(
            output_dim=self._options['encoder']['projection_dim'],
            char_embedder=char_embedder,
            filters=c['filters'],
            n_highway=c['n_highway'],
            activation=c['activation'],
        )
        self._token_embedder.load_state_dict(
            torch.load(self._token_embedder_file,
                       map_location=lambda storage, loc: storage)
        )

        for p in self._token_embedder.parameters():
            p.require_grads = requires_grad
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None

        if vocab_to_cache:
            logger.info("Caching character cnn layers for words in vocabulary.")
            self.create_cached_cnn_embeddings(vocab_to_cache)

        c = self._options['encoder']
        self._encoder = ElmoLstm(
            input_size=c['projection_dim'],
            hidden_size=c['projection_dim'],
            cell_size=c['dim'],
            requires_grad=True,
            num_layers=c['n_layers'],
            recurrent_dropout_probability=self._options['dropout'],
            memory_cell_clip_value=c['cell_clip'],
            state_projection_clip_value=c['proj_clip'],
            stateful=False
        )
        self._encoder.load_state_dict(
            torch.load(self._encoder_file,
                       map_location=lambda storage, loc: storage)
        )
        for p in self._encoder.parameters():
            p.require_grads = requires_grad

    def forward(self,
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        if self._word_embedding is not None and word_inputs is not None:
            raise NotImplementedError("Caching word has not been implemented.")
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        outputs = self._encoder(type_representation, mask)
        output_tensors = [
            torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(outputs, outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
            'activations': output_tensors,
            'mask': mask
        }

    def get_output_dim(self):
        return self._encoder.hidden_size * 2

    def get_num_layers(self):
        # also counting layer-0
        return self._encoder.num_layers + 1


class UnicodeElmo(torch.nn.Module):
    def __init__(self,
                 options_file: str,
                 encoder_file: str,
                 token_embedder_file: str,
                 char_vocab_file: str,
                 num_output_representations: int,
                 do_layer_norm: bool = False,
                 requires_grad: bool = False,
                 dropout: float = 0.5,
                 keep_sentence_boundaries: bool = False,
                 vocab_to_cache: List[str] = None,
                 scalar_mix_parameters: List[float] = None) -> None:
        super(UnicodeElmo, self).__init__()

        self._engine = _UnicodeElmoBilm(options_file=options_file,
                                        token_embedder_file=token_embedder_file,
                                        encoder_file=encoder_file,
                                        char_vocab_file=char_vocab_file,
                                        requires_grad=requires_grad,
                                        vocab_to_cache=vocab_to_cache)

        self._has_cached_vocab = vocab_to_cache is not None
        self._keep_sentence_boundaries = keep_sentence_boundaries
        self._dropout = torch.nn.Dropout(p=dropout)
        self._scalar_mixes = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(
                self._engine.get_num_layers(),
                do_layer_norm=do_layer_norm,
                initial_scalar_parameters=scalar_mix_parameters,
                trainable=scalar_mix_parameters is None)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def forward(self,
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
            elif not self._has_cached_vocab:
                logger.warning("Word inputs were passed to ELMo but it does not have a cached vocab.")
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        encoder_output = self._engine(reshaped_inputs, reshaped_word_inputs)
        layer_activations = encoder_output['activations']
        mask_with_bos_eos = encoder_output['mask']

        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos
                )
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [representation.view(original_word_size + (-1, ))
                                    for representation in representations]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1, ))
                                    for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {'representations': elmo_representations, 'mask': mask}

    def get_output_dim(self):
        return self._engine.get_output_dim()


@TokenEmbedder.register("unicode_elmo_token_embedder")
class UnicodeElmoTokenEmbedder(TokenEmbedder):
    def __init__(self,
                 options_file: str,
                 encoder_file: str,
                 token_embedder_file: str,
                 char_vocab_file: str = None,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 requires_grad: bool = False,
                 projection_dim: int = None,
                 vocab_to_cache: List[str] = None,
                 scalar_mix_parameters: List[float] = None) -> None:
        super(UnicodeElmoTokenEmbedder, self).__init__()

        self._module = UnicodeElmo(options_file=options_file,
                                   token_embedder_file=token_embedder_file,
                                   encoder_file=encoder_file,
                                   char_vocab_file=char_vocab_file,
                                   num_output_representations=1,
                                   do_layer_norm=do_layer_norm,
                                   requires_grad=requires_grad,
                                   dropout=dropout,
                                   vocab_to_cache=vocab_to_cache,
                                   scalar_mix_parameters=scalar_mix_parameters)

        if projection_dim:
            self._projection = torch.nn.Linear(self._module.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._module.get_output_dim()

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> torch.Tensor:
        output = self._module(inputs, word_inputs)
        representations = output['representations'][0]
        if self._projection:
            projection = self._projection
            for _ in range(representations.dim() - 2):
                projection = TimeDistributed(projection)
            representations = projection(representations)
        return representations

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SelfAttentionLBLTokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('encoder_file')
        params.add_file_to_archive('token_embedder_file')
        params.add_file_to_archive('char_vocab_file')

        options_file = params.pop('options_file')
        encoder_file = params.pop('encoder_file')
        token_embedder_file = params.pop('token_embedder_file')
        char_vocab_file = params.pop('char_vocab_file')

        requires_grad = params.pop('requires_grad', False)
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        dropout = params.pop_float("dropout", 0.5)
        namespace_to_cache = params.pop("namespace_to_cache", None)
        if namespace_to_cache is not None:
            vocab_to_cache = list(vocab.get_token_to_index_vocabulary(namespace_to_cache).keys())
        else:
            vocab_to_cache = None
        projection_dim = params.pop_int("projection_dim", None)
        scalar_mix_parameters = params.pop('scalar_mix_parameters', None)
        params.assert_empty(cls.__name__)
        return cls(options_file=options_file,
                   encoder_file=encoder_file,
                   token_embedder_file=token_embedder_file,
                   char_vocab_file=char_vocab_file,
                   do_layer_norm=do_layer_norm,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim,
                   vocab_to_cache=vocab_to_cache,
                   scalar_mix_parameters=scalar_mix_parameters)
