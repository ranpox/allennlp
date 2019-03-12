from typing import Tuple, Callable, List, Dict, Union
import torch
import numpy as np
import logging
import json
from allennlp.data.token_indexers.elmo_indexer_v2 import ELMoCharacterMapperV2
from allennlp.common.file_utils import cached_path
from allennlp.nn.activations import Activation
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import MultiHeadedAttention
from allennlp.nn.util import clone
from allennlp.nn.util import add_sentence_boundary_token_ids, remove_sentence_boundaries

logger = logging.getLogger(__name__)


def local_mask(size, width, device, left_to_right=True):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=-width - 1) * (1 - np.triu(np.ones(attn_shape), k=1))
    if not left_to_right:
        mask = np.flip(mask)
    mask = mask.astype('uint8')
    return torch.from_numpy(mask).to(device)


def attention_with_relative_position(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     rel_pos_score: torch.Tensor,
                                     mask: torch.Tensor = None,
                                     dropout: Callable = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
    scores += rel_pos_score

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = torch.nn.functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionWithRelativePosition(torch.nn.Module):
    def __init__(self, num_heads: int, input_dim: int, width: int,
                 left_to_right: bool,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        # We assume d_v always equals d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        self.width = width
        self.left_to_right = left_to_right
        # These linear layers are
        #  [query_projection, key_projection, value_projection, concatenated_heads_projection]
        self.linears = clone(torch.nn.Linear(input_dim, input_dim), 4)
        self.rel_pos_score = torch.nn.Parameter(torch.randn(num_heads, width + 1))
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            # Shape (batch_size, num_heads, timesteps, timesteps)
            mask = mask.unsqueeze(1).expand([-1, self.num_heads, -1, -1])

        nbatches = query.size(0)
        seq_len = query.size(-2)

        rel_pos_score = query.new_zeros(1, self.num_heads, seq_len, seq_len)
        if self.left_to_right:
            for current in range(seq_len):
                start = max(current - self.width, 0)
                length = current + 1 - start
                rel_pos_score[:, :, current, start: current + 1] = self.rel_pos_score[:, -length:]
        else:
            for current in range(seq_len):
                end = min(seq_len, current + self.width + 1)
                length = end - current
                rel_pos_score[:, :, current, current: end] = self.rel_pos_score[:, :length]

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [layer(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for layer, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = attention_with_relative_position(query, key, value, rel_pos_score=rel_pos_score,
                                                mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


class _SelfAttentiveLBLEncoder(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_heads: int,
                 n_layers: int,
                 n_highway: int,
                 use_position: bool = False,
                 use_relative_position: bool = False,
                 dropout: float = 0.0):
        super(_SelfAttentiveLBLEncoder, self).__init__()
        self.use_position = use_position
        self.use_relative_position_weights = use_relative_position
        self.n_layers = n_layers
        self.n_highway = n_highway
        self.n_heads = n_heads
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        forward_attns, backward_attns = [], []
        forward_blocks, backward_blocks = [], []

        for _ in range(n_layers):
            if self.use_relative_position_weights:
                forward_attn = MultiHeadedAttentionWithRelativePosition(n_heads, hidden_size, width=width + 1,
                                                                        left_to_right=True, dropout=dropout)
                backward_attn = MultiHeadedAttentionWithRelativePosition(n_heads, hidden_size, width=width + 1,
                                                                         left_to_right=False, dropout=dropout)
            else:
                forward_attn = MultiHeadedAttention(n_heads, hidden_size, dropout)
                backward_attn = MultiHeadedAttention(n_heads, hidden_size, dropout)

            forward_attns.append(forward_attn)
            backward_attns.append(backward_attn)
            forward_blocks.append(Highway(hidden_size, n_highway))
            backward_blocks.append(Highway(hidden_size, n_highway))

        self.forward_attns = torch.nn.ModuleList(forward_attns)
        self.backward_attns = torch.nn.ModuleList(backward_attns)

        self.forward_blocks = torch.nn.ModuleList(forward_blocks)
        self.backward_blocks = torch.nn.ModuleList(backward_blocks)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

    def get_output_dim(self):
        return self.hidden_size * 2

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        batch_size, sequence_len, dim = inputs.size()
        sequence_outputs = []

        forward_output_sequence = inputs
        backward_output_sequence = inputs

        forward_mask = local_mask(sequence_len, self.width, inputs.device)
        backward_mask = local_mask(sequence_len, self.width, inputs.device, left_to_right=False)

        for layer_index in range(self.n_layers):
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if self.use_position:
                forward_output_sequence = self.position(forward_output_sequence)
                backward_output_sequence = self.position(backward_output_sequence)

            forward_output_sequence = self.forward_attns[layer_index](forward_output_sequence,
                                                                      forward_output_sequence,
                                                                      forward_output_sequence,
                                                                      forward_mask)
            backward_output_sequence = self.backward_attns[layer_index](backward_output_sequence,
                                                                        backward_output_sequence,
                                                                        backward_output_sequence,
                                                                        backward_mask)

            forward_output_sequence = self.forward_blocks[layer_index](forward_output_sequence)
            backward_output_sequence = self.backward_blocks[layer_index](backward_output_sequence)

            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache

            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], dim=-1))

        return torch.stack(sequence_outputs, dim=0)


class _SelfAttentiveLBLEmbeddings(torch.nn.Module):
    def __init__(self,
                 n_d: int,
                 word2id: Dict[str, int],
                 input_field_name: str = None):
        super(_SelfAttentiveLBLEmbeddings, self).__init__()
        self.input_field_name = input_field_name
        self.word2id = word2id
        self.id2word = {i: word for word, i in word2id.items()}
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[ELMoCharacterMapperV2.oov_token]
        self.padid = word2id[ELMoCharacterMapperV2.pad_token]
        self.embedding = torch.nn.Embedding(self.n_V, n_d, padding_idx=self.padid)
        scale = np.sqrt(3.0 / n_d)
        self.embedding.weight.data.uniform_(-scale, scale)

    def forward(self, input_):
        return self.embedding(input_)

    def get_output_dim(self):
        return self.n_d


class _SelfAttentiveLBLCharacterEncoder(torch.nn.Module):
    def __init__(self,
                 output_dim: int,
                 char_embedder: _SelfAttentiveLBLEmbeddings,
                 filters: List[Tuple[int, int]],
                 n_highway: int,
                 activation: str):
        super(_SelfAttentiveLBLCharacterEncoder, self).__init__()

        self.output_dim = output_dim
        self.char_embedder = char_embedder
        self._beginning_of_sentence_characters = torch.from_numpy(
            np.array(char_embedder.word2id.get(ELMoCharacterMapperV2.bos_token))
        )
        self._end_of_sentence_characters = torch.from_numpy(
            np.array(char_embedder.word2id.get(ELMoCharacterMapperV2.eos_token))
        )

        self.emb_dim = 0
        assert char_embedder is not None

        self.convolutions = []
        char_embed_dim = char_embedder.n_d

        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=char_embed_dim,
                                   out_channels=num,
                                   kernel_size=width,
                                   bias=True)
            self.convolutions.append(conv)

        self.convolutions = torch.nn.ModuleList(self.convolutions)

        self.n_filters = sum(f[1] for f in filters)
        self.n_highway = n_highway

        self.highways = Highway(self.n_filters, self.n_highway, activation=Activation.by_name("relu")())
        self.emb_dim += self.n_filters
        self.activation = Activation.by_name(activation)()

        self.projection = torch.nn.Linear(self.emb_dim, self.output_dim, bias=True)

    def forward(self, inputs: torch.Tensor):
        # NOTE: by default, 0 is for padding.
        mask = ((inputs != self.char_embedder.padid).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
            inputs,
            mask,
            self._beginning_of_sentence_characters,
            self._end_of_sentence_characters
        )

        batch_size, seq_len = character_ids_with_bos_eos.size()[:2]
        character_embeddings = self.char_embedder(
            character_ids_with_bos_eos.view(batch_size * seq_len, -1)
        )

        character_embeddings = torch.transpose(character_embeddings, 1, 2)

        convs = []
        for i in range(len(self.convolutions)):
            convolved = self.convolutions[i](character_embeddings)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self.activation(convolved)
            convs.append(convolved)

        token_embedding = torch.cat(convs, dim=-1)
        token_embedding = self.highways(token_embedding)
        token_embedding = self.projection(token_embedding)

        return {
            'mask': mask_with_bos_eos,
            'token_embedding': token_embedding.view(batch_size, seq_len, -1)
        }


class _SelfAttentiveLBLBiLm(torch.nn.Module):
    def __init__(self,
                 options_file: str,
                 token_embedder_file: str,
                 encoder_file: str,
                 char_vocab_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None) -> None:
        super(_SelfAttentiveLBLBiLm, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)

        self._token_embedder_file = token_embedder_file
        self._encoder_file = encoder_file

        logger.info("Initializing SelfAttentiveLBL")
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
            char_embedder = _SelfAttentiveLBLEmbeddings(dim, mapping)
        else:
            char_embedder = None

        self._token_embedder = _SelfAttentiveLBLCharacterEncoder(
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
        self._encoder = _SelfAttentiveLBLEncoder(
            width=c['width'],
            input_size=c['projection_dim'],
            hidden_size=c['projection_dim'],
            n_heads=c['n_heads'],
            n_layers=c['n_layers'],
            n_highway=c['n_highway'],
            use_position=c.get('position', False),
            use_relative_position=c.get('relative_position_weights', False),
            dropout=self._options['dropout']
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
        return self._encoder.get_output_dim()

    def get_num_layers(self):
        # also counting layer-0
        return self._encoder.n_layers + 1


class SelfAttentiveLBL(torch.nn.Module):
    def __init__(self,
                 options_file: str,
                 token_embedder_file: str,
                 encoder_file: str,
                 char_vocab_file: str,
                 num_output_representations: int,
                 do_layer_norm: bool = False,
                 requires_grad: bool = False,
                 dropout: float = 0.,
                 vocab_to_cache: List[str] = None,
                 keep_sentence_boundaries: bool = False,
                 scalar_mix_parameters: List[float] = None) -> None:
        super(SelfAttentiveLBL, self).__init__()

        self._engine = _SelfAttentiveLBLBiLm(options_file=options_file,
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
