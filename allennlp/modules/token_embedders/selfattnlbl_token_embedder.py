from typing import List
import torch
import logging
from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.selfattnlbl_encoder import SelfAttentiveLBL
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)


@TokenEmbedder.register("selfattnlbl_token_embedder")
class SelfAttentionLBLTokenEmbedder(TokenEmbedder):
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
        super(SelfAttentionLBLTokenEmbedder, self).__init__()

        self._module = SelfAttentiveLBL(options_file=options_file,
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

        options_file = params.pop('options_file')
        encoder_file = params.pop('encoder_file')
        token_embedder_file = params.pop('token_embedder_file')
        char_vocab_file = params.pop('char_vocab_file', None)

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
