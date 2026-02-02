import torch
import sentencepiece as spm
from model import Storier
from torch.distributions.categorical import Categorical


class Inference:

    def __init__(self,
                 topk=50,
                 temp=0.7):

        self._topk = topk
        self._temp = temp

        self._skyer = Storier(num_layers=2,
                            input_dim=128,
                            hide_dim=96,
                            n_q_heads=12,
                            n_kv_heads=2,
                            max_len=16384,
                            num_vocs=50000,
                            cache_max_batch_size=1,
                            cache_max_seq_len=1024).cuda()
        self._skyer.eval()
        # self._skyer.load_state_dict(torch.load(
        #     "/root/workspace/myllm/mp_rank_00_model_states.pt")["module"])

        self._spm = spm.SentencePieceProcessor()
        self._spm.Load("tokenizer.model")

    def __call__(self, prompt):
        _vocs = prompt
        _prompt_ids = [2]+self._spm.Encode(prompt)
        _ids = torch.tensor(_prompt_ids, dtype=torch.long)[None].cuda()
        _id, _voc = self.forward(_ids, 0)
        _vocs += _voc
        _start_pos = _ids.shape[1]

        for _ in range(100):
            _id, _voc = self.forward(_id, _start_pos)
            if _id.item()==3:break
            _start_pos += 1
            _vocs += _voc
        return _vocs

    def forward(self, ids, start_pos):
        _os = self._skyer(ids, start_pos)
        _o = _os[:, -1]
        _weight, _indices = torch.topk(_o, self._topk, dim=-1)
        _probs = self._tsoftmax(_weight, self._temp)
        # _m = Categorical(_probs)
        # _s = _m.sample()
        _s = torch.multinomial(_probs, 1)
        _id = torch.gather(_indices, dim=-1, index=_s)
        return _id, self._spm.Decode(_id.item())

    @staticmethod
    def _tsoftmax(xs, temp=1.0):
        eps = 1e-5
        xs = xs-xs.mean()
        return torch.exp(xs/temp)/(torch.exp(xs/temp).sum(-1)+eps)


if __name__ == '__main__':

    env = Inference()

    voc = env("&lt;s&gt;user\n你是谁？\n&lt;/s&gt;&lt;s&gt;assistant\n")
    print(voc)
