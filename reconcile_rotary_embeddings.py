import torch
import torchtune
import flash_attn
import flash_attn.layers.rotary


class RotaryEmbeddingKyleLikeFA(torch.nn.Module):
    """
    Has the same function signature as FA, for interleaved=True and separate q, kv. 
    seqlen_offset = 0
    Does not operate inplace, but that's fine for how it's used in Nanotron. 
    """
    def __init__(self, dim: int, base: float):
        super().__init__()
        self.dim = dim
        self.base = float(base)

        self.max_seq_len = None
        self.rpe = None

    def forward(self, q, kv):
        bs, q_len, n_heads, _ = q.shape
        assert self.dim == _

        assert (bs, q_len, 2, n_heads, self.dim) == kv.shape

        if (self.rpe is None) or (self.max_seq_len != q_len):
            self.max_seq_len = q_len 
            self.rpe = torchtune.modules.RotaryPositionalEmbeddings(dim=self.dim, 
                                                                    max_seq_len=self.max_seq_len,
                                                                    base=self.base).to(device)
        q_out = self.rpe(q)
        kv_out = torch.stack((self.rpe(kv[:, :, 0]), kv[:, :, 1]), 2)
        return q_out, kv_out



if __name__ == "__main__":
    device = torch.device(0) 
    theta = 10000

    batch_size = 3
    dim_qk = 4
    q_len = 256
    kv_len = 256
    n_heads = 4

    max_seq_len = max(q_len, kv_len) 

    print(max_seq_len) 


    query_states = torch.rand(batch_size, q_len, n_heads, dim_qk, device=device) 
    key_value_states = torch.rand(batch_size, kv_len, 2, n_heads, dim_qk, device=device).contiguous()


    interleaved = True 
    # interleaved = False
    re1 = flash_attn.layers.rotary.RotaryEmbedding(dim=dim_qk, interleaved=interleaved, base=theta).to(device)
    re2 = torchtune.modules.RotaryPositionalEmbeddings(dim=dim_qk, max_seq_len=max_seq_len, base=theta).to(device)
    re3 = RotaryEmbeddingKyleLikeFA(dim=dim_qk, base=theta).to(device)


   
    print(key_value_states[:, :, 0].shape)

    out2 = re2(query_states)
    out3 = re2(key_value_states[:, :, 0]) 
    # out4 = re2(key_value_states[:, :, 1]) 

    out_eq = re3(query_states, kv=key_value_states)
 
    # torch.testing.assert_close(out2, query_states)
    out1 = re1(query_states, kv=key_value_states)

    torch.testing.assert_close(out_eq[0], out1[0])
    torch.testing.assert_close(out_eq[1], out1[1])


    # Do this second, since the computation is inplace
    torch.testing.assert_close(out1[0], query_states)

    test = torch.stack((out3, key_value_states[:, :, 1]), 2)
    torch.testing.assert_close(out1[1], test)
    # torch.testing.assert_close(out1[1][:, :, 0], out3) 


    torch.testing.assert_close(out1[0], out2)

    print("done")

