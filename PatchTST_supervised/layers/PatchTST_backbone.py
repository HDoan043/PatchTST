__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

ratio_patches = [1,1.5,2,2.5,3]
# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, 
                 context_window:int, 
                 target_window:int, 
                 patch_len:int, 
                 stride:int, 
                 max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, 
                 d_model=128, 
                 n_heads=16, 
                 d_k:Optional[int]=None, 
                 d_v:Optional[int]=None,
                 d_ff:int=256, 
                 norm:str='BatchNorm', 
                 attn_dropout:float=0., 
                 dropout:float=0., 
                 act:str="gelu", 
                 key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, 
                 attn_mask:Optional[Tensor]=None, 
                 res_attention:bool=True, 
                 pre_norm:bool=False, 
                 store_attn:bool=False,
                 pe:str='zeros', 
                 learn_pe:bool=True, 
                 fc_dropout:float=0., 
                 head_dropout = 0, 
                 padding_patch = None,
                 pretrain_head:bool=False, 
                 head_type = 'flatten', 
                 individual = False, 
                 revin = True, 
                 affine = True, 
                 subtract_last = False,
                 verbose:bool=False, 
                 multi_patches = False,
                 hybrid = False,
                 bert_ratio = None,
                 **kwargs):
        
        super().__init__()

        self.seq_len = context_window
        self.pred_len = target_window
        self.hybrid = hybrid
                     
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_length = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.multi_patches = multi_patches
        if self.multi_patches:
            self.patch_len = [int(patch_len * ratio) for ratio in ratio_patches]
            patch_len = self.patch_len
            patch_num = [int((context_window - each)/stride + 1) for each in self.patch_len]
        else:
            patch_num = int((context_window - patch_len)/stride + 1)
            
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            if self.multi_patches:
                patch_num = [each+1 for each in patch_num]
            else:
                patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, multi_patches = multi_patches, hybrid = hybrid, nvars = c_in, **kwargs)

        # Head
        self.head_nf = d_model * patch_num if not multi_patches else d_model * patch_num[0]
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        if hybrid:
            self.padding_for_hybrid = nn.ReflectionPad1d((0, context_window - target_window))
            self.reconstruct_head = Reconstruct_Head(n_heads, d_model, 2, patch_len)
            self.reconstruct_loss = nn.MSELoss(reduction = 'none')
            self.forecast_loss = nn.MSELoss(reduction = 'none')
            self.combine_loss = HybridLoss()

        if bert_ratio:
            self.bert = True
            self.bert_mask = BertMask(bert_ratio = bert_ratio)
            self.bert_head = BertHead(d_model = d_model, patch_len = patch_len)
            self.bert_loss = nn.MSELoss(reduction = 'none')
        else:
            self.bert = False
            
    def forward(self, z):                                                                   # z: [bs x nvars x (seq_len + pred_len)]
        if self.hybrid:
            # RECONSTRUCT
            old_z = z
            reconstruct_z = self.padding_for_hybrid(z)                                                         # z: [bs x nvars x 2 * seq_len]
            reconstruct_z = reconstruct_z.unfold(dimension = -1, size = self.seq_len, step = self.seq_len)     # z: [bs x nvars x 2 x seq_len]
            
            if self.padding_patch == 'end':
                bs = z.shape[0]
                reconstruct_z = torch.reshape(reconstruct_z, (reconstruct_z.shape[0]* reconstruct_z.shape[1], reconstruct_z.shape[2], reconstruct_z.shape[3]))  # z: [bs * nvars x (pred_len+1) x seq_len]
                reconstruct_z = self.padding_patch_layer(reconstruct_z)                                                
                reconstruct_z = torch.reshape(reconstruct_z, (bs, -1, reconstruct_z.shape[1], reconstruct_z.shape[2]))             # z: [bs x nvars x (pred_len+1) x seq_len]
            gt_reconstruct_z = reconstruct_z.unfold(dimension = -1, size = self.patch_length, step = self.stride)      # z: [bs x nvars x (pred_len + 1) x patch_num x patch_len]
            reconstruct_z = self.backbone(gt_reconstruct_z)                                                            # z: [bs x nvars x (pred_len +1) x patch_num x d_model]
            reconstruct_z = reconstruct_z.permute(0,1,3,2,4)                                                           # z: [bs x nvars x patch_num x (pred_len + 1) x d_model] 
            reconstruct_z = self.reconstruct_head(reconstruct_z)                                                       # z: [bs x nvars x (pred_len + 1) x patch_num x patch_len]

            reconstruct_loss = self.reconstruct_loss(reconstruct_z, gt_reconstruct_z)                                  # reconstruct_loss: [bs x nvars x (pred_len + 1) x patch_num x patch_len]
            bs, nvars, sn, pn, pl = reconstruct_loss.shape
            reconstruct_loss = torch.reshape(reconstruct_loss, (bs, nvars*sn*pn*pl))                                   # reconstruct_loss: [bs x nvars * (pred_len + 1) x patch_num x patch_len]
            reconstruct_loss = reconstruct_loss.mean(dim = 1).squeeze()                                                # reconstruct_loss: [bs]
            
            # FORECASTING
            forecast_z = old_z[:, :, :self.seq_len]
            gt_forecast_z = old_z[:, :, self.seq_len:]
            # norm
            if self.revin: 
                forecast_z = forecast_z.permute(0,2,1)
                forecast_z = self.revin_layer(forecast_z, 'norm')
                forecast_z = forecast_z.permute(0,2,1)
                
            # do patching
            if self.padding_patch == 'end':
                forecast_z = self.padding_patch_layer(forecast_z)
            forecast_z = forecast_z.unfold(dimension=-1, size=self.patch_length, step=self.stride)                # z: [bs x nvars x patch_num x patch_len]

             # model
            forecast_z = self.backbone(forecast_z)                                                                # z: [bs x nvars x d_model x patch_num]
            forecast_z = self.head(forecast_z)                                                                    # z: [bs x nvars x target_window] 
            
            # denorm
            if self.revin: 
                forecast_z = forecast_z.permute(0,2,1)
                forecast_z = self.revin_layer(forecast_z, 'denorm')
                forecast_z = forecast_z.permute(0,2,1)

            forecast_loss = self.forecast_loss(forecast_z, gt_forecast_z)                                        # forecast_loss: [bs x nvars x target_window]
            bs, nvars, target_window = forecast_loss.shape
            forecast_loss = torch.reshape(forecast_loss, (bs, nvars* target_window))                             # forecast_loss: [bs x nvars * target_window]
            forecast_loss = forecast_loss.mean(dim = 1).squeeze()                                                # forecast_loss: [bs x 1]


            # BERT
            if self.bert:
                # do patching
                if self.padding_patch == 'end':
                    bert_z = self.padding_patch_layer(z)
                patching_bert_z = bert_z.unfold(dimension=-1, size=self.patch_length, step=self.stride)       # z: [bs x nvars x patch_num x patch_len]

                # mask
                bert_z, mask = self.bert_mask(old_z)                                                          # z: [bs x nvars x patch_num x patch_len]
                gt = old_z * (1 - mask)
                bert_z = self.backbone(bert_z)                                                                # z: [bs x nvars x d_models x patch_num]
                # recover
                bert_z = self.bert_head(bert_z)                                                               # z: [bs x nvars x patch_num x patch_len]

                # forecast masked positions
                bert_z = bert_z * (1 - mask)

                # bert loss
                bert_loss = self.bert_loss(bert_z, gt)                                                        # bert_loss: [bs x nvar x patch_num x patch_len]
                bs, nvars, patch_num , patch_len = bert_loss.shape
                bert_loss = torch.reshape(bert_loss, (bs, nvars * patch_num * patch_len))                     # bert_loss: [bs x nvars * patch_num * patch_len]
                bert_loss = bert_loss.mean(dim = 1).squeeze()                                                 # bert_loss: [bs x ]
                
            # COMBINING LOSSES
            combining_loss = self.combine_loss(forecast_loss, reconstruct_loss)                                  # combine_loss: [bs x 1]

            return combining_loss

        # if self.hybrid == 1:
        #     reconstruct_z = reconstruct_z
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        if self.multi_patches:
            old_z = z
            z = []
            for patch in self.patch_len:
                tem = old_z
                tem = tem.unfold(dimension = -1, size = patch, step = self.stride)
                tem = tem.permute(0,1,3,2)                                                      # tem: [bs x nvars x patch_num x patch_len]
                z.append(tem)                                                                   # z: [len_ratio_patches x [bs x nvars x patch_num_i x patch_len_i]]
            
        else:
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

class Combine_Channels(nn.Module):
    def __init__(self, in_channels, d_model, out_channels):
        super().__init__()
        self.normalize = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(d_model, 8, batch_first = True)
        ls_ff = []
        for _ in range(3):
            ls_ff.extend(
                [nn.Linear(in_channels, 1024), 
                nn.ReLU(), 
                nn.Linear(1024, in_channels), 
                nn.ReLU(), 
                nn.Linear(in_channels, in_channels)]
            )  
        
        self.ff = nn.Sequential( 
            *ls_ff,
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):                                # x: [bs x nvars x (seq_num x ) patch_num x d_model] 
        # permute x in order to make the nvars be the last shape because the LayerNorm will work with the last dim of samples
        if len(x.shape) == 5:
            x = x.permute(0,2,3,4,1)                     # x: [bs x seq_num x patch_num x d_model x nvars]
        else: 
            x = x.permute(0,2,3,1)                       # x: [bs x patch_num x d_model x nvars]
        x = self.normalize(x)
        if len(x.shape) == 5:
            x = x.permute(0,1,2,4,3)                     # x: [bs x seq_num x patch_num x nvars x d_model]
        else:
            x = x.permute(0,1,3,2)                       # x: [bs x patch_num x nvars x d_model]
        old_shape = x.shape                              # x: [bs x (seq_num x) patch_num x nvars x d_model]
        if len(old_shape) == 5:
            x = torch.reshape( x, (x.shape[0]*x.shape[1]*x.shape[2], x.shape[3], x.shape[4])) # x: [bs * seq_num * patch_num x nvars x d_model]
        else:
            x = torch.reshape( x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))  # x: [bs * patch_num x nvars x d_model]
        att, _ = self.attention(x,x,x)
        x = att + x                                      # x: [bs * nvars * (seq_num x ) patch_num x d_model]
        x = torch.reshape(x, old_shape)                  # x: [bs x (seq_num x) patch_num x nvars x d_model]
        if len(x.shape) == 5:
            x = x.permute(0,1,2,4,3)                     # x: [bs x seq_num x patch_num x d_model x nvars]
        else:
            x = x.permute(0,1,3,2)                       # x: [bs x patch_num x d_model x nvars]
        x = self.ff(x)                                   # x: [bs x (seq_num x ) patch_num x d_model x num_out_channels]
        if len(x.shape) == 5:
            x = x.permute(0, 4, 1, 2, 3)                 # x: [bs x num_out_channels x seq_num x patch_num x d_model]
        else:
            x = x.permute(0, 3, 1, 2)                    # x: [bs x num_out_channels x patch_num x d_model]
        return x
        
class Reconstruct_Head(nn.Module):
    def __init__(self, n_heads, d_model, seq_num, patch_len):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first = True)
        ls_ff = []
        for _ in range(3):
            ls_ff.extend([nn.Linear(d_model, 512), nn.ReLU(), nn.Linear(512, d_model), nn.ReLU()])
        self.ff = nn.Sequential(*ls_ff)
        self.reconstruct = nn.Linear(d_model, patch_len)
        
    def forward(self, x):                                 # x: [bs x nvars x patch_num x seq_num x d_model]
        bs, nvars, pn, sn, d = x.shape
        x = torch.reshape(x, (bs*nvars*pn, sn, d ))       # x: [bs * nvars * patch_num x seq_num x d_model]
        att, _ = self.attention(x,x,x)                    # x: [bs * nvars * patch_num x seq_num x d_model]
        x = x + att                                       # x: [bs * nvars * patch_num x seq_num x d_model]
        x = self.ff(x)                                    # x: [bs * nvars * patch_num x seq_num x d_model]
        x = self.reconstruct(x)                           # x: [bs * nvars * patch_num x seq_num x patch_len]
        x = torch.reshape(x, (bs, nvars, pn, sn, -1))     # x: [bs x nvars x patch_num x seq_num x patch_len] 
        x = x.permute(0,1,3,2,4)                          # x: [bs x nvars x seq_num x patch_num x patch_len]
        
        return x
        
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
class BertMask():
    def __init__(self, bert_ratio=0.3):
        self.bert_ratio = bert_ratio
        
    def forward(self, x):                                        # x : [bs x nvars x patch_num x patch_len]
        shape = x.shape
        mask = (torch.randn(shape) > self.bert_ratio).float()    # mask: [bs x nvars x patch_num x patch_len]
        mask_x = x * mask                                        # mask_x: [bs x nvars x patch_num x patch_len]

        return mask_x, mask
        
class BertHead():
    def __init__(self, d_model, patch_len):
        self.recover = torch.Sequential(
            torch.nn(d_model , 512), torch.relu(),
            torch.nn(512, 1024), torch.relu(),
            torch.nn(1024, 512), torch.relu(),
            torch.nn(512, patch_len)
        )
        
    def forward(self, x):                                        # x: [bs x nvars x d_model x patch_num]
        x = x.permute(0,1,3,2)                                   # x: [bs x nvars x patch_num x d_model]
        x = self.recover(x)                                      # x: [bs x nvars x patch_num x patch_len]

        return x
        
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, multi_patches = False, hybrid = False, nvars = 1, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num                        # self.patch_num : int if not multi_patches, list of int else
        self.patch_len = patch_len                        # self.patch_len : int if not multi_patches, list of int else
        self.multi_patches = multi_patches
        self.hybrid = hybrid
        if hybrid: self.multi_patches = False
        # Input encoding
        q_len = patch_num
        if self.multi_patches:
            self.W_P_list = nn.ModuleList([nn.Linear(patch_length, d_model) for patch_length in self.patch_len])      # [patch_num_i x patch_len_i ] --> [patch_num_i x d_model]
            self.seq_len = q_len
            
            # Positional encoding
            self.W_pos_list = [positional_encoding(pe, learn_pe, each, d_model) for each in q_len]
            self.W_pos_list = [pos_enc.to(torch.device("cuda" if torch.cuda.is_available else "cpu")) for pos_enc in self.W_pos_list]
            final_patch_num = patch_num[0]
            self.reshape_patch_list = nn.ModuleList([nn.Linear(p_num, final_patch_num) for p_num in self.patch_num])  # [patch_num_i x d_model] --> [patch_num x d_model]
            self.combination = nn.Linear(len(q_len), 1)                                                # [patch_num x d_model] --> patch_num x d_model
        else:
            self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
            self.seq_len = q_len

            # Positional encoding
            self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        if self.hybrid: 
            self.new_channel = Combine_Channels(nvars, d_model, 1)
            self.combine_channels = Combine_Channels(nvars + 1, d_model, nvars)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              
        
        if self.multi_patches:                                                   # x: [len_ratio_patches x [bs x nvars x patch_len_i x patch_num_i]] if multi_patches
            n_vars = x[0].shape[1]
            x = [each.permute(0,1,3,2) for each in x]                            # x: [len_ratio_patches x [bs x nvars x patch_num_i x patch_len_i]]
            u_ls = []
            for project, reshape_patch, positional_encoding, each in list(zip(self.W_P_list, self.reshape_patch_list,self.W_pos_list, x)):
                projection = project(each)                                        # projection: [bs x nvars x patch_num_i x d_model]
                emb = torch.reshape(projection, (projection.shape[0]*projection.shape[1], projection.shape[2], projection.shape[3])) # emb: [bs * nvars x patch_num_i x d_model]
                emb = self.dropout(emb + positional_encoding)                     # emb: [bs * nvars x patch_num_i x d_model]
                
                emb = emb.permute(0,2,1)                                         # emb: [bs * nvars x d_model x patch_num_i]
                emb = reshape_patch(emb)                                         # emb: [bs * nvars x d_model x patch_num]
                emb = emb.permute(0,2,1)                                         # emb: [bs * nvars x patch_num x d_model]

                u_ls.append(emb)                                                 # u_ls: [len_ratio_patches x [bs * nvars x patch_num x d_model]]
            
            u = torch.stack(u_ls)                                                # u: [len_ratio_patches x bs * nvars x patch_num x d_model]
            u = u.permute(1,2,3,0)                                               # u: [bs *nvars x patch_num x d_model x len_ratio_patches]
            u = self.combination(u)                                              # u: [bs *nvars x patch_num x d_model x 1]
            u = u.squeeze()                                                      # u: [bs *nvars x patch_num x d_model]

        else:                                                                    # x: [bs x nvars x (seq_num x ) patch_num x patch_len]
            n_vars = x.shape[1]
            # Input encoding
            x = self.W_P(x)                                                      # x: [bs x nvars x (seq_num x ) patch_num x d_model]
            old_shape = x.shape
            if len(x.shape)==5:
                u = torch.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2], x.shape[3], x.shape[4]))  # u: [bs * nvars (* seq_num ) x patch_num x d_model]
            else:
                u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars ( * seq_num ) x patch_num x d_model]
            if self.hybrid:
                tem = torch.reshape(u, old_shape)                                    # tem: [bs x nvars x patch_num x d_model]
                tem1 = self.new_channel(tem)                                         # x: [bs x 1 x (seq_num x) patch_num x d_model]
                u = torch.cat([tem, tem1], dim = 1)                                  # x: [bs x (nvars + 1) x (seq_num x ) patch_num x d_model]           
                old_shape = u.shape
                if len(u.shape)==5:
                    u = torch.reshape(u, (u.shape[0]*u.shape[1]*u.shape[2], u.shape[3], u.shape[4]))  # u: [bs * (nvars + 1) (* seq_num ) x patch_num x d_model]
                else:
                    u = torch.reshape(u, (u.shape[0]*u.shape[1],u.shape[2],u.shape[3]))  # u: [bs * (nvars +1) x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                          # z: [bs * nvars/(nvars + 1) x patch_num x d_model]
        z = torch.reshape(z, old_shape)                                              # z: [bs x nvars/(nvars + 1) x ( seq_num x ) patch_num x d_model]
        if self.hybrid:                                                              # z: [bs x (nvars + 1) x (seq_num x ) patch_num x d_model]
            z = self.combine_channels(z)                                             # z: [bs x nvars x ( seq_num x ) patch_num x d_model]
        if len(old_shape) == 4:
            z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
              
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

