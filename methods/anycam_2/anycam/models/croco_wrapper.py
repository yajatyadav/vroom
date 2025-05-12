from copy import deepcopy
import torch
import torch.nn as nn
import os

from croco.models.croco import CroCoNet  # noqa

inf = float('inf')


class CroCoExtractor(CroCoNet):
    def __init__(self, **kwargs):
        super(CroCoExtractor, self).__init__(**kwargs)

        self.return_features_at = kwargs.get("return_features_at", [i-1 for i in range(0, self.dec_depth+1, self.dec_depth//4)][1:])

        # Learnable class tokens        
        self.cls_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim)) for _ in range(self.dec_depth)])
        self.cross_attentions = nn.ModuleList([nn.MultiheadAttention(self.dec_embed_dim, num_heads=8) for _ in range(self.dec_depth)])

        # Initialize class tokens
        for cls_token in self.cls_tokens:
            torch.nn.init.normal_(cls_token, std=.02)  # Initialize cls_token



    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        #out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        #out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        out, pos, _ = self._encode_image( torch.cat( (img1,img2), dim=0), do_mask=False, return_all_blocks=return_all_blocks )
        if return_all_blocks:
            out,out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))
            out2 = out2[-1]
        else:
            out,out2 = out.chunk(2, dim=0)
        pos,pos2 = pos.chunk(2, dim=0)            
        return out, out2, pos, pos2


    def encode_image_sequence(self, imgs, return_all_blocks=False):
        n, f, c, h, w = imgs.shape

        out, pos, _ = self._encode_image(imgs.view(n*f, c, h, w), do_mask=False, return_all_blocks=return_all_blocks)
        if return_all_blocks:
            out = [o.unflatten(0, (n, f)) for o in out]
            pos = [p.unflatten(0, (n, f)) for p in pos]
        else:
            out = out.view(n, f, out.shape[1], out.shape[2], out.shape[3])
            pos = pos.view(n, f, pos.shape[1], pos.shape[2])
        return out, pos
    

    def forward(self, img1, img2):
        return_all_blocks = True

        if img2 is not None:
            n, c, h, w = img1.shape

            out, out2, pos, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks)

            decout = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
            pred_cls_tokens = []

            for i in range(self.dec_depth):
                pred_cls_token = self.cross_attentions[i](self.cls_tokens[i].expand(n, -1, -1).permute(1, 0, 2), decout[i].permute(1, 0, 2), decout[i].permute(1, 0, 2))[0]
                pred_cls_token = pred_cls_token.permute(1, 0, 2)
                pred_cls_tokens.append(pred_cls_token)
            
            outputs = []

            for i in self.return_features_at:
                outputs = [torch.cat((pred_cls_tokens[i], decout[i]), dim=1)] + outputs
            
            return outputs
        else:
            n, f, c, h, w = img1.shape
            n_ = n * (f-1)

            out, pos = self.encode_image_sequence(img1, return_all_blocks=False)

            decout = self._decoder(out[:, :-1].flatten(0, 1), pos[:, :-1].flatten(0, 1), None, out[:, 1:].flatten(0, 1), pos[:, 1:].flatten(0, 1), return_all_blocks=return_all_blocks)

            pred_cls_tokens = []

            for i in range(self.dec_depth):
                pred_cls_token = self.cross_attentions[i](self.cls_tokens[i].expand(n_, -1, -1).permute(1, 0, 2), decout[i].permute(1, 0, 2), decout[i].permute(1, 0, 2))[0]
                pred_cls_token = pred_cls_token.permute(1, 0, 2)
                pred_cls_tokens.append(pred_cls_token)
            
            outputs = []

            for i in self.return_features_at:
                outputs = [torch.cat((pred_cls_tokens[i], decout[i]), dim=1)] + outputs

            outputs = [o.unflatten(0, (n, f-1)) for o in outputs]
            
            return outputs