import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with 
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        # embedding model
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(p)
        #size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features

        # linear heads of original CheXbert (need to change to 7)
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    # def forward(self, source_padded, attention_mask):
    #     """ Forward pass of the labeler
    #     @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
    #     @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
    #     @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape 
    #                                         (batch_size, 4) and the last has shape (batch_size, 2)  
    #     """
    #     #shape (batch_size, max_len, hidden_size)
    #     final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
    #     #shape (batch_size, hidden_size)
    #     cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
    #     cls_hidden = self.dropout(cls_hidden)
    #     out = []
    #     # for i in range(14):
    #     for i in range(7):
    #         out.append(self.linear_heads[i](cls_hidden))
    #     return out

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        charbert=False,
        **kwargs
    ):

        # print(charbert)
        if not charbert:
            # can use bert model to output final tensor -> input size = (batch_size, seq_length)
            final_hidden = self.bert(input_ids, attention_mask=attention_mask)[0]
        else:
            # cannot use bert module because input size is not equal to original bert model -> input size = (batch_size, seq_length, max_character_length)
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids[:,:,0].size()
#             print('input_ids', input_ids.size())
#             print('input_shape', input_shape)
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#             print('token_type_ids', token_type_ids.size())

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                # Provided a padding mask of dimensions [batch_size, seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
                if self.bert.config.is_decoder:
                    batch_size, seq_length = input_shape
                    seq_ids = torch.arange(seq_length, device=device)
                    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                    causal_mask = causal_mask.to(
                        attention_mask.dtype
                    )  # causal and attention masks must have same type with pytorch version < 1.3
                    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                else:
                    extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                        input_shape, attention_mask.shape
                    )
                )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            if self.bert.config.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

                if encoder_attention_mask.dim() == 3:
                    encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
                elif encoder_attention_mask.dim() == 2:
                    encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
                else:
                    raise ValueError(
                        "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                            encoder_hidden_shape, encoder_attention_mask.shape
                        )
                    )

                encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                    dtype=next(self.bert.parameters()).dtype
                )  # fp16 compatibility
                encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(self.bert.config.num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = (
                        head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    )  # We can specify head_mask for each layer
                head_mask = head_mask.to(
                    dtype=next(self.bert.parameters()).dtype
                )  # switch to fload if need + fp16 compatibility
            else:
                head_mask = [None] * self.bert.config.num_hidden_layers

            embedding_output = self.bert.embeddings(
                input_ids=input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids
            )
            encoder_outputs = self.bert.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
            )
            sequence_output = encoder_outputs[0]
            pooled_output = self.bert.pooler(sequence_output)

            outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
            final_hidden = outputs[0]

        # use output from bert model as input of linear heads
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        # print(cls_hidden[0].shape)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(7):
            out.append(self.linear_heads[i](cls_hidden))
        # output is probabilities of each class of each label
        return out
