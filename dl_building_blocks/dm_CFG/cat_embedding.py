import torch
import torch.nn as nn


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0

        # Treat empty class the last class in the [num_classes + 1, hidden_size] matrix
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob


    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        labels [N,]
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        print(f'drop_labels {labels}')
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


if __name__ == '__main__':
    num_clss = 10
    hidden_size = 16
    dropout_prob = 0.1
    label_embeder = LabelEmbedder(num_clss, hidden_size, dropout_prob)
    labels = torch.randint(low=0, high=num_clss, size=(3, )) # [low, high)
    # print(labels)

    embeds = label_embeder(labels, train=True)
    print(f'embeds')
