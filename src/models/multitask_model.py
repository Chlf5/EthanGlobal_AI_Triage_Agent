
import torch
import torch.nn as nn

from transformers import (
    ModernBertModel,
    ModernBertPreTrainedModel
)


class EthanMultiTaskBERT(ModernBertPreTrainedModel):

    # Must match the checkpoint parameter prefix: model.*
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        # Shared ModernBERT encoder
        # IMPORTANT: use self.model, not self.bert
        self.model = ModernBertModel(config)

        # Classification heads
        self.resolver_classifier = nn.Linear(
            config.hidden_size,
            config.num_resolver_labels
        )

        self.priority_classifier = nn.Linear(
            config.hidden_size,
            config.num_priority_labels
        )

        # ModernBERT classification dropout
        self.dropout = nn.Dropout(
            getattr(config, "classifier_dropout", 0.0)
        )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        resolver_labels=None,
        priority_labels=None
    ):

        # Shared ModernBERT encoder
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Keep the same first-token pooling used in your
        # previously supplied training implementation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)

        # Resolver prediction
        resolver_logits = self.resolver_classifier(
            pooled_output
        )

        # Priority prediction
        priority_logits = self.priority_classifier(
            pooled_output
        )

        loss = None

        # Multitask loss
        if (
            resolver_labels is not None
            and priority_labels is not None
        ):

            resolver_loss = nn.CrossEntropyLoss()(
                resolver_logits,
                resolver_labels
            )

            priority_loss = nn.CrossEntropyLoss()(
                priority_logits,
                priority_labels
            )

            loss = resolver_loss + priority_loss

        # Inference
        if loss is None:
            return resolver_logits, priority_logits

        # Training
        return loss, resolver_logits, priority_logits
