def tokenize_function(examples,tokenizer,max_input=128,max_target=128):
    model_inputs=tokenizer(
        examples["input_text"],
        max_length=max_input,
        truncation=True,
        padding="max_length"
    )
    labels=tokenizer(examples["target_text"],
                     max_length=max_target,
        truncation=True,
        padding="max_length")
    labels_ids=labels["input_ids"]
    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs