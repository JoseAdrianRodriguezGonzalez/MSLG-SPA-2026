from transformers import Trainer,TrainingArguments
def build_trainer(model,train_dataset,val_dataset,collator,compute_metrics):
    args=TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        num_train_epochs=10,

    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics
    )
