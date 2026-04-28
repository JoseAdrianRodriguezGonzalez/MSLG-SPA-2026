from ingest.pipeline import pipeline
from model.translator import MSLGTranslator
from model.tokenizer import tokenize_function
from training.collator import get_collator
from training.trainer import build_trainer
from evaluation.metrics import build_compute_metrics

def dock(args):
    train_ds,val_ds,test_ds=pipeline(args.src_input)
    model=MSLGTranslator(args.model_name)
    tokenizer=model.tokenizer
    train_ds=train_ds.map(lambda x: tokenize_function(x,tokenizer),batched=True)
    val_ds=val_ds.map(lambda x: tokenize_function(x,tokenizer),batched=True)
    collator=get_collator(tokenizer,model)

    trainer=build_trainer(model=model.model,
                          train_dataset=train_ds,
                          val_dataset=val_ds,
                          collator=collator,
                          compute_metrics=build_compute_metrics(tokenizer=tokenizer))
    trainer.train()
    results=trainer.evaluate(test_ds)
    trainer.save_model(args.output_dir)
    print(results)
    return results