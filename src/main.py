from ingest.pipeline import pipeline
from model.translator import MSLGTranslator
from model.tokenizer import tokenize_function
from training.collator import get_collator
from training.trainer import build_trainer
from evaluation.metrics import build_compute_metrics

def dock(args):
    print("[INFO] inicio ingesta de datos")
    train_ds,val_ds,test_ds=pipeline(args.src_input)
    print("[INFO] inicio modelo y tokenizer")
    model=MSLGTranslator(args.model_name)
    tokenizer=model.tokenizer
    train_ds=train_ds.map(lambda x: tokenize_function(x,tokenizer),batched=True)
    val_ds=val_ds.map(lambda x: tokenize_function(x,tokenizer),batched=True)
    test_ds=test_ds.map(lambda x: tokenize_function(x,tokenizer),batched=True)
    collator=get_collator(tokenizer,model)
    print("[INFO] inicio constructor de entrenamiento")
    trainer=build_trainer(model=model.model,
                          train_dataset=train_ds,
                          val_dataset=val_ds,
                          collator=collator,
                          compute_metrics=build_compute_metrics(tokenizer=tokenizer))
    print("[INFO] inicio entreno")
    trainer.train()
    print("[INFO] resultados")
    results=trainer.evaluate(test_ds)
    out_dir= args.output_dir if args.flag else args.output_dir_1
    print(f"[INFO] iGaurdo en {out_dir}")
    trainer.save_model(out_dir)
    print(results)
    return results