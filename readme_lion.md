python3 launcher.py nanotron=fedf_ablation_3B run=todi_normal ++nanotron.parallelism.tp=4 ++nanotron.parallelism.dp=64 ++nanotron.tokens.train_steps=38200 



python3 launcher.py nanotron=llama3_large_baseline run=todi_normal ++nanotron.optimizer.optimizer_factory.name=lion ++run.paths.nanotron_src=/iopsstor/scratch/cscs/kmatoba/nanotron-lion ++nanotron.optimizer.optimizer_factory.adam_beta2=.99 ++nanotron.optimizer.learning_rate_scheduler.learning_rate=0.0001




python3 launcher.py nanotron=llama3_large_baseline run=todi_normal ++nanotron.optimizer.optimizer_factory.name=lion ++run.paths.nanotron_src=/iopsstor/scratch/cscs/kmatoba/nanotron-lion ++nanotron.optimizer.optimizer_factory.adam_beta2=.99 ++nanotron.optimizer.learning_rate_scheduler.learning_rate=0.0001


    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    name: adamW


 suitable learning rate for Lion is typically 3-10x smaller than that for AdamW. Since the effective weight decay is lr * λ, the value of decoupled weight decay λ used for Lion is 3-10x larger than that for AdamW in order to maintain a similar strength. The initial value, peak value, a

 in Lion, the default values for β1 and β2 are discovered through the program search process and set as 0.9 and 0.99

  learning_rate_scheduler:
    learning_rate: 0.0006
    lr_decay_starting_step: 1000000





python3 launcher.py nanotron=fedf_ablation_3B \
                    run=todi_debug \
                    ++nanotron.parallelism.tp=4 \
                    ++nanotron.parallelism.dp=2 \
                    ++nanotron.general.run=Lion \
                    ++nanotron.optimizer.optimizer_factory.name=lion \
                    ++run.paths.nanotron_src=/iopsstor/scratch/cscs/kmatoba/nanotron-lion \
                    ++nanotron.optimizer.optimizer_factory.adam_beta2=.99 \
                    ++nanotron.optimizer.weight_decay=.5


python3 launcher.py nanotron=fedf_ablation_3B \
                    run=todi_debug \
                    ++nanotron.parallelism.tp=4 \
                    ++nanotron.parallelism.dp=2 \
                    ++nanotron.general.run=baseline




