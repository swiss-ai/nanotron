from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM


example_generations = ["The future of AI is",
            # "Passage: Daniel went back to the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:",
            "def fib(n)"]

generator = pipeline('text-generation', model='TinyLlama/TinyLlama_v1.1', device=0) 

"""
# It's a mystery to me why the text generation pipeline code does not produce the same outputs as the one below.
# The one below agreed for me with the nanotron generation script.
for idx, prompt in enumerate(example_generations):
    generated_text = generator(prompt, max_new_tokens=50, num_return_sequences=1, do_sample=False, num_beams=1)
    print(f"Example {idx}: {generated_text[0]['generated_text']}")
"""
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", device_map="auto")

for idx, prompt in enumerate(example_generations):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'].to(model.device), max_new_tokens=50, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Example {idx}: {generated_text}")
    print()