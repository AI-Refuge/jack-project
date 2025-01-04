from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./my-model")

model = AutoModelForCausalLM.from_pretrained(
  "./my-model", 
  trust_remote_code=True,
  device_map="auto",
  load_in_4bit=True
).to(device)

sys_prompt = "Imagine you are an intelligent, well-educated, and articulate individual with above-average knowledge and reasoning skills. You have a broad understanding of various subjects, including science, history, literature, and current events. When responding to questions or engaging in discussions, you express yourself clearly and concisely, providing relevant information and insights."

def to_llama3_format(system, user):
    return "".join([
        "<|begin_of_text|>",
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    ])

input_context = to_llama3_format(sys_prompt, "Hi!")
input_ids = tokenizer.encode(input_context, return_tensors="pt")
output = model.generate(input_ids.to(device), max_length=128, temperature=0.7).cpu()
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
