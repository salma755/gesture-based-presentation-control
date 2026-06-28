import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

model_id = "microsoft/Phi-3-mini-4k-instruct"

# 1. تحميل النموذج الأساسي بنفس الإعدادات السابقة
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 2. دمج الأوزان الجديدة المدربة (Lora Weights) مع النموذج الأساسي
model = PeftModel.from_pretrained(base_model, "./phi3_lora_weights")
tokenizer = AutoTokenizer.from_pretrained("./phi3_lora_weights")

# 3. بناء الـ Pipeline الخاص بإنشاء النصوص
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7
)

# 4. قائمة الأسئلة المأخوذة من ملف الـ Notebook الخاص بكِ للاختبار
test_prompts = [
    "Who built Petra?",
    "How to cook Mansaf?",
    "Has Jordan country qualified to World Cup 2026?"
]

# 5. تشغيل الاختبار وطباعة الإجابات المعدلة بناءً على التدريب الجديد
print("\n--- نتائج اختبار النموذج بعد الـ Fine-Tuning ---")
for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    output = generator(messages)
    print(f"\nUser: {prompt}")
    print(f"Phi-3: {output[0]['generated_text']}\n")
    print("-" * 40)