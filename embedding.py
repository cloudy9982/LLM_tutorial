from transformers import GPT2TokenizerFast, GPT2Model

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# 要獲取embedding的內容
text = "This is an example sentence."

# 使用tokenizer將text tokenize並編碼為tensor
inputs = tokenizer(text, return_tensors="pt")
# "pt": 返回PyTorch tensor
# "tf": 返回TensorFlow tensor
# "np": 返回Numpy array
# None: 返回Python list

# 將編碼好的tensor輸入模型進行forward計算
outputs = model(**inputs)

# 模型最後一層的hidden state就是我們想要的token embedding
embeddings = outputs.last_hidden_state

print(embeddings.size())


# Hugging Face 提供的一些 GPT 模型包括：

# GPT：OpenAI 開發的原始生成式預訓練 Transformer 模型。
# GPT-2：由 OpenAI 開發的更大、更強大的 GPT 版本。
# DistilGPT2：GPT-2 的較小、較快的版本，專為更快的推理而設計。
# GPT-Neo：EleutherAI 開發的 GPT-3 的開源版本。
# GPT-J：EleutherAI 和 Constitutional AI 團隊對 GPT-3 的另一個開源版本。
# DialoGPT：針對對話和對話任務進行微調的 GPT 模型。