from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

#files = ['clip_train_texts.txt']
#tokenizer.train(files, trainer)

#tokenizer.save('clip_tokenizer.json')

tokenizer = Tokenizer.from_file('clip_tokenizer.json')

output = tokenizer(["i am not a dog, i am a suitcase", "the quick brown fox jumped over the lazy dog"])
print(output.tokens)
print(len(tokenizer))
