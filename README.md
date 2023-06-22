# KoAlpaca-PEFT
KoAlapaca + PEFT

# Inference

```python
def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=50,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))
  
gen('건강하게 살기 위한 세 가지 방법은?')

# output
### 질문: 건강하게 살기 위한 세 가지 방법은?

### 답변: 첫째, 적당한 운동을 합시다. 둘째, 항상 긍정적인 마음을 갖는 것이 중요합니다. 셋째, 몸에 좋은 음식을 골고루 섭취하세요. 다만, 지방과 탄수화물은 피해야 합니다.
###
```
